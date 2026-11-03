"""PopArt invariants, optimizer ownership, and actual original-TG PPO."""
import contextlib
import copy
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT,ROOT/"src",ROOT/"tests"):
    sys.path.insert(0,str(path))
import numpy as np
import torch
from onpolicy.algorithms.utils.popart import PopArt, popart_rollout, remap_popart_values
from buoy_network.wm_runtime import tensor_digest
from buoy_network.task_reward_audit import audit_reward_csv
from buoy_network.task_reward import reward_contract
from test_legacy_targeted_fixes import make_environment, make_learner
from test_legacy_wm_actor import run, checkpoint as wm_checkpoint
from test_legacy_pm_reward import checkpoint as pm_checkpoint


class PopArtFunctions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_parameter_identity_output_preservation_and_real_optimizer_step(self):
        torch.manual_seed(11)
        model = PopArt(3,2)
        optimizer = torch.optim.Adam(model.parameters(),lr=.01)
        ids = {k:id(v) for k,v in model.named_parameters()}
        x = torch.randn(8,3)
        for targets in (torch.randn(20,2)*5+50, torch.ones(20,2)*-1000, torch.randn(20,2)*.02):
            with torch.no_grad():
                before = model.denormalize(model(x)).copy()
            model.update(targets)
            with torch.no_grad():
                after = model.denormalize(model(x))
            np.testing.assert_allclose(after,before,rtol=1e-3,atol=1e-4)
            self.assertEqual(ids,{k:id(v) for k,v in model.named_parameters()})
            self.assertFalse(any(v.requires_grad for k,v in model.named_parameters() if k not in ("weight","bias")))
            weight = model.weight.detach().clone()
            optimizer.zero_grad()
            (model(x)-model.normalize(torch.zeros(8,2))).square().mean().backward()
            optimizer.step()
            self.assertFalse(torch.equal(weight,model.weight))
        self.assertEqual(int(optimizer.state[model.weight]["step"]),3)

    def test_invalid_targets_do_not_mutate_and_roundtrip_is_finite(self):
        model = PopArt(2,1)
        for targets in (torch.empty(0,1),torch.tensor([[float("nan")]]),torch.tensor([[float("inf")]])):
            before = tensor_digest(model.state_dict())
            with self.assertRaises(ValueError):
                model.update(targets)
            self.assertEqual(before,tensor_digest(model.state_dict()))
        model.update(torch.tensor([[-30.],[-5.],[5.],[20.]]))
        x = torch.tensor([[-20.],[0.],[10.]])
        np.testing.assert_allclose(model.denormalize(model.normalize(x)),x.numpy(),atol=1e-5)

    def test_rollout_fixed_stats_value_clip_coordinates_and_exception_cleanup(self):
        model = PopArt(2,1)
        trainer = SimpleNamespace(_use_popart=True,_use_value_active_masks=True,value_normalizer=model)
        old_values = torch.tensor([[.2],[-.4]])
        raw = model.denormalize(old_values).copy()
        b = SimpleNamespace(returns=np.array([[[[4.],[400.]]],[[[6.],[600.]]],[[[0.],[0.]]]],dtype=np.float32),
            active_masks=np.array([[[[1.],[0.]]],[[[1.],[0.]]],[[[0.],[0.]]]],dtype=np.float32))
        with popart_rollout(trainer,b):
            mean,var = model.debiased_mean_var()
            self.assertAlmostEqual(float(mean),5.,places=5)
            self.assertAlmostEqual(float(var),1.,places=5)
            remapped = remap_popart_values(trainer,old_values)
            np.testing.assert_allclose(model.denormalize(remapped),raw,atol=1e-6)
            with self.assertRaises(ValueError):
                with popart_rollout(trainer,b):
                    pass
        self.assertIsNone(trainer._popart_old_stats)
        with self.assertRaises(RuntimeError):
            with popart_rollout(trainer,b):
                raise RuntimeError("test cleanup")
        self.assertIsNone(trainer._popart_old_stats)
        np.testing.assert_array_equal(old_values.numpy(),np.array([[.2],[-.4]],dtype=np.float32))

    def test_empty_rollout_does_not_update_stats(self):
        model = PopArt(2,1)
        trainer = SimpleNamespace(_use_popart=True,_use_value_active_masks=True,value_normalizer=model)
        b = SimpleNamespace(returns=np.zeros((3,1,2,1),np.float32),active_masks=np.zeros((3,1,2,1),np.float32))
        before = tensor_digest(model.state_dict())
        with popart_rollout(trainer,b):
            self.assertEqual(before,tensor_digest(model.state_dict()))


class PopArtModules(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_original_tg_multiple_epochs_fixed_stats_head_learns_and_buffer_is_unchanged(self):
        with tempfile.TemporaryDirectory() as temp:
            env = make_environment(temp,length=4)
            env.all_args.ppo_epoch = 2
            env.all_args.num_episodes = 2
            learner = make_learner(env)
            updates,steps = [],[]
            norm = learner.trainer.value_normalizer
            update = norm.update
            def checked_update(targets):
                updates.append(len(targets))
                update(targets)
            norm.update = checked_update
            optimizer = learner.policy.critic_optimizer
            original_step = optimizer.step
            def step(*args,**kwargs):
                self.assertIn(norm.weight,optimizer.state if optimizer.state else {norm.weight:None})
                registered = {id(p) for g in optimizer.param_groups for p in g["params"]}
                self.assertIn(id(norm.weight),registered)
                snapshot = {k:v.clone() for k,v in norm.state_dict().items() if k not in ("weight","bias")}
                before = norm.weight.detach().clone()
                result = original_step(*args,**kwargs)
                self.assertTrue(all(torch.equal(snapshot[k],norm.state_dict()[k]) for k in snapshot))
                steps.append(not torch.equal(before,norm.weight))
                return result
            optimizer.step = step
            train = learner.trainer.train
            def checked_train(buffer):
                old_values,old_returns = buffer.value_preds.copy(),buffer.returns.copy()
                result = train(buffer)
                np.testing.assert_array_equal(old_values,buffer.value_preds)
                np.testing.assert_array_equal(old_returns,buffer.returns)
                return result
            learner.trainer.train = checked_train
            run(env,learner)
            self.assertEqual(len(updates),2)
            self.assertEqual(len(steps),8)
            self.assertTrue(all(steps))
            self.assertEqual([r["critic_head_optimizer_steps"] for r in learner.training_records],[4,8])
            self.assertIsNone(learner.trainer._popart_old_stats)

    def test_shared_nongraph_mappo_popart_path_has_real_updates(self):
        from onpolicy.algorithms.MAPPOPolicy import R_MAPPOPolicy
        from onpolicy.algorithms.mappo import R_MAPPO
        from onpolicy.utils.shared_buffer import SharedReplayBuffer
        with tempfile.TemporaryDirectory() as temp:
            env = make_environment(temp,length=4)
            args = env.all_args
            obs,act = env.observation_space[0],env.action_space[0]
            policy = R_MAPPOPolicy(args,obs,obs,act)
            trainer = R_MAPPO(args,policy)
            b = SharedReplayBuffer(100,args,25,obs,obs,act)
            rng = np.random.RandomState(11)
            b.obs[:] = rng.normal(size=b.obs.shape)
            b.share_obs[:] = b.obs
            with torch.no_grad():
                for t in range(4):
                    values,actions,logp,_,_ = policy.get_actions(b.share_obs[t,0],b.obs[t,0],b.rnn_states[t,0],
                        b.rnn_states_critic[t,0],b.masks[t,0],b.available_actions[t,0])
                    b.value_preds[t,0],b.actions[t,0],b.action_log_probs[t,0] = values.numpy(),actions.numpy(),logp.numpy()
            b.rewards[:] = -.1
            b.masks[-1] = 0
            b.compute_returns(np.zeros_like(b.value_preds[-1]),trainer.value_normalizer)
            with contextlib.redirect_stdout(io.StringIO()):
                result = trainer.train(b)
            head = policy.critic.v_out
            self.assertEqual(int(policy.critic_optimizer.state[head.weight]["step"]),2)
            self.assertTrue(all(np.isfinite(float(x)) for x in result.values()))
            self.assertIsNone(trainer._popart_old_stats)


class PopArtIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_default_popart_wm_pm_rewards_gae_and_frozen_reload(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            wm,pm = root/"wm.pt",root/"pm.pt"
            wm_checkpoint(wm)
            pm_checkpoint(pm)
            def make(name):
                env = make_environment(root/name,length=4)
                args = env.all_args
                args.tg_reward_version = "task_v2"
                args.use_wm_actor, args.wm_actor_checkpoint = True,str(wm)
                args.use_pm_reward, args.pm_reward_checkpoint = True,str(pm)
                return env,make_learner(env)
            env,learner = make("train")
            self.assertTrue(learner.trainer._use_popart)
            run(env,learner)
            self.assertEqual(learner.training_records[0]["critic_head_optimizer_steps"],2)
            self.assertTrue(learner.wm_update_records[0]["wm_frozen"])
            learner.pm_runtime.assert_frozen()
            audit_reward_csv(root/"train/moves_0/logs",reward_contract(env.all_args))
            policy = root/"train/moves_0/checkpoints/tg_policy_latest.pt"
            env2,restored = make("eval")
            env2.all_args.tg_evaluate = True
            restored.load_policy(policy)
            restored.train = lambda: self.fail("Frozen evaluation called train")
            run(env2,restored)
            report = json.loads((root/"eval/moves_0/logs/TG_policy_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["source_training_implementation"],"tg_ppo_popart_v2")
            self.assertTrue(report["actor_unchanged"] and report["critic_unchanged"])
            self.assertEqual(report["optimizer_updates"],0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
