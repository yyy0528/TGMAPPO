"""WM sensitivity scalar: exact default, mask/gradient behavior, policy I/O."""
import json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'tests')]
import torch
from onpolicy.algorithms.utils.wm_fusion import residual_scale
from onpolicy.algorithms.utils.distributions import FixedCategorical
from test_legacy_targeted_fixes import make_environment,make_learner
from test_legacy_wm_actor import checkpoint,run


class ResidualScaleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_scale_contract_and_cli_whitelist(self):
        from src.legacy_cli import reject_research_options
        self.assertEqual(residual_scale(SimpleNamespace()),1.)
        reject_research_options(['--wm_actor_residual_scale','2'])
        for x in [float('nan'),float('inf'),-1.]:
            with self.assertRaises(ValueError):residual_scale(SimpleNamespace(use_wm_actor=True,wm_actor_residual_scale=x))
        with self.assertRaises(ValueError):residual_scale(SimpleNamespace(wm_actor_residual_scale=0.))
        self.assertEqual(residual_scale(SimpleNamespace(use_wm_actor=True,wm_actor_residual_scale=0.)),0.)

    def test_exact_default_and_scaled_logits_masks_and_gradients(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);checkpoint(root/'wm.pt');env=make_environment(root/'run');env.all_args.use_wm_actor=True;env.all_args.wm_actor_checkpoint=str(root/'wm.pt')
            model=make_learner(env);actor=model.policy.actor
            with torch.no_grad():actor.wm_fusion[-1].weight.fill_(.3)
            features=torch.randn(2,actor.hidden_size);wm=torch.randn(2,26,3);mask=torch.ones(2,26);mask[:,3]=0
            def expected(scale):
                r=actor._wm_residual(features,wm,mask>0)
                logits=actor.act.action_out.linear(features)+(r if scale==1 else r*scale)
                logits=logits.masked_fill(mask<=0,float('-inf'));logits=logits-logits.max(-1,keepdim=True)[0]
                return FixedCategorical(logits=logits)
            self.assertTrue(torch.equal(actor._wm_distribution(features,mask,wm).logits,expected(1).logits))
            for scale in [0.,2.]:
                actor.wm_residual_scale=scale;actor.zero_grad();dist=actor._wm_distribution(features,mask,wm)
                torch.testing.assert_close(dist.logits,expected(scale).logits,rtol=0,atol=0)
                self.assertTrue((dist.probs[:,3]==0).all());(-dist.log_probs(torch.tensor([[1],[2]])).mean()).backward()
                norms=sum(float(p.grad.abs().sum()) for p in actor.wm_fusion.parameters() if p.grad is not None)
                self.assertEqual(norms,0.) if scale==0 else self.assertGreater(norms,0.)

    def test_native_ppo_checkpoint_scale_and_old_default_compatibility(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);checkpoint(root/'wm.pt')
            def build(name,scale):
                env=make_environment(root/name);env.all_args.use_wm_actor=True;env.all_args.wm_actor_checkpoint=str(root/'wm.pt');env.all_args.wm_actor_residual_scale=scale
                return env,make_learner(env)
            env,model=build('run',2.);run(env,model);path=root/'run/moves_0/checkpoints/tg_policy_latest.pt'
            payload=torch.load(path,weights_only=True);self.assertEqual(payload['wm_actor_residual_scale'],2.)
            _,good=build('good',2.);good.load_policy(path)
            _,bad=build('bad',1.)
            with self.assertRaisesRegex(ValueError,'residual scale'):bad.load_policy(path)
            good.all_args.wm_actor_residual_scale=1.
            with self.assertRaisesRegex(ValueError,'residual scale'):good.load_policy(path)
            env,old=build('default',1.);run(env,old);cp=root/'default/moves_0/checkpoints/tg_policy_latest.pt'
            legacy=torch.load(cp,weights_only=True);legacy.pop('wm_actor_residual_scale');torch.save(legacy,root/'legacy.pt')
            _,loaded=build('legacy_load',1.);loaded.load_policy(root/'legacy.pt')


if __name__=='__main__':unittest.main()
