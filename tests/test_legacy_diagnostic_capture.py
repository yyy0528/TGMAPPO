"""Torch 1.8 read-only hooks: distribution capture must not perturb greedy replay."""
import sys
import tempfile
import unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT,ROOT/"src",ROOT/"scripts",ROOT/"tests"):
    sys.path.insert(0,str(directory))
import numpy as np
import torch
from diagnose_tg_policy import DistributionCapture
from buoy_network.wm_runtime import tensor_digest
from test_legacy_targeted_fixes import make_environment,make_learner
from test_legacy_wm_actor import checkpoint


class CaptureModules(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_baseline_capture_is_read_only_and_detaches_cleanly(self):
        with tempfile.TemporaryDirectory() as temp:
            env = make_environment(temp)
            env.packet_generation()
            learner = make_learner(env)
            learner.warmup()
            actor,buffer = learner.policy.actor,learner.buffer
            actor.eval()
            capture = DistributionCapture(actor)
            state, rng = tensor_digest(actor.state_dict()),torch.get_rng_state().clone()
            def forward():
                return actor(buffer.obs[0,0],buffer.node_obs[0,0],buffer.adj[0,0],buffer.agent_id[0,0],
                    buffer.rnn_states[0,0],buffer.masks[0,0],available_actions=buffer.available_actions[0,0],deterministic=True)
            with torch.no_grad():
                result = forward()
            np.testing.assert_array_equal(result[0].numpy()[:,0],capture.last.argmax(-1))
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            self.assertEqual(state,tensor_digest(actor.state_dict()))
            capture.close()
            capture.last = None
            with torch.no_grad():
                forward()
            self.assertIsNone(capture.last)

    def test_wm_distribution_capture_uses_cached_inputs_without_sampling(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"wm.pt"
            checkpoint(path)
            env = make_environment(temp)
            env.all_args.use_wm_actor = env.all_args.tg_evaluate = True
            env.all_args.wm_actor_checkpoint = str(path)
            env.packet_generation()
            learner = make_learner(env)
            learner.warmup()
            capture = DistributionCapture(learner.policy.actor)
            rng = torch.get_rng_state().clone()
            result = learner.collect(0)
            np.testing.assert_array_equal(result[1][0,:,0],capture.last.argmax(-1))
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            self.assertTrue(learner.buffer.wm_feature_ready[0])
            learner.wm_runtime.assert_frozen()
            capture.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
