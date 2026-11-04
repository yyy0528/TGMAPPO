import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT,ROOT/'src',ROOT/'tests'):
    sys.path.insert(0,str(path))
from scripts.diagnose_tg_credit import RolloutCapture
from onpolicy.algorithms.utils.distributions import Categorical
from buoy_network.wm_runtime import tensor_digest
from test_legacy_targeted_fixes import make_environment,make_learner
from test_legacy_wm_actor import run


class CaptureModule(unittest.TestCase):
    def test_training_graph_keeps_lazy_probability_cache_unread(self):
        layer = Categorical(4,3)
        actor = SimpleNamespace(act=SimpleNamespace(action_out=layer))
        capture = RolloutCapture(actor)
        try:
            dist = layer(torch.ones(2,4))
            self.assertNotIn('probs',dist.__dict__)
            self.assertIsNone(capture.last)
            with torch.no_grad():
                dist = layer(torch.ones(2,4))
            self.assertIn('probs',dist.__dict__)
            self.assertIsNotNone(capture.last)
        finally:
            capture.close()


class CaptureIntegration(unittest.TestCase):
    def test_actual_ppo_weights_and_losses_are_bitwise_unchanged(self):
        torch.set_num_threads(1)
        results = []
        with tempfile.TemporaryDirectory() as temp:
            for enabled in (False,True):
                random.seed(33)
                np.random.seed(33)
                torch.manual_seed(33)
                env = make_environment(Path(temp)/str(enabled),length=4)
                env.all_args.num_episodes = 2
                env.all_args.ppo_epoch = 4
                learner = make_learner(env)
                capture = RolloutCapture(learner.policy.actor) if enabled else None
                try:
                    run(env,learner)
                finally:
                    if capture:
                        capture.close()
                results.append((tensor_digest(learner.policy.actor.state_dict()),
                                tensor_digest(learner.policy.critic.state_dict()),learner.training_records))
        self.assertEqual(results[0],results[1])


if __name__ == '__main__':
    unittest.main(verbosity=2)
