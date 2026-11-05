"""Training-graph and RNG contracts for the observational PPO probes."""
import copy
import random
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts')]
from diagnose_tg_credit import RolloutCapture
from diagnose_tg_pm_updates import probe


class Distribution(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 3)

    def forward(self, x):
        return torch.distributions.Categorical(logits=self.linear(x))


class Actor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.distribution = Distribution()
        self.act = SimpleNamespace(action_out=self.distribution)
        self.failure = False

    def forward(self, x, *args, **kwargs):
        result = self.distribution(torch.as_tensor(x, dtype=torch.float32))
        if self.failure:
            raise RuntimeError('probe forward failed')
        return result.probs.argmax(-1)


class ObserverFunctions(unittest.TestCase):
    def test_training_hook_preserves_lazy_graph_and_exact_optimizer_update(self):
        torch.manual_seed(22)
        original = Actor()
        observed = copy.deepcopy(original)
        capture = RolloutCapture(observed)
        x = torch.tensor([[1., 2.], [3., 4.]])
        for actor in (original, observed):
            optimizer = torch.optim.Adam(actor.parameters(), lr=.001)
            dist = actor.distribution(x)
            self.assertNotIn('probs', dist.__dict__)
            loss = -dist.log_prob(torch.tensor([0, 2])).mean()-.01*dist.entropy().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        for a, b in zip(original.parameters(), observed.parameters()):
            self.assertTrue(torch.equal(a, b))
        self.assertIsNone(capture.last)
        capture.close()


class ObserverModules(unittest.TestCase):
    def test_probe_preserves_rng_mode_weights_and_last_behavior_probabilities(self):
        actor = Actor()
        capture = RolloutCapture(actor)
        marker = np.array([.2, .3, .5])
        capture.last = marker
        x = np.array([[1., 2.], [3., 4.]], dtype=np.float32)
        inputs = [x]+[np.ones((2, 1)) for _ in range(6)]
        states = [p.detach().clone() for p in actor.parameters()]
        rng = torch.get_rng_state().clone()
        python_rng, numpy_rng = random.getstate(), np.random.get_state()
        probabilities = probe(actor, capture, inputs)
        np.testing.assert_allclose(probabilities.sum(-1), 1., atol=1e-7)
        self.assertIs(capture.last, marker)
        self.assertTrue(actor.training)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(python_rng, random.getstate())
        np.testing.assert_equal(numpy_rng, np.random.get_state())
        for p, before in zip(actor.parameters(), states):
            self.assertTrue(torch.equal(p, before))
        capture.close()

    def test_forward_failure_restores_mode_and_hook_context(self):
        actor = Actor()
        capture = RolloutCapture(actor)
        marker = np.array([1.])
        capture.last = marker
        actor.failure = True
        with self.assertRaisesRegex(RuntimeError, 'probe forward failed'):
            probe(actor, capture, [np.ones((1, 2))]+[np.ones((1, 1)) for _ in range(6)])
        self.assertTrue(actor.training)
        self.assertIs(capture.last, marker)
        capture.close()


if __name__ == '__main__':
    unittest.main(verbosity=2)
