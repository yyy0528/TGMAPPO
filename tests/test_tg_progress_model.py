import copy
import math
import unittest
import torch
from src.buoy_network.progress_model import ProgressModel, survival_terms, prefix_features, pack, evaluate


def data():
    return pack([{"features": torch.randn(3, 14), "length": 3, "horizon": 4, "event": True},
                 {"features": torch.randn(4, 14), "length": 4, "horizon": 4, "event": False}])


class ProgressFunctions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_censored_likelihood_is_observed_survival_not_failure(self):
        logits = torch.full((2, 2, 2), math.log(0.2/0.8), requires_grad=True)
        losses = survival_terms(logits, torch.tensor([2, 2]), torch.tensor([True, False]))
        self.assertAlmostEqual(losses[0].item(), (-math.log(0.8)-2*math.log(0.2))/2, places=6)
        self.assertAlmostEqual(losses[1].item(), -1.5*math.log(0.8), places=6)
        losses.sum().backward()
        self.assertLess(logits.grad[0, 0, 1], 0)
        self.assertGreater(logits.grad[1, 0, 1], 0)

    def test_prefix_features_only_use_current_and_past(self):
        actor = {"node_features": [[0,0,0.5,0], [1,0,0.5,0], [2,0,0,1]],
                 "distances": [[0,1,2], [1,0,1], [2,1,0]], "obs": [[0,0.5,0.5,0]]*3}
        before, state = prefix_features(actor, 0, 2, [0], 0, 4)
        after, _ = prefix_features(actor, 0, 2, [0], 1, 4, state)
        self.assertEqual(before.shape, (14,))
        self.assertAlmostEqual(after[11], 0.25)
        actor["diagnostic_only"] = {"eventual_outcome": 999}
        altered, _ = prefix_features(actor, 0, 2, [0], 0, 4)
        self.assertTrue((before == altered).all())

    def test_causal_gru_and_online_prefix_equivalence(self):
        model, training = ProgressModel(4, 16), data()
        model.fit(training)
        with torch.no_grad():
            model.hazard.weight.normal_(0, 0.1)
        original = model(training["features"], training["lengths"])
        changed = training["features"].clone()
        changed[:, 2:] += 100
        altered = model(changed, training["lengths"])
        self.assertTrue(torch.allclose(original[:, :2], altered[:, :2], atol=1e-6))
        hidden = torch.zeros(1, 2, 16)
        for step in range(2):
            value, hidden = model.step(training["features"][:, step], hidden)
            self.assertTrue(torch.allclose(value, original[:, step], atol=1e-6))

    def test_packet_weighting_padding_and_real_gradient_update(self):
        training = data()
        model = ProgressModel(4, 16)
        model.fit(training)
        normalizer = {key: value.clone() for key,value in model.named_buffers()}
        initial = evaluate(model, training)
        self.assertAlmostEqual(initial["survival_nll"], initial["prior_survival_nll"], places=6)
        before = model.hazard.weight.detach().clone()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        for _ in range(10):
            optimizer.zero_grad()
            loss = survival_terms(model(training["features"], training["lengths"]), training["lengths"], training["events"]).mean()
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            optimizer.step()
        self.assertFalse(torch.equal(before, model.hazard.weight))
        self.assertLess(evaluate(model, training)["survival_nll"], initial["survival_nll"])
        for key,value in model.named_buffers():
            self.assertTrue(torch.equal(value, normalizer[key]))

    def test_unknown_window_results_do_not_count_as_zero_brier(self):
        training = data()
        model = ProgressModel(4, 16)
        model.fit(training)
        training["events"][:] = False
        training["lengths"][:] = 2
        with self.assertRaisesRegex(ValueError, "No identifiable"):
            evaluate(model, training)


if __name__ == "__main__":
    unittest.main(verbosity=2)
