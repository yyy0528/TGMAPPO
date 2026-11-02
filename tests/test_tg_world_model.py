import copy
import tempfile
import unittest
from pathlib import Path

import torch
from src.buoy_network.world_model import GraphWorldModel, encode_sample, batch, evaluate


def sample(action=1, ack=True):
    nodes = [[0, 0, 0.5, 0], [1, 0, 0.5, 0], [2, 0, 0, 1]]
    return {"agent_id": 0, "inputs": {"candidate_action": action, "history": [
        {"step": 0, "node_features": nodes, "distances": [[0,1,2],[1,0,1],[2,1,0]],
         "own_obs": [0,0.5,0.4,0], "own_previous_action": None}]},
        "targets": {"next_node_features": [[0.1,0,0,0],[1,0,1,0],[2,0,0,1]],
                    "ack": ack, "ack_loss_mask": action != 3}}


def data():
    records = [encode_sample(sample()), encode_sample(sample(3, None))]
    return {key: torch.stack([row[key] for row in records]) for key in records[0]}


class WorldModelFunctions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_future_targets_do_not_change_inputs_and_wait_is_masked(self):
        original = sample()
        modified = copy.deepcopy(original)
        modified["targets"]["next_node_features"][0] = [999,999,999,1]
        modified["targets"]["ack"] = False
        before, after = encode_sample(original), encode_sample(modified)
        for key in ("nodes", "distances", "context", "candidate", "length", "agent", "target"):
            self.assertTrue(torch.equal(before[key], after[key]), key)
        wait = encode_sample(sample(3, None))
        self.assertFalse(wait["ack_mask"])
        self.assertFalse(wait["queue_mask"][1])
        self.assertEqual(wait["length"], 1)

    def test_topology_is_independent_of_current_candidate_action(self):
        torch.manual_seed(1)
        model = GraphWorldModel(16)
        training = data()
        model.fit_normalization(training)
        changed = {key: value.clone() for key,value in training.items()}
        changed["candidate"].fill_(0.9)
        changed["target"].fill_(2)
        before, after = model(training), model(changed)
        self.assertTrue(torch.equal(before["motion"], after["motion"]))
        self.assertFalse(torch.equal(before["ack_logit"], after["ack_logit"]))

    def test_real_gradients_checkpoint_and_evaluation_preserve_normalization(self):
        training = data()
        model = GraphWorldModel(16)
        model.fit_normalization(training)
        norm = {key:value.clone() for key,value in model.named_buffers()}
        before = torch.cat([p.detach().flatten() for p in model.parameters()])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        for _ in range(3):
            optimizer.zero_grad()
            loss = model.loss(batch(training, torch.arange(2)))
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            optimizer.step()
        self.assertFalse(torch.equal(before, torch.cat([p.detach().flatten() for p in model.parameters()])))
        report = evaluate(model, training)
        self.assertEqual(report["attempts"], 1)
        for key, value in model.named_buffers():
            self.assertTrue(torch.equal(value, norm[key]))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"weights.pt"
            torch.save(model.state_dict(), path)
            restored = GraphWorldModel(16)
            restored.load_state_dict(torch.load(path, weights_only=True))
            self.assertEqual(evaluate(restored, training), report)


if __name__ == "__main__":
    unittest.main()
