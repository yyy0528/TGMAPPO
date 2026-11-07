"""Device selection must preserve old CPU runs and reject silent fallback."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from buoy_network.compute_device import resolve_tg_device


class DeviceSelection(unittest.TestCase):
    def test_default_stays_cpu_even_when_cuda_available(self):
        with patch("torch.cuda.is_available", return_value=True):
            self.assertEqual(str(resolve_tg_device(SimpleNamespace())), "cpu")

    def test_explicit_cpu_needs_no_cuda(self):
        with patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(str(resolve_tg_device(SimpleNamespace(tg_device="cpu"))), "cpu")

    def test_explicit_cuda_and_unavailable_error(self):
        args = SimpleNamespace(tg_device="cuda")
        with patch("torch.cuda.is_available", return_value=True):
            self.assertEqual(str(resolve_tg_device(args)), "cuda")
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(ValueError, "cannot use CUDA"):
                resolve_tg_device(args)

    def test_unknown_programmatic_device_rejected(self):
        with self.assertRaises(ValueError):
            resolve_tg_device(SimpleNamespace(tg_device="auto"))


class NormalizerRegistration(unittest.TestCase):
    def check_device(self, device):
        from onpolicy.algorithms.utils.popart import PopArt
        from onpolicy.utils.valuenorm import ValueNorm
        torch.set_num_threads(1)
        head = PopArt(4, 1, device=torch.device(device))
        expected = {"weight", "bias", "mean", "mean_sq", "stddev", "debiasing_term"}
        self.assertEqual(set(head.state_dict()), expected)
        self.assertEqual(set(dict(head.named_parameters())), expected)
        self.assertTrue(all(p.device.type == device for p in head.parameters()))
        opt = torch.optim.Adam(head.parameters(), lr=.001)
        before = head.weight.detach().clone()
        head(torch.ones(3, 4, device=device)).sum().backward()
        opt.step()
        self.assertFalse(torch.equal(before, head.weight))
        norm = ValueNorm(1, device=torch.device(device))
        self.assertEqual(set(norm.state_dict()), {"running_mean", "running_mean_sq", "debiasing_term"})
        norm.update(torch.ones(3, 1, device=device))
        restored = ValueNorm(1, device=torch.device(device))
        restored.load_state_dict(norm.state_dict())
        self.assertTrue(torch.equal(norm.running_mean, restored.running_mean))

    def test_cpu_registered_and_optimizer_updates(self):
        self.check_device("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable in this interpreter")
    def test_cuda_registered_and_optimizer_updates(self):
        self.check_device("cuda")


if __name__ == "__main__":
    unittest.main()
