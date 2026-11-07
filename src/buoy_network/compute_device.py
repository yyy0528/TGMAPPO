"""Explicit opt-in CUDA selection for the original TG entry point."""
import torch


def resolve_tg_device(args):
    name = getattr(args, "tg_device", "cpu")
    if name not in ("cpu", "cuda"):
        raise ValueError("tg_device must be cpu or cuda")
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("--tg_device cuda requested, but this Python/PyTorch cannot use CUDA")
    return torch.device(name)
