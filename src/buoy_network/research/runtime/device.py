"""Import torch only for device probing, never for argument parsing/imports."""

import importlib.util


def resolve_device(requested: str) -> str:
    if requested not in ("auto", "cpu", "cuda"):
        raise ValueError("Unknown device: " + requested)
    if requested == "cpu":
        return "cpu"
    available = False
    if importlib.util.find_spec("torch") is not None:
        import torch
        available = torch.cuda.is_available()
    if requested == "cuda" and not available:
        raise ValueError("CUDA requested, but this Python environment has no usable CUDA")
    return "cuda" if available else "cpu"
