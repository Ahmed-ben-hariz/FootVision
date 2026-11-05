"""Automatic device selection: MPS first, then CPU. CUDA is not assumed."""
import os
import torch

# Let unsupported ops fall back to CPU instead of crashing on MPS.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def get_device(preference: str = "auto") -> torch.device:
    if preference == "cpu":
        return torch.device("cpu")
    if preference in ("auto", "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if preference == "mps":
        print("MPS requested but unavailable; falling back to CPU.")
    return torch.device("cpu")


if __name__ == "__main__":
    import torchvision
    print("torch", torch.__version__, "| torchvision", torchvision.__version__)
    print("MPS built:", torch.backends.mps.is_built(),
          "| MPS available:", torch.backends.mps.is_available())
    print("Selected device:", get_device())
