"""Environment setup: Python, PyTorch, MPS/CPU device selection."""
import sys
import torch


def get_device():
    import os
    if os.environ.get("FORCE_CPU", "0") == "1":
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def info():
    device = get_device()
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "mps_available": bool(torch.backends.mps.is_available()),
        "cuda_available": bool(torch.cuda.is_available()),
        "device": str(device),
    }


if __name__ == "__main__":
    for k, v in info().items():
        print(f"{k}: {v}")
