"""Shared execution helpers for the self-learning experiments."""
import random
from pathlib import Path
import torch
import numpy as np

ROOT = Path(__file__).resolve().parent

def select_device():
    # CUDA is preferred on Colab; MPS supports Apple Silicon; CPU is the fallback.
    return torch.device("cuda" if torch.cuda.is_available() else
                        "mps" if torch.backends.mps.is_available() else "cpu")

def loss_weight(criterion, targets):
    # Weighted CE divides by the sum of target weights, not batch length.
    if isinstance(criterion, torch.nn.CrossEntropyLoss) and criterion.weight is not None:
        return float(criterion.weight[targets].sum().item())
    return targets.numel()

def rng_state():
    n = np.random.get_state()
    state = {"torch": torch.get_rng_state(), "python": random.getstate(),
             "numpy": (n[0], n[1].tolist(), n[2], n[3], n[4])}
    if torch.cuda.is_available(): state["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available(): state["mps"] = torch.mps.get_rng_state()
    return state

def restore_rng(state):
    torch.set_rng_state(state["torch"].cpu())
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([v.cpu() for v in state["cuda"]])
    if "mps" in state and torch.backends.mps.is_available(): torch.mps.set_rng_state(state["mps"].cpu())


def clean_validation(train, val):
    """Remove exact training-image matches from validation; preserve official test."""
    import hashlib
    from torch.utils.data import Subset
    hashes = {hashlib.sha256(image.tobytes()).digest() for image in train.imgs}
    keep = [i for i, image in enumerate(val.imgs)
            if hashlib.sha256(image.tobytes()).digest() not in hashes]
    print(f"Validation deduplication: retained {len(keep)} / {len(val)} images")
    return Subset(val, keep)
