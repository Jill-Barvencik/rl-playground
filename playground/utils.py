from typing import Iterable, List
import numpy as np
import random
import os

def smooth(x: Iterable[float], window: int = 50) -> np.ndarray:
    x = np.asarray(list(x), dtype=float)
    if len(x) == 0:
        return x
    w = max(1, int(window))
    if w == 1:
        return x
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode="same")

def q_to_policy(Q: np.ndarray) -> np.ndarray:
    """Liefert die greedy-Policy (argmax) je Zustand."""
    return np.argmax(Q, axis=1)

def seed_everything(seed: int = 42):
    np.random.seed(seed)
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:
        pass
    os.environ["PYTHONHASHSEED"] = str(seed)
