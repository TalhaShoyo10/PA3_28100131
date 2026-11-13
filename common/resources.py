import time
from pathlib import Path

import torch


class Timer:
    def __init__(self):
        self.elapsed_s = None

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.elapsed_s = time.perf_counter() - self.start


def peak_mem_gb():
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / 1024**3


def reset_peak_mem():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def count_params(model, trainable_only=False):
    seen = set()
    total = 0
    for p in model.parameters():
        if id(p) in seen or (trainable_only and not p.requires_grad):
            continue
        seen.add(id(p))
        total += p.numel()
    return total


def path_size_bytes(path):
    path = Path(path)
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
