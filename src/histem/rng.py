from collections.abc import Iterator
from contextlib import contextmanager

import torch


@contextmanager
def seeded(seed: int) -> Iterator[None]:
    """Run with torch's global RNG seeded, then restore the outer RNG state."""
    device = "mps" if torch.backends.mps.is_available() else "cuda"
    with torch.random.fork_rng(device_type=device):
        torch.manual_seed(seed)
        yield
