from collections.abc import Iterator
from contextlib import contextmanager

import torch


@contextmanager
def seeded(seed: int) -> Iterator[None]:
    """Run with torch's global RNG seeded, then restore the outer RNG state."""
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        yield
