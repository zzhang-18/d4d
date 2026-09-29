"""Small helpers for the optimizer."""

from __future__ import annotations

import random
from collections import deque
from collections.abc import Sequence

import numpy as np
import torch

__all__ = ["MovingAverage", "maybe_clamp", "safe_cat", "safe_stack", "seed_everything"]


class MovingAverage:
    """Fixed-window mean over a stream of scalars."""

    def __init__(self, window_size: int) -> None:
        self.sum: float = 0.0
        self.values: deque[float] = deque(maxlen=window_size)

    def clear(self) -> None:
        self.sum = 0.0
        self.values.clear()

    def add(self, value: float) -> None:
        self.sum += value
        if len(self.values) == self.values.maxlen:
            self.sum -= self.values[0]
            self.values.popleft()
        self.values.append(value)

    def mean(self) -> float:
        return self.sum / len(self.values) if len(self.values) > 0 else 0.0


def maybe_clamp(
    x: torch.Tensor, min: float | None = None, max: float | None = None
) -> torch.Tensor:
    """``x.clamp(min, max)``, or ``x`` itself when both bounds are ``None``. Keeps ``x``'s shape."""
    if min is None and max is None:
        return x
    return x.clamp(min=min, max=max)


def safe_cat(
    objs: Sequence[torch.Tensor],
    other_dim: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Concatenate ``(n_i, *other_dim)`` tensors into ``(sum n_i, *other_dim)``; ``[]`` gives ``(0, *other_dim)``."""
    if len(objs) == 0:
        return torch.empty((0, *other_dim), device=device, dtype=dtype)
    return torch.cat(list(objs), dim=0)


def safe_stack(
    objs: Sequence[torch.Tensor],
    other_dim: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Stack ``other_dim``-shaped tensors into ``(len(objs), *other_dim)``; ``[]`` gives ``(0, *other_dim)``."""
    if len(objs) == 0:
        return torch.empty((0, *other_dim), device=device, dtype=dtype)
    return torch.stack(list(objs), dim=0)


def seed_everything(seed: int) -> None:
    """Seed Python's ``random``, NumPy and torch on every device."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
