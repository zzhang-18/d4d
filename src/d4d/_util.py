"""Small helpers for the optimzer."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence

import torch

__all__ = ["MovingAverage", "maybe_clamp", "safe_cat", "safe_stack"]


class MovingAverage:
    """Fixed-window mean over a stream of scalars.

    Drives both the rewrite trigger (``proposal_trigger="rel_loss"``) and the
    early-stopping check, which compares the smoothed loss between consecutive
    rewrite events. ``clear()`` is called at every rewrite so the window never
    straddles a discrete jump.
    """

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
    """``x.clamp`` that is a no-op when both bounds are ``None``."""
    if min is None and max is None:
        return x
    return x.clamp(min=min, max=max)


def safe_cat(
    objs: Sequence[torch.Tensor],
    other_dim: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """``torch.cat`` along dim 0 that tolerates an empty list."""
    if len(objs) == 0:
        return torch.empty((0, *other_dim), device=device, dtype=dtype)
    return torch.cat(list(objs), dim=0)


def safe_stack(
    objs: Sequence[torch.Tensor],
    other_dim: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """``torch.stack`` along dim 0 that tolerates an empty list."""
    if len(objs) == 0:
        return torch.empty((0, *other_dim), device=device, dtype=dtype)
    return torch.stack(list(objs), dim=0)
