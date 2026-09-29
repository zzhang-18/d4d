"""A piecewise-constant fit to a 1-D signal, with a grammar-chosen number of segments.

``Split`` halves a segment and gives both halves its value, so the represented
function is unchanged. ``Remove`` merges a segment into its right neighbour.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import torch

from d4d import ExtraMetrics, Grammar, ListCollection, ListSpec, StepContext


@dataclass(frozen=True)
class Piecewise:
    """``values`` is ``(n,)``; ``edges`` holds ``n + 1`` increasing positions spanning [0, 1]."""

    edges: tuple[float, ...]
    values: torch.Tensor

    @property
    def n(self) -> int:
        return len(self.values)


@dataclass(frozen=True)
class Split:
    """Halve segment ``i``, both halves keeping its current value."""

    i: int


@dataclass(frozen=True)
class Remove:
    """Merge segment ``i`` into its right neighbour, averaging by width."""

    i: int


class PiecewiseGrammar(Grammar[Piecewise, ListCollection[Piecewise], Any, None]):
    """Fit ``target``, a ``(T,)`` signal sampled at ``T`` evenly spaced points in [0, 1]."""

    list_spec = ListSpec(
        params_of=lambda o: [o.values],
        with_params=lambda o, ts: replace(o, values=ts[0]),
        names=("values",),
    )

    def __init__(
        self,
        target: torch.Tensor,
        n_initial: int = 1,
        max_segments: int = 64,
        allow_remove: bool = True,
    ) -> None:
        self.target = target
        self.xs = (torch.arange(len(target), dtype=torch.float32) + 0.5) / len(target)  # (T,)
        self.n_initial = n_initial
        self.max_segments = max_segments
        self.allow_remove = allow_remove
        self.visualize_calls = 0

    # -- construction ------------------------------------------------------
    def initial(self) -> Piecewise:
        n = self.n_initial
        edges = tuple(i / n for i in range(n + 1))
        return Piecewise(edges=edges, values=torch.full((n,), float(self.target.mean())))

    # -- rewriting ---------------------------------------------------------
    def propose(self, obj: Piecewise, budget: int) -> list[Any]:
        out: list[Any] = []
        if obj.n < self.max_segments:
            out.extend(Split(i) for i in range(obj.n))
        if self.allow_remove and obj.n > 1:
            out.extend(Remove(i) for i in range(obj.n - 1))
        if budget > 0 and len(out) > budget:
            out = out[:budget]
        return out

    def apply(self, obj: Piecewise, rewrite: Any) -> Piecewise:
        if isinstance(rewrite, Split):
            i = rewrite.i
            mid = 0.5 * (obj.edges[i] + obj.edges[i + 1])
            edges = obj.edges[: i + 1] + (mid,) + obj.edges[i + 1 :]
            v = obj.values.detach()
            values = torch.cat([v[:i], v[i : i + 1], v[i : i + 1], v[i + 1 :]])
            return Piecewise(edges=edges, values=values)
        if isinstance(rewrite, Remove):
            i = rewrite.i
            w0 = obj.edges[i + 1] - obj.edges[i]
            w1 = obj.edges[i + 2] - obj.edges[i + 1]
            v = obj.values.detach()
            merged = (v[i] * w0 + v[i + 1] * w1) / (w0 + w1)
            edges = obj.edges[: i + 1] + obj.edges[i + 2 :]
            values = torch.cat([v[:i], merged.reshape(1), v[i + 2 :]])
            return Piecewise(edges=edges, values=values)
        raise ValueError(f"unknown rewrite {rewrite!r}")

    def conflicts(self, a: Any, b: Any) -> bool:
        """Rewrites on the same or neighbouring segments conflict."""
        return abs(a.i - b.i) <= 1

    def apply_all(
        self, base: Piecewise, rewrites: Sequence[Any], improvements: Sequence[float]
    ) -> Piecewise:
        out = base  # right to left, so base indices stay valid
        for rewrite in sorted(rewrites, key=lambda r: -r.i):
            out = self.apply(out, rewrite)
        return out

    # -- evaluation --------------------------------------------------------
    def _predict(self, obj: Piecewise) -> torch.Tensor:
        """``(T,)`` values of ``obj`` at ``xs``."""
        edges = torch.tensor(obj.edges[1:-1], dtype=torch.float32)
        idx = torch.searchsorted(edges, self.xs.contiguous())
        return obj.values[idx]

    def loss(
        self, batch: ListCollection[Piecewise], ctx: StepContext, state: None
    ) -> tuple[torch.Tensor, ExtraMetrics]:
        """``(B,)`` mean squared error; ``extra["n_segments"]`` counts segments."""
        losses = torch.stack([((self._predict(o) - self.target) ** 2).mean() for o in batch.objects])
        extra: dict[str, Sequence[float]] = {}
        if ctx.compute_extra:
            extra["n_segments"] = [float(o.n) for o in batch.objects]
        return losses, extra

    def simplicity(self, batch: ListCollection[Piecewise], ctx: StepContext) -> Sequence[float]:
        return [float(o.n) for o in batch.objects]

    def visualize(
        self, batch: ListCollection[Piecewise], ctx: StepContext, state: None
    ) -> Any | None:
        """``(8, T, 3)`` uint8 grayscale strip of the first object's prediction."""
        import numpy as np

        self.visualize_calls += 1
        pred = self._predict(batch.objects[0]).detach().numpy()
        row = (255 * (pred - pred.min()) / max(float(np.ptp(pred)), 1e-9)).astype(np.uint8)
        return np.repeat(row[None, :, None], 3, axis=2).repeat(8, axis=0)

    def config(self) -> dict[str, Any]:
        return {"n_initial": self.n_initial, "max_segments": self.max_segments}


def step_target(n: int = 64) -> torch.Tensor:
    """``(n,)`` monotone staircase with levels 0, 0.3, 0.6, 1; exact with 4 segments."""
    t = torch.zeros(n)
    q = n // 4
    for k, v in enumerate([0.0, 0.3, 0.6, 1.0]):
        t[k * q : (k + 1) * q] = v
    return t


def ramp_target(n: int = 32) -> torch.Tensor:
    """``(n,)`` linear ramp from 0 to 1."""
    return torch.linspace(0.0, 1.0, n)
