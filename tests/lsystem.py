"""The smallest grammar worth writing: a parametric L-system over turtle strings.

An object is a string over ``{F, R}`` with one number attached to each symbol:
``F(l)`` draws forward by ``l``, ``R(a)`` turns left by ``a`` radians. The turtle
starts at the origin heading along +x. There is a single production,

    F(l)  ->  F(l/2) R(0) F(l/2)

which draws exactly the same path -- so, like ``Split`` in ``toy.py``, it is
**loss-preserving at the moment it fires**. Descent then bends the new corner.

Run it: ``uv run python tests/lsystem.py``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import torch

from d4d import ExtraMetrics, Grammar, ListCollection, ListSpec, StepContext


@dataclass(frozen=True)
class Turtle:
    """``program[i]`` is a symbol and ``params[i]`` its parameter."""

    program: str
    params: torch.Tensor

    def __str__(self) -> str:
        p = self.params.detach().tolist()
        return " ".join(
            f"F({v:.2f})" if c == "F" else f"R({math.degrees(v):+.0f}°)" for c, v in zip(self.program, p)
        )


@dataclass(frozen=True)
class Expand:
    """Apply ``F -> F R F`` to the ``F`` at position ``i``."""

    i: int


class TurtleGrammar(Grammar[Turtle]):
    """Draw a polyline that traces ``target``, a ``(N, 2)`` point set."""

    list_spec = ListSpec(
        params_of=lambda o: [o.params],
        with_params=lambda o, ts: replace(o, params=ts[0]),
        names=("params",),
    )

    def __init__(self, target: torch.Tensor, w_symbol: float = 0.0, max_symbols: int = 32) -> None:
        self.target = target
        self.target_length = torch.cat([torch.zeros(1, 2), target]).diff(dim=0).norm(dim=1).sum()
        self.w_symbol = w_symbol
        self.max_symbols = max_symbols

    # -- construction ------------------------------------------------------
    def initial(self) -> Turtle:
        return Turtle("F", torch.tensor([1.0]))

    # -- rewriting ---------------------------------------------------------
    def propose(self, obj: Turtle, budget: int) -> list[Expand]:
        if len(obj.program) + 2 > self.max_symbols:
            return []
        out = [Expand(i) for i, c in enumerate(obj.program) if c == "F"]
        return out[:budget] if budget > 0 else out

    def apply(self, obj: Turtle, rewrite: Expand) -> Turtle:
        i, v = rewrite.i, obj.params.detach()
        half = v[i : i + 1] / 2
        return Turtle(
            program=obj.program[:i] + "FRF" + obj.program[i + 1 :],
            params=torch.cat([v[:i], half, torch.zeros(1), half, v[i + 1 :]]),
        )

    # -- combining ---------------------------------------------------------
    def conflicts(self, a: Expand, b: Expand) -> bool:
        return False  # distinct positions never interfere...

    def apply_all(self, base: Turtle, rewrites: Sequence[Expand], improvements: Sequence[float]) -> Turtle:
        out = base  # ...provided we splice right-to-left, so earlier indices stay valid
        for r in sorted(rewrites, key=lambda r: -r.i):
            out = self.apply(out, r)
        return out

    # -- evaluation --------------------------------------------------------
    def vertices(self, obj: Turtle) -> torch.Tensor:
        """Interpret the string: the ``(n_F + 1, 2)`` polyline the turtle draws."""
        pos, heading, out = torch.zeros(2), torch.zeros(()), [torch.zeros(2)]
        for c, v in zip(obj.program, obj.params):
            if c == "R":
                heading = heading + v
            else:
                pos = pos + v * torch.stack([torch.cos(heading), torch.sin(heading)])
                out.append(pos)
        return torch.stack(out)

    def _loss_one(self, obj: Turtle) -> torch.Tensor:
        # Every term depends only on the drawn path, not on how it is split into
        # symbols -- so Expand, which redraws the same path, leaves the loss unchanged.
        verts = self.vertices(obj)
        a, b = verts[:-1], verts[1:]                           # (S, 2) segments
        ab, aq = b - a, self.target[:, None, :] - a            # (S, 2), (N, S, 2)
        t = ((aq * ab).sum(-1) / (ab * ab).sum(-1).clamp_min(1e-12)).clamp(0, 1)
        dist2 = ((aq - t[..., None] * ab) ** 2).sum(-1).min(dim=1).values  # target point -> path
        end2 = ((verts[-1] - self.target[-1]) ** 2).sum()      # finish where the target does
        length = ab.norm(dim=1).sum()                          # ...having drawn as much as it
        return dist2.mean() + end2 + (length - self.target_length) ** 2

    def loss(
        self, batch: ListCollection[Turtle], ctx: StepContext, state: None
    ) -> tuple[torch.Tensor, ExtraMetrics]:
        losses = torch.stack([self._loss_one(o) for o in batch.objects])
        extra = {"n_F": [float(o.program.count("F")) for o in batch.objects]} if ctx.compute_extra else {}
        return losses, extra

    def simplicity(self, batch: ListCollection[Turtle], ctx: StepContext) -> Sequence[float]:
        return [self.w_symbol * len(o.program) for o in batch.objects]

    def config(self) -> dict[str, Any]:
        return {"w_symbol": self.w_symbol, "max_symbols": self.max_symbols}


def u_target(n: int = 16) -> torch.Tensor:
    """``3n`` points along the open square (0,0) -> (1,0) -> (1,1) -> (0,1), ending at (0,1).

    Exactly drawn by ``F(1) R(+90°) F(1) R(+90°) F(1)``; one ``F`` is hopeless.
    """
    s = torch.arange(1, n + 1, dtype=torch.float32) / n
    one, zero = torch.ones(n), torch.zeros(n)
    return torch.cat([torch.stack([s, zero], 1), torch.stack([one, s], 1), torch.stack([1 - s, one], 1)])


if __name__ == "__main__":
    from d4d import Callback, OptimizeArgs, optimize

    class PrintRewrites(Callback):
        def on_step_end(self, ev):
            if ev.rewrite or ev.step == 0:
                print(f"step {ev.step:4d}  loss {ev.loss_cont:.4f}  {ev.get_object()}")

    grammar = TurtleGrammar(u_target(), w_symbol=1e-3)
    args = OptimizeArgs(n_steps=600, lr=0.1, propose_every=50, seed=0)
    result = optimize(grammar, args, [PrintRewrites()])
    print(f"best     loss {result.best_loss:.4f}  {result.best}")
