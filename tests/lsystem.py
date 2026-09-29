"""A parametric L-system over turtle strings.

An object is a string over ``{F, R}`` with one number per symbol: ``F(l)`` draws
forward by ``l`` and ``R(a)`` turns left by ``a`` radians, starting at the origin
heading along +x. The single production ``F(l) -> F(l/2) R(0) F(l/2)`` draws the
same path.

Run it: ``uv run python tests/lsystem.py``.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import torch

from d4d import ExtraMetrics, Grammar, ListCollection, ListSpec, StepContext


####################
# The Object
####################

@dataclass(frozen=True)
class Turtle:
    """``program[i]`` is a symbol and ``params[i]`` its parameter; ``params`` is ``(len(program),)``."""

    program: str
    params: torch.Tensor

    def __str__(self) -> str:
        p = self.params.detach().tolist()
        return " ".join(
            f"F({v:.2f})" if c == "F" else f"R({math.degrees(v):+.0f}°)" for c, v in zip(self.program, p)
        )


####################
# The Rewrites
####################

@dataclass(frozen=True)
class Expand:
    """Apply ``F -> F R F`` to the ``F`` at position ``i``."""

    i: int


Rewrite = Expand  # Expand | Contract | ... once there are more rules


####################
# The Grammar
####################

# Grammar[TObject, TCollection, TRewrite, TState]
class TurtleGrammar(Grammar[Turtle, ListCollection[Turtle], Rewrite, None]):
    """Draw a polyline that traces ``target``, an ``(N, 2)`` point set."""

    list_spec = ListSpec(
        # How to extract the parameters from the object
        params_of=lambda o: [o.params],
        # How to replace the parameters in the object.
        # This must create a new instance of the object.
        with_params=lambda o, ts: replace(o, params=ts[0]),
        names=("params",),
    )

    def __init__(self, target: torch.Tensor, max_symbols: int = 32) -> None:
        self.target = target
        self.target_length = torch.cat([torch.zeros(1, 2), target]).diff(dim=0).norm(dim=1).sum()
        self.max_symbols = max_symbols

    # -- construction ------------------------------------------------------
    def initial(self) -> Turtle:
        """The starting object."""
        return Turtle("F", torch.tensor([1.0]))

    # -- rewriting ---------------------------------------------------------
    def propose(self, obj: Turtle, budget: int) -> list[Rewrite]:
        """Sample different rewrites for the object."""
        if len(obj.program) + 2 > self.max_symbols:
            return []
        specs = [Expand(i) for i, c in enumerate(obj.program) if c == "F"]
        # Sample if too many
        if len(specs) > budget and budget > 0:
            specs = random.sample(specs, budget)
        return specs

    def apply(self, obj: Turtle, rewrite: Rewrite) -> Turtle:
        if isinstance(rewrite, Expand):
            i, v = rewrite.i, obj.params.detach()
            half = v[i : i + 1] / 2
            return Turtle(
                program=obj.program[:i] + "FRF" + obj.program[i + 1 :],
                params=torch.cat([v[:i], half, torch.zeros(1), half, v[i + 1 :]]),
            )
        # elif isinstance(rewrite, Contract):
        #     ...
        else:
            raise NotImplementedError(f"Unknown rewrite {rewrite}")

    # -- combining ---------------------------------------------------------
    def conflicts(self, a: Rewrite, b: Rewrite) -> bool:
        """Whether two rewrites conflict."""
        if isinstance(a, Expand) and isinstance(b, Expand):
            return a.i == b.i
        return False

    def apply_all(self, base: Turtle, rewrites: Sequence[Rewrite]) -> Turtle:
        """Optional. Apply right to left, so the base indices stay valid."""
        out = base
        for r in sorted(rewrites, key=lambda r: -r.i):
            out = self.apply(out, r)
        return out

    # -- evaluation --------------------------------------------------------
    def vertices(self, obj: Turtle) -> torch.Tensor:
        """The ``(n_F + 1, 2)`` polyline the turtle draws."""
        pos, heading, out = torch.zeros(2), torch.zeros(()), [torch.zeros(2)]
        for c, v in zip(obj.program, obj.params):
            if c == "R":
                heading = heading + v
            else:
                pos = pos + v * torch.stack([torch.cos(heading), torch.sin(heading)])
                out.append(pos)
        return torch.stack(out)

    def _loss_one(self, obj: Turtle) -> torch.Tensor:
        """``()``: target-to-path distance + endpoint error + length mismatch."""
        verts = self.vertices(obj)
        a, b = verts[:-1], verts[1:]                           # (S, 2) segments
        ab, aq = b - a, self.target[:, None, :] - a            # (S, 2), (N, S, 2)
        t = ((aq * ab).sum(-1) / (ab * ab).sum(-1).clamp_min(1e-12)).clamp(0, 1)  # (N, S)
        dist2 = ((aq - t[..., None] * ab) ** 2).sum(-1).min(dim=1).values  # (N,) target point -> path
        end2 = ((verts[-1] - self.target[-1]) ** 2).sum()      # () endpoint error
        length = ab.norm(dim=1).sum()                          # () path length
        return dist2.mean() + end2 + (length - self.target_length) ** 2

    def loss(
        self, batch: ListCollection[Turtle], ctx: StepContext, state: None
    ) -> tuple[torch.Tensor, ExtraMetrics]:
        """``(B,)`` losses; ``extra["n_F"]`` counts ``F`` symbols."""
        losses = torch.stack([self._loss_one(o) for o in batch.objects])
        extra = {"n_F": [float(o.program.count("F")) for o in batch.objects]} if ctx.compute_extra else {}
        return losses, extra

    def simplicity(self, batch: ListCollection[Turtle], ctx: StepContext) -> list[float]:
        """Optional. Usually the program size, weighted by OptimizeArgs.w_simplicity; never differentiated."""
        return [len(o.program) for o in batch.objects]

    def config(self) -> dict[str, Any]:
        return {"max_symbols": self.max_symbols}


def u_target(n: int = 16) -> torch.Tensor:
    """``(3n, 2)`` points along the open square (0,0) -> (1,0) -> (1,1) -> (0,1).

    Drawn exactly by ``F(1) R(+90°) F(1) R(+90°) F(1)``.
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

    grammar = TurtleGrammar(u_target())
    args = OptimizeArgs(n_steps=600, lr=0.1, propose_every=50, w_simplicity=1e-3, seed=0)
    result = optimize(grammar, args, [PrintRewrites()])
    print(f"best     loss {result.best_loss:.4f}  {result.best}")
