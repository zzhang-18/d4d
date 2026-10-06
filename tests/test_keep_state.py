"""OptimizeArgs.keep_optimizer_state: events that change nothing leave the optimizer alone; accepted rewrites carry
per-row state through Grammar.param_correspondence."""

from __future__ import annotations

from dataclasses import replace

import torch
from test_optimize import base_args
from toy import Piecewise, PiecewiseGrammar, Split, step_target

from d4d import optimize
from d4d.optimize import _transfer_state


def test_transfer_state_gathers_rows_and_seeds_new_rows() -> None:
    torch.manual_seed(0)
    src = torch.randn(3, 2, requires_grad=True)
    opt = torch.optim.Adam([src], lr=0.1)
    for _ in range(2):
        opt.zero_grad()
        (src * torch.randn(3, 2)).sum().backward()
        opt.step()
    dst = torch.zeros(4, 2, requires_grad=True)
    dst_opt = torch.optim.Adam([dst], lr=0.1)
    _transfer_state(opt, [src], dst_opt, [dst], [torch.tensor([2, -1, 0, 1])])
    s, d = opt.state[src], dst_opt.state[dst]
    assert torch.equal(d["exp_avg"][[0, 2, 3]], s["exp_avg"][[2, 0, 1]])
    assert torch.equal(d["exp_avg"][1], torch.zeros(2))
    assert torch.allclose(d["exp_avg_sq"][1], s["exp_avg_sq"].mean(0))
    assert float(d["step"]) == float(s["step"]) == 2.0
    dst_opt.zero_grad()
    dst.sum().backward()
    dst_opt.step()  # the seeded state is a valid Adam state


def test_events_without_accepts_leave_the_optimizer_alone() -> None:
    """Nothing is ever accepted (huge simplicity price, no removes): with keep_optimizer_state the run must match one
    with no proposal events at all, step for step."""
    target = step_target(64)
    args = base_args(w_simplicity=1e6, keep_optimizer_state=True, n_steps=100)
    g = PiecewiseGrammar(target, n_initial=3, allow_remove=False)
    res = optimize(g, args)
    assert res.final.n == 3
    ref = optimize(PiecewiseGrammar(target, n_initial=3, allow_remove=False), replace(args, propose_every=10_000))
    assert torch.allclose(torch.tensor(res.metrics["$loss"]), torch.tensor(ref.metrics["$loss"]), atol=0, rtol=0)


class CarryGrammar(PiecewiseGrammar):
    """Toy grammar with a row correspondence: a split segment's halves both continue it."""

    def param_correspondence(self, old: Piecewise, new: Piecewise) -> list[torch.Tensor | None]:
        src = []
        j = 0
        for e0, e1 in zip(new.edges, new.edges[1:]):
            while j < old.n - 1 and old.edges[j + 1] <= e0 + 1e-12:
                j += 1
            src.append(j if old.edges[j] <= e0 + 1e-12 and e1 <= old.edges[j + 1] + 1e-12 else -1)
        return [torch.tensor(src)]


def test_keep_state_with_accepts_still_fits() -> None:
    target = step_target(64)
    for g in (PiecewiseGrammar(target, n_initial=1, allow_remove=False), CarryGrammar(target, n_initial=1, allow_remove=False)):
        res = optimize(g, base_args(keep_optimizer_state=True))
        assert res.final.n >= 4 and res.best_loss < 0.02, (type(g).__name__, res.final.n, res.best_loss)


def test_carry_grammar_maps_split_halves_to_the_parent() -> None:
    g = CarryGrammar(step_target(8), n_initial=2)
    old = g.initial()
    new = g.apply(old, Split(1))
    (m,) = g.param_correspondence(old, new)
    assert m.tolist() == [0, 1, 1]
