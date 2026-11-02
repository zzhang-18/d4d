"""Two-stage scoring (``proposal_stage2_steps``) and verified combination (``accept_verify``)."""

from __future__ import annotations

from typing import Any

import pytest
from toy import PiecewiseGrammar, step_target

from d4d import Callback, OptimizeArgs, optimize


def base_args(**kw: Any) -> OptimizeArgs:
    defaults: dict[str, Any] = {
        "n_steps": 100, "optimizer": "Adam", "scheduler": "none", "lr": 0.1, "clip_grad": None,
        "propose_every": 20, "proposal_size": 0, "proposal_criterion": "loss", "proposal_steps": 2,
        "cleanup_every": 10, "visualize_every": 0, "batch_size": 8, "w_simplicity": 0.0,
        "stopping_patience": None, "seed": 0,
    }
    defaults.update(kw)
    return OptimizeArgs(**defaults)


class Recorder(Callback):
    def __init__(self) -> None:
        self.rewrites: list[Any] = []

    def on_rewrite(self, ev: Any) -> None:
        self.rewrites.append(ev)


class KindGrammar(PiecewiseGrammar):
    """The toy grammar with its rewrite types as stage-2 groups."""

    def rewrite_group(self, rewrite: Any) -> str:
        return type(rewrite).__name__


def test_defaults_record_no_stage2_or_verification():
    rec = Recorder()
    optimize(PiecewiseGrammar(step_target(64)), base_args(), [rec])
    assert rec.rewrites
    assert all(ev.stage2 == () and ev.combined == () and ev.verify_base_loss is None for ev in rec.rewrites)


def test_stage2_rescores_the_best_per_group_and_accepts_only_them():
    rec = Recorder()
    g = KindGrammar(step_target(64), n_initial=2)
    optimize(g, base_args(proposal_stage2_steps=6, proposal_stage2_top=1), [rec])
    assert any(ev.changed for ev in rec.rewrites)
    for ev in rec.rewrites:
        groups: dict[str, list[int]] = {}
        for i, r in enumerate(ev.rewrites):
            groups.setdefault(type(r).__name__, []).append(i)
        best = sorted(min(idx, key=lambda i: ev.losses[i]) for idx in groups.values())
        assert list(ev.stage2) == best
        assert ev.stage2_base_loss is not None and len(ev.stage2_losses) == len(ev.stage2)
        kept = {id(ev.rewrites[i]) for i in ev.stage2}
        assert all(id(a) in kept for a in ev.accepted)
        for a in ev.accepted:  # improving under the second stage's scores
            j = next(k for k, i in enumerate(ev.stage2) if ev.rewrites[i] is a)
            assert ev.stage2_losses[j] < ev.stage2_base_loss


def test_verify_keeps_the_best_prefix_that_beats_the_base():
    rec = Recorder()
    g = PiecewiseGrammar(step_target(64), n_initial=1, allow_remove=False)
    optimize(g, base_args(accept_verify=True, n_steps=120), [rec])
    checked = [ev for ev in rec.rewrites if ev.combined]
    assert checked, "no event combined two or more rewrites"
    for ev in checked:
        assert len(ev.verify_losses) == len(ev.combined) and ev.verify_base_loss is not None
        assert tuple(ev.accepted) == tuple(ev.combined[: len(ev.accepted)])
        if ev.accepted:
            k = len(ev.accepted) - 1
            assert ev.verify_losses[k] == min(ev.verify_losses) < ev.verify_base_loss
        else:
            assert min(ev.verify_losses) >= ev.verify_base_loss


@pytest.mark.parametrize("kw", [{"proposal_stage2_steps": 5}, {"accept_verify": True}])
def test_stage2_and_verify_need_the_loss_criterion(kw: dict[str, Any]):
    with pytest.raises(ValueError):
        base_args(proposal_criterion="grad", **kw)
    with pytest.raises(ValueError):
        base_args(proposal_stage2_top=0)
