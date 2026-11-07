"""Resuming a run at a step (``OptimizeArgs.start_step``)."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from toy import Piecewise, PiecewiseGrammar, step_target

from d4d import Callback, OptimizeArgs, optimize


def base_args(**kw: Any) -> OptimizeArgs:
    defaults: dict[str, Any] = {
        "n_steps": 100, "optimizer": "Adam", "scheduler": "none", "lr": 0.1, "clip_grad": None,
        "propose_every": 20, "proposal_size": 0, "proposal_criterion": "loss", "proposal_steps": 2,
        "cleanup_every": 20, "visualize_every": 0, "batch_size": 8, "w_simplicity": 0.0,
        "stopping_patience": None, "seed": 0,
    }
    defaults.update(kw)
    return OptimizeArgs(**defaults)


class Recorder(Callback):
    """Per-step losses and steps, rewrite events, and the object at the end of step ``capture_at``."""

    def __init__(self, capture_at: int = -1) -> None:
        self.capture_at = capture_at
        self.captured: Piecewise | None = None
        self.steps: list[int] = []
        self.losses: list[float] = []
        self.rewrites: list[tuple[int, int]] = []

    def on_step_end(self, ev: Any) -> None:
        self.steps.append(ev.step)
        self.losses.append(ev.loss)
        if ev.step == self.capture_at:
            self.captured = ev.get_object()

    def on_rewrite(self, ev: Any) -> None:
        self.rewrites.append((ev.step, len(ev.accepted)))


class FromObject(PiecewiseGrammar):
    def __init__(self, start: Piecewise, *a: Any, **kw: Any) -> None:
        super().__init__(*a, **kw)
        self.start = start

    def initial(self) -> Piecewise:
        return Piecewise(self.start.edges, self.start.values.detach().clone())


def test_resume_at_an_event_reproduces_the_rest_of_the_run():
    """The optimizer is rebuilt at every event, so a run resumed at event step 40 from the object at the end of step 39
    replays steps 40..99 of the full run exactly: same losses, same rewrites."""
    full = Recorder(capture_at=39)
    optimize(PiecewiseGrammar(step_target(64)), base_args(), [full])
    assert full.captured is not None
    resumed = Recorder()
    optimize(FromObject(full.captured, step_target(64)), base_args(start_step=40), [resumed])
    assert resumed.steps == list(range(40, 100))
    assert resumed.losses == full.losses[40:]
    assert resumed.rewrites == [r for r in full.rewrites if r[0] >= 40]
    assert resumed.rewrites[0][0] == 40  # the event at the resumed step fires


def test_steps_and_result_counters_are_absolute():
    r = Recorder()
    res = optimize(PiecewiseGrammar(step_target(64)), base_args(start_step=50), [r])
    assert r.steps[0] == 50 and r.steps[-1] == 99
    assert res.n_steps_run == 100
    assert res.best_step >= 50


def test_default_is_a_fresh_run():
    r1, r2 = Recorder(), Recorder()
    optimize(PiecewiseGrammar(step_target(64)), base_args(), [r1])
    optimize(PiecewiseGrammar(step_target(64)), base_args(start_step=0), [r2])
    assert r1.losses == r2.losses and r1.rewrites == r2.rewrites


@pytest.mark.parametrize("start", [-1, 100, 150])
def test_start_step_out_of_range_is_rejected(start: int):
    with pytest.raises(ValueError, match="start_step"):
        base_args(start_step=start)


def test_linear_lr_spans_the_remaining_steps():
    lrs: list[float] = []

    class LR(Callback):
        def on_step_end(self, ev: Any) -> None:
            lrs.append(ev.lr)

    optimize(PiecewiseGrammar(step_target(64)), base_args(scheduler="LinearLR", start_step=60, propose_every=1000,
                                                          cleanup_every=1000), [LR()])  # fmt: skip
    assert len(lrs) == 40
    assert lrs[0] == pytest.approx(0.1 * (1 - 1 / 40))
    assert lrs[-1] == pytest.approx(0.0, abs=1e-12)
    assert torch.tensor(lrs).diff().le(0).all()
