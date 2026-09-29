"""End-to-end tests of the optimization loop, via the toy piecewise grammar."""

from __future__ import annotations

import math
import warnings

import pytest
import torch
from toy import PiecewiseGrammar, Split, ramp_target, step_target

from d4d import (
    Callback,
    EarlyStopOnNaN,
    HistoryRecorder,
    OptimizeArgs,
    StepContext,
    StepEnd,
    optimize,
)


def base_args(**kw) -> OptimizeArgs:
    defaults: dict[str, object] = {
        "n_steps": 120, "optimizer": "Adam", "scheduler": "none", "lr": 0.1, "clip_grad": None,
        "propose_every": 20, "proposal_size": 0, "proposal_criterion": "loss", "proposal_steps": 2,
        "cleanup_every": 10, "visualize_every": 0, "batch_size": 8, "w_simplicity": 0.0,
        "stopping_patience": None, "seed": 0,
    }
    defaults.update(kw)
    return OptimizeArgs(**defaults)  # type: ignore[arg-type]


class Recorder(Callback):
    def __init__(self) -> None:
        self.steps: list[StepEnd] = []
        self.rewrites: list = []
        self.ended = False

    def on_step_end(self, ev): self.steps.append(ev)
    def on_rewrite(self, ev): self.rewrites.append(ev)
    def on_run_end(self, ev): self.ended = True


def test_grows_the_program_and_fits_the_target():
    """A 4-level staircase is unreachable with one segment; the grammar must split."""
    target = step_target(64)
    g = PiecewiseGrammar(target, n_initial=1, allow_remove=False)
    rec = Recorder()
    res = optimize(g, base_args(), [rec])

    assert res.metrics["n_segments"][0] == 1.0
    assert res.final.n >= 4, "grammar failed to grow"
    assert res.best_loss < 0.02, f"poor fit: {res.best_loss}"
    assert res.metrics["$loss"][-1] < res.metrics["$loss"][0]
    assert len(rec.steps) == res.n_steps_run
    assert rec.ended


def test_split_is_exactly_loss_preserving():
    """The central claim, tested directly: a rewrite costs nothing when it fires.

    Split divides a segment into two halves carrying the same value, so the
    represented function is unchanged and only the parameter count grows. The
    loss must be identical before and after -- which is what lets the optimizer
    take a discrete jump without ever paying for it.

    Tested on the grammar rather than on the loss series, because the series
    records each step's loss *before* that step's parameter update, so a
    series-level comparison would bundle a gradient step in with the rewrite.
    """
    g = PiecewiseGrammar(step_target(64), n_initial=4, allow_remove=False)
    ctx = StepContext(step=0, total_steps=1, phase="step")
    obj = g.initial()
    obj = g.apply(g.apply(obj, Split(0)), Split(3))  # a couple of splits deep
    before, _ = g.loss(g.collate([obj]), ctx, None)

    for rewrite in g.propose(obj, budget=0):
        after, _ = g.loss(g.collate([g.apply(obj, rewrite)]), ctx, None)
        assert after.item() == pytest.approx(before.item(), abs=1e-9), (
            f"{rewrite} changed the loss: {before.item():.9e} -> {after.item():.9e}"
        )

    # and jointly, through the same apply_all path combine() uses
    joint = g.apply_all(obj, [Split(0), Split(2)])
    after, _ = g.loss(g.collate([joint]), ctx, None)
    assert after.item() == pytest.approx(before.item(), abs=1e-9)


def test_every_accepted_rewrite_improved_on_the_base():
    """combine must never accept a candidate that scored worse than the base."""
    g = PiecewiseGrammar(step_target(64), n_initial=1, allow_remove=False)
    rec = Recorder()
    optimize(g, base_args(n_steps=100, propose_every=20), [rec])

    assert any(ev.changed for ev in rec.rewrites), "no rewrite was accepted"
    for ev in rec.rewrites:
        by_id = {id(r): imp for r, imp in zip(ev.rewrites, ev.improvements())}
        for accepted in ev.accepted:
            assert by_id[id(accepted)] > 0.0, f"accepted a non-improving rewrite at step {ev.step}"


def test_simplicity_suppresses_growth():
    """Pricing segments high enough should stop the grammar growing at all."""
    target = step_target(64)
    cheap = optimize(PiecewiseGrammar(target, allow_remove=False), base_args(w_simplicity=0.0))
    dear = optimize(
        PiecewiseGrammar(target, allow_remove=False),
        base_args(w_simplicity=1.0),
    )
    assert dear.final.n < cheap.final.n
    assert dear.final.n == 1


def test_accept_top_k_limits_rewrites_per_step():
    target = step_target(64)
    rec_all, rec_one = Recorder(), Recorder()
    optimize(PiecewiseGrammar(target, allow_remove=False), base_args(accept_top_k=0), [rec_all])
    optimize(PiecewiseGrammar(target, allow_remove=False), base_args(accept_top_k=1), [rec_one])

    assert all(len(ev.accepted) <= 1 for ev in rec_one.rewrites)
    assert max(len(ev.accepted) for ev in rec_all.rewrites) > 1


@pytest.mark.parametrize("criterion", ["loss", "grad", "grad_only"])
def test_all_proposal_criteria_run(criterion: str):
    target = step_target(64)
    g = PiecewiseGrammar(target, n_initial=1, allow_remove=False)
    res = optimize(g, base_args(proposal_criterion=criterion), [])
    assert res.n_steps_run == 120
    assert math.isfinite(res.best_loss)


@pytest.mark.parametrize("scheduler", ["none", "ReduceLROnPlateau", "AdaptiveLR", "LinearLR", "ExponentialLR"])
def test_all_schedulers_run(scheduler: str):
    target = step_target(32)
    g = PiecewiseGrammar(target, allow_remove=False)
    res = optimize(g, base_args(n_steps=40, scheduler=scheduler), [])
    assert len(res.metrics["$lr"]) == 40
    assert all(math.isfinite(x) for x in res.metrics["$lr"])


def test_metrics_series_are_aligned_and_complete():
    target = step_target(32)
    res = optimize(PiecewiseGrammar(target, allow_remove=False), base_args(n_steps=50), [])
    for key in ("$loss", "$loss_ma", "$loss_cont", "$loss_simp", "$lr", "$timestamp", "n_segments"):
        assert key in res.metrics, key
        assert len(res.metrics[key]) == res.n_steps_run, key
    # $loss is $loss_cont + w_simplicity * $loss_simp, and w_simplicity is 0 here
    assert res.metrics["$loss"] == pytest.approx(res.metrics["$loss_cont"])


def test_best_is_the_argmin_not_the_last():
    """Upstream returned the final loss as 'best'. It must be the actual minimum."""
    target = step_target(32)
    res = optimize(PiecewiseGrammar(target, allow_remove=False), base_args(n_steps=60), [])
    assert res.best_loss == pytest.approx(min(res.metrics["$loss"]))
    assert res.metrics["$loss"][res.best_step] == pytest.approx(res.best_loss)


def test_history_is_opt_in_and_strided():
    target = step_target(32)
    hist = HistoryRecorder(every=10)
    optimize(PiecewiseGrammar(target, allow_remove=False), base_args(n_steps=50), [hist])
    assert hist.steps == list(range(0, 50, 10))
    assert len(hist.objects) == 5
    assert all(o is not None for o in hist.objects)


def test_visualization_fires_on_schedule_and_after_rewrites():
    target = step_target(32)
    g = PiecewiseGrammar(target, allow_remove=False)
    rec = Recorder()

    class VizCount(Callback):
        def __init__(self): self.n = 0
        def on_visualize(self, ev): self.n += 1

    vc = VizCount()
    optimize(g, base_args(n_steps=60, visualize_every=10, propose_every=25), [vc, rec])
    # every 10th step, plus rewrite steps 25 and 50 which are not multiples of 10
    assert vc.n >= 6
    assert g.visualize_calls == vc.n


def test_seed_makes_runs_reproducible():
    target = step_target(32)
    a = optimize(PiecewiseGrammar(target, allow_remove=False), base_args(n_steps=40, seed=7), [])
    b = optimize(PiecewiseGrammar(target, allow_remove=False), base_args(n_steps=40, seed=7), [])
    assert a.metrics["$loss"] == pytest.approx(b.metrics["$loss"])
    assert a.final.edges == b.final.edges


def test_seed_covers_sampled_proposals():
    """A budget below the proposal count makes propose() call random.sample; seed must pin it."""
    target = step_target(32)
    args = base_args(n_steps=80, propose_every=10, proposal_size=2, seed=3)
    a = optimize(PiecewiseGrammar(target, allow_remove=True), args, [])
    b = optimize(PiecewiseGrammar(target, allow_remove=True), args, [])
    assert a.metrics["$loss"] == pytest.approx(b.metrics["$loss"])
    assert a.final.edges == b.final.edges


def test_early_stopping_halts_on_stalled_rewrites():
    """With growth capped, the loss plateaus and the rewrite budget is abandoned.

    The target is a ramp, so the plateau sits at a *positive* loss. That matters:
    the stall test is relative (``ma > prev * (1 - eps)``), which can never fire
    when the loss is exactly zero.
    """
    g = PiecewiseGrammar(ramp_target(32), n_initial=1, allow_remove=False, max_segments=1)
    res = optimize(g, base_args(n_steps=400, propose_every=10, stopping_patience=2), [])
    assert res.stopped_early
    assert res.n_steps_run < 400


def test_callback_can_stop_the_run_cleanly():
    """StopRun is early stopping, not failure: a result is still produced."""

    class StopAt(Callback):
        def on_step_end(self, ev):
            if ev.step == 17:
                from d4d import StopRun

                raise StopRun("done")

    target = step_target(32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = optimize(PiecewiseGrammar(target, allow_remove=False), base_args(n_steps=100), [StopAt()])
    assert res.stopped_early
    assert res.n_steps_run == 18


def test_nan_guard_stops_instead_of_burning_steps():
    """A non-finite loss ends the run rather than burning the remaining budget."""

    class NaNAfter(PiecewiseGrammar):
        def loss(self, batch, ctx, state):
            losses, extra = super().loss(batch, ctx, state)
            if ctx.phase == "step" and ctx.step >= 5:
                losses = losses * float("nan")
            return losses, extra

    g = NaNAfter(step_target(32), allow_remove=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = optimize(g, base_args(n_steps=200), [EarlyStopOnNaN()])
    assert res.n_steps_run == 6
    assert res.stopped_early


def test_run_end_fires_even_when_the_grammar_raises():
    class Boom(PiecewiseGrammar):
        def loss(self, batch, ctx, state):
            raise RuntimeError("boom")

    rec = Recorder()
    with pytest.raises(RuntimeError, match="boom"):
        optimize(Boom(step_target(32)), base_args(n_steps=10), [rec])
    assert rec.ended, "on_run_end must fire from the finally block"


def test_broken_callback_warns_but_does_not_kill_the_run():
    class Bad(Callback):
        def on_step_end(self, ev):
            raise ValueError("callback bug")

    with pytest.warns(UserWarning, match="callback bug"):
        res = optimize(PiecewiseGrammar(step_target(32), allow_remove=False),
                       base_args(n_steps=20), [Bad()])
    assert res.n_steps_run == 20


def test_remove_rewrites_are_reachable():
    """Both rule families must actually fire, not just Split."""
    target = torch.full((32,), 0.5)
    g = PiecewiseGrammar(target, n_initial=6, allow_remove=True)
    rec = Recorder()
    optimize(g, base_args(n_steps=120, propose_every=15, w_simplicity=0.05), [rec])
    kinds = {type(r).__name__ for ev in rec.rewrites for r in ev.accepted}
    assert "Remove" in kinds, f"only saw {kinds}"
