"""Scoring LR multiplier (``proposal_lr_mult``), grammar-defined candidate scores (``proposal_eval``) and warm-restart
cosine segments (``scheduler='CosineSegment'``)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import pytest
from toy import PiecewiseGrammar, step_target

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
    def __init__(self) -> None:
        self.rewrites: list[Any] = []
        self.lrs: list[float] = []

    def on_rewrite(self, ev: Any) -> None:
        self.rewrites.append(ev)

    def on_step_end(self, ev: Any) -> None:
        self.lrs.append(ev.lr)


class ScoredGrammar(PiecewiseGrammar):
    """The toy grammar whose proposal score is its loss plus a constant, and counts its calls."""

    def __init__(self, *a: Any, **kw: Any) -> None:
        super().__init__(*a, **kw)
        self.calls = 0

    def proposal_score(self, batch: Any, ctx: Any, state: Any) -> Sequence[float]:
        self.calls += 1
        losses, _ = self.loss(batch, ctx, state)
        return [float(v) + 1.0 for v in losses]


def test_defaults_are_unchanged_by_the_new_fields():
    """The defaults (lr mult 1, eval 'loss', scheduler 'none') give the same run as before: same losses and accepts."""
    r1, r2 = Recorder(), Recorder()
    optimize(PiecewiseGrammar(step_target(64)), base_args(), [r1])
    optimize(PiecewiseGrammar(step_target(64)), base_args(proposal_lr_mult=1.0, proposal_eval="loss"), [r2])
    assert [ev.losses for ev in r1.rewrites] == [ev.losses for ev in r2.rewrites]
    assert [len(ev.accepted) for ev in r1.rewrites] == [len(ev.accepted) for ev in r2.rewrites]


def test_lr_mult_changes_scores_only_through_scoring():
    r1, r2 = Recorder(), Recorder()
    optimize(PiecewiseGrammar(step_target(64)), base_args(n_steps=21), [r1])
    optimize(PiecewiseGrammar(step_target(64)), base_args(n_steps=21, proposal_lr_mult=3.0), [r2])
    assert len(r1.rewrites) == len(r2.rewrites) == 1
    assert r1.rewrites[0].losses != r2.rewrites[0].losses  # the same candidates, scored after different steps
    assert r1.lrs[:20] == r2.lrs[:20]  # the main loop's LR is untouched


def test_grammar_eval_uses_proposal_score():
    g = ScoredGrammar(step_target(64))
    rec = Recorder()
    optimize(g, base_args(proposal_eval="grammar"), [rec])
    assert g.calls > 0
    ref = Recorder()
    optimize(PiecewiseGrammar(step_target(64)), base_args(n_steps=21), [ref])
    g2, rec2 = ScoredGrammar(step_target(64)), Recorder()
    optimize(g2, base_args(n_steps=21, proposal_eval="grammar"), [rec2])
    # the first event sees the same state: scores are the losses + 1, base included, so the decisions are the same
    assert rec2.rewrites[0].losses == pytest.approx([v + 1.0 for v in ref.rewrites[0].losses])
    assert rec2.rewrites[0].base_loss == pytest.approx(ref.rewrites[0].base_loss + 1.0)


def test_grammar_eval_needs_the_loss_criterion_and_a_score():
    with pytest.raises(ValueError):
        base_args(proposal_eval="grammar", proposal_criterion="grad")
    with pytest.raises(NotImplementedError):
        optimize(PiecewiseGrammar(step_target(64)), base_args(n_steps=21, proposal_eval="grammar"), [])


def test_cosine_segment_restarts_every_round():
    rec = Recorder()
    end = 0.1
    optimize(PiecewiseGrammar(step_target(64)), base_args(scheduler="CosineSegment", segment_end_factor=end), [rec])
    # StepEnd reports the LR after the step's scheduler update: step s of a segment of 20 starting at step k * 20
    for s, lr in enumerate(rec.lrs):
        t = s % 20 + 1
        want = 0.1 * (end + (1 - end) * 0.5 * (1 + math.cos(math.pi * min(t, 20) / 20)))
        assert lr == pytest.approx(want), s


def test_cosine_segment_scores_from_the_restart_lr():
    """Scoring at an event starts from 'lr', not from the decayed end of the round."""
    r1, r2 = Recorder(), Recorder()
    optimize(PiecewiseGrammar(step_target(64)), base_args(n_steps=21), [r1])
    optimize(PiecewiseGrammar(step_target(64)), base_args(n_steps=21, scheduler="CosineSegment", segment_end_factor=1.0), [r2])
    # end factor 1: the LR is constant, so the run equals the constant-LR run
    assert r1.rewrites[0].losses == r2.rewrites[0].losses
    assert r1.lrs == pytest.approx(r2.lrs)


def test_validation():
    with pytest.raises(ValueError):
        base_args(proposal_lr_mult=0.0)
    with pytest.raises(ValueError):
        base_args(segment_end_factor=1.5)


def test_warmup_ramps_after_every_rebuild():
    rec = Recorder()
    optimize(PiecewiseGrammar(step_target(64)), base_args(warmup_steps=5), [rec])
    for s, lr in enumerate(rec.lrs):  # rebuilds at every multiple of 20 (cleanup / rewrite steps)
        assert lr == pytest.approx(0.1 * min(1.0, (s % 20 + 2) / 5)), s  # StepEnd: after the step's scheduler update


def test_warmup_with_cosine_segment_composes():
    rec = Recorder()
    optimize(PiecewiseGrammar(step_target(64)), base_args(warmup_steps=5, scheduler="CosineSegment", segment_end_factor=0.1), [rec])
    for s, lr in enumerate(rec.lrs):
        t = s % 20 + 1
        cos = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(t, 20) / 20))
        assert lr == pytest.approx(0.1 * cos * min(1.0, (t + 1) / 5)), s


def test_proposal_warmup_changes_scores_not_the_main_loop():
    r1, r2 = Recorder(), Recorder()
    optimize(PiecewiseGrammar(step_target(64)), base_args(n_steps=21), [r1])
    optimize(PiecewiseGrammar(step_target(64)), base_args(n_steps=21, proposal_warmup_steps=10), [r2])
    assert r1.lrs[:20] == r2.lrs[:20]
    assert r1.rewrites[0].losses != r2.rewrites[0].losses


def test_warmup_validation():
    with pytest.raises(ValueError):
        base_args(warmup_steps=5, scheduler="LinearLR")
    with pytest.raises(ValueError):
        base_args(proposal_warmup_steps=-1)
