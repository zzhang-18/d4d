"""Tests of the string L-system toy grammar used in the README."""

from __future__ import annotations

import pytest
import torch
from lsystem import Expand, Turtle, TurtleGrammar, u_target

from d4d import OptimizeArgs, StepContext, optimize

CTX = StepContext(step=0, total_steps=1, phase="step")


def loss_of(g: TurtleGrammar, obj: Turtle) -> float:
    return g.loss(g.collate([obj]), CTX, None)[0].item()


def bent() -> Turtle:
    """A three-segment turtle that is not a straight line."""
    return Turtle("FRFRF", torch.tensor([0.7, 0.4, 0.9, 1.1, 0.5]))


def test_expand_rewrites_the_string():
    g = TurtleGrammar(u_target())
    out = g.apply(g.initial(), Expand(0))
    assert out.program == "FRF"
    assert out.params.tolist() == pytest.approx([0.5, 0.0, 0.5])


def test_propose_targets_every_F():
    g = TurtleGrammar(u_target())
    assert g.propose(bent(), budget=0) == [Expand(0), Expand(2), Expand(4)]
    assert len(g.propose(bent(), budget=2)) == 2


def test_expand_is_exactly_loss_preserving():
    """F -> F R(0) F redraws the same path, so the loss must not move."""
    g = TurtleGrammar(u_target())
    obj = bent()
    before = loss_of(g, obj)
    for rewrite in g.propose(obj, budget=0):
        assert loss_of(g, g.apply(obj, rewrite)) == pytest.approx(before, abs=1e-6), rewrite


def test_apply_all_expands_every_F_at_once():
    """conflicts() is False, so apply_all must splice all rewrites against base indices."""
    g = TurtleGrammar(u_target())
    obj = bent()
    rewrites = g.propose(obj, budget=0)
    assert not any(g.conflicts(a, b) for a in rewrites for b in rewrites)

    joint = g.apply_all(obj, rewrites)
    assert joint.program == "FRF" + "R" + "FRF" + "R" + "FRF"
    assert torch.allclose(g.vertices(joint)[-1], g.vertices(obj)[-1], atol=1e-6)
    assert loss_of(g, joint) == pytest.approx(loss_of(g, obj), abs=1e-6)


def test_grows_from_one_F_into_the_U():
    """A single straight F cannot draw a U; the grammar must add corners."""
    g = TurtleGrammar(u_target())
    res = optimize(g, OptimizeArgs(n_steps=600, lr=0.1, propose_every=50, w_simplicity=1e-3, seed=0))
    assert res.best.program.count("F") >= 3
    assert res.best_loss < 0.05 < loss_of(g, g.initial())
