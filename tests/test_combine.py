"""Unit tests for the greedy acceptance search."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pytest

from d4d import DEFAULT_ACCEPT_RULE, REJECT, AcceptRule, Grammar, greedy_combine


@dataclass(frozen=True)
class Touch:
    """A rewrite that occupies a set of slots and appends a tag to the object."""

    slots: frozenset[int]
    tag: str


class SlotGrammar(Grammar[tuple[str, ...], Any, Touch, None]):
    """Minimal grammar: an object is a tuple of applied tags."""

    def __init__(self, incremental: bool = False, rule: AcceptRule = DEFAULT_ACCEPT_RULE) -> None:
        self.incremental_apply = incremental
        self.accept_rule = rule
        self.apply_all_calls = 0

    def initial(self) -> tuple[str, ...]:
        return ()

    def collate(self, objects: Sequence[tuple[str, ...]]) -> Any:
        raise NotImplementedError("combine() never collates")

    def propose(self, obj: tuple[str, ...], budget: int) -> list[Touch]:
        return []

    def apply(self, obj: tuple[str, ...], rewrite: Touch) -> tuple[str, ...]:
        return obj + (rewrite.tag,)

    def loss(self, batch, ctx, state):
        raise NotImplementedError("combine() never evaluates a loss")

    def conflicts(self, a: Touch, b: Touch) -> bool:
        return bool(a.slots & b.slots)

    def apply_all(self, base, rewrites, improvements):
        self.apply_all_calls += 1
        return super().apply_all(base, rewrites, improvements)


def touch(tag: str, *slots: int) -> Touch:
    return Touch(frozenset(slots), tag)


def test_accepts_all_non_conflicting_best_first():
    g = SlotGrammar()
    rewrites = [touch("a", 0), touch("b", 1), touch("c", 0)]
    # improvements: a=1.0, b=3.0, c=2.0 -> order b, c, a; c blocks a (shares slot 0)
    obj, accepted = greedy_combine(g, (), rewrites, base_loss=10.0, losses=[9.0, 7.0, 8.0])
    assert [r.tag for r in accepted] == ["b", "c"]
    assert obj == ("b", "c")


def test_rejects_non_improving():
    g = SlotGrammar()
    rewrites = [touch("a", 0), touch("b", 1)]
    obj, accepted = greedy_combine(g, (), rewrites, base_loss=10.0, losses=[10.0, 11.0])
    assert accepted == []
    assert obj == ()  # base returned unchanged


def test_select_top_k_limits_acceptances():
    g = SlotGrammar()
    rewrites = [touch("a", 0), touch("b", 1), touch("c", 2)]
    _, accepted = greedy_combine(
        g, (), rewrites, base_loss=10.0, losses=[9.0, 8.0, 7.0], select_top_k=1
    )
    assert [r.tag for r in accepted] == ["c"]


def test_default_conflicts_accepts_exactly_one():
    """The base Grammar.conflicts says everything conflicts."""

    class AllConflict(SlotGrammar):
        def conflicts(self, a: Touch, b: Touch) -> bool:
            return Grammar.conflicts(self, a, b)

    g = AllConflict()
    rewrites = [touch("a", 0), touch("b", 1), touch("c", 2)]
    _, accepted = greedy_combine(g, (), rewrites, base_loss=10.0, losses=[9.0, 8.0, 7.0])
    assert len(accepted) == 1


def test_ties_break_by_index_ascending():
    g = SlotGrammar()
    rewrites = [touch("a", 0), touch("b", 0)]
    _, accepted = greedy_combine(g, (), rewrites, base_loss=10.0, losses=[9.0, 9.0])
    assert [r.tag for r in accepted] == ["a"]


def test_empty_rewrites_is_a_noop():
    g = SlotGrammar()
    obj, accepted = greedy_combine(g, ("x",), [], base_loss=1.0, losses=[])
    assert obj == ("x",) and accepted == []


def test_length_mismatch_raises():
    g = SlotGrammar()
    with pytest.raises(ValueError, match="length mismatch"):
        greedy_combine(g, (), [touch("a", 0)], base_loss=1.0, losses=[])


def test_incremental_apply_folds_and_skips_apply_all():
    g = SlotGrammar(incremental=True)
    rewrites = [touch("a", 0), touch("b", 1)]
    obj, _ = greedy_combine(g, (), rewrites, base_loss=10.0, losses=[8.0, 9.0])
    assert obj == ("a", "b")
    assert g.apply_all_calls == 0  # incremental path must not call apply_all


def test_batch_apply_uses_apply_all():
    g = SlotGrammar(incremental=False)
    rewrites = [touch("a", 0), touch("b", 1)]
    greedy_combine(g, (), rewrites, base_loss=10.0, losses=[8.0, 9.0])
    assert g.apply_all_calls == 1


def test_combine_admit_can_veto_statefully():
    """An accumulator can enforce a budget no pairwise test could express."""

    class BudgetGrammar(SlotGrammar):
        def combine_init(self, base):
            return 0

        def combine_admit(self, base, partial, acc, rewrite, accepted):
            if acc + len(rewrite.slots) > 2:  # total slots capped at 2
                return REJECT
            return acc + len(rewrite.slots)

    g = BudgetGrammar()
    rewrites = [touch("a", 0, 1), touch("b", 2), touch("c", 3)]
    _, accepted = greedy_combine(g, (), rewrites, base_loss=10.0, losses=[7.0, 8.0, 9.0])
    assert [r.tag for r in accepted] == ["a"]  # a uses the whole budget


class TestAcceptRule:
    def test_default_is_strictly_positive(self):
        r = AcceptRule()
        assert r.better(1.0, 0.999) and not r.better(1.0, 1.0)

    def test_min_accepts_if_either_floor_cleared(self):
        r = AcceptRule(abs_eps=1.0, rel_eps=0.001, combine="min")
        assert r.better(100.0, 99.5)  # clears rel (0.1) but not abs (1.0)

    def test_max_requires_both(self):
        r = AcceptRule(abs_eps=1.0, rel_eps=0.001, combine="max")
        assert not r.better(100.0, 99.5)
        assert r.better(100.0, 98.5)

    def test_rel_eps_uses_absolute_base_loss(self):
        """A negative base loss must not invert the threshold into acceptance."""
        r = AcceptRule(abs_eps=1e9, rel_eps=0.1, combine="min")
        assert r.threshold(-100.0) == pytest.approx(10.0)
        assert r.threshold(100.0) == pytest.approx(10.0)
