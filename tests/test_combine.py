"""Unit tests for acceptance: the optimizer's ranking and the grammar's default combine."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pytest
from toy import PiecewiseGrammar, step_target

from d4d import Grammar, OptimizeArgs, optimize
from d4d.optimize import _rank_improving


@dataclass(frozen=True)
class Touch:
    """A rewrite that occupies a set of slots and appends a tag to the object."""

    slots: frozenset[int]
    tag: str


class SlotGrammar(Grammar[tuple[str, ...], Any, Touch, None]):
    """Minimal grammar: an object is a tuple of applied tags."""

    def __init__(self) -> None:
        self.apply_all_calls: list[list[Touch]] = []

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

    def apply_all(self, base, rewrites):
        self.apply_all_calls.append(list(rewrites))
        return super().apply_all(base, rewrites)


def touch(tag: str, *slots: int) -> Touch:
    return Touch(frozenset(slots), tag)


class TestCombine:
    def test_accepts_all_non_conflicting_best_first(self):
        g = SlotGrammar()
        ranked = [touch("b", 1), touch("c", 0), touch("a", 0)]  # c blocks a (shares slot 0)
        obj, accepted = g.combine((), ranked, [3.0, 2.0, 1.0])
        assert [r.tag for r in accepted] == ["b", "c"]
        assert obj == ("b", "c")

    def test_top_k_limits_acceptances(self):
        g = SlotGrammar()
        ranked = [touch("c", 2), touch("b", 1), touch("a", 0)]
        _, accepted = g.combine((), ranked, [3.0, 2.0, 1.0], top_k=1)
        assert [r.tag for r in accepted] == ["c"]

    def test_all_conflicting_accepts_exactly_one(self):
        """When every pair conflicts, only the best rewrite is accepted."""

        class AllConflict(SlotGrammar):
            def conflicts(self, a: Touch, b: Touch) -> bool:
                return True

        g = AllConflict()
        ranked = [touch("c", 2), touch("b", 1), touch("a", 0)]
        _, accepted = g.combine((), ranked, [3.0, 2.0, 1.0])
        assert [r.tag for r in accepted] == ["c"]

    def test_apply_all_gets_the_kept_set_once(self):
        """apply_all is called once, with only the kept rewrites, best first."""
        g = SlotGrammar()
        ranked = [touch("b", 1), touch("c", 0), touch("a", 0)]
        g.combine((), ranked, [3.0, 2.0, 1.0])
        assert [[r.tag for r in rw] for rw in g.apply_all_calls] == [["b", "c"]]


class TestRankImproving:
    def test_default_is_strictly_positive(self):
        assert _rank_improving(OptimizeArgs(), 10.0, [10.0, 11.0, 9.999]) == [2]

    def test_best_first_ties_by_index(self):
        assert _rank_improving(OptimizeArgs(), 10.0, [9.0, 7.0, 9.0, 8.0]) == [1, 3, 0, 2]

    def test_or_accepts_if_either_floor_cleared(self):
        args = OptimizeArgs(accept_abs_eps=1.0, accept_rel_eps=0.001, accept_eps_op="or")
        assert _rank_improving(args, 100.0, [99.5]) == [0]  # clears rel (0.1) but not abs (1.0)

    def test_and_requires_both(self):
        args = OptimizeArgs(accept_abs_eps=1.0, accept_rel_eps=0.001, accept_eps_op="and")
        assert _rank_improving(args, 100.0, [99.5, 98.5]) == [1]

    def test_none_disables_a_floor(self):
        abs_only = OptimizeArgs(accept_abs_eps=1.0, accept_eps_op="or")
        rel_only = OptimizeArgs(accept_rel_eps=0.001, accept_eps_op="and")
        assert _rank_improving(abs_only, 100.0, [99.5]) == []
        assert _rank_improving(rel_only, 100.0, [99.5]) == [0]

    def test_rel_eps_uses_absolute_base_loss(self):
        """A negative base loss must not invert the floor into accepting regressions."""
        args = OptimizeArgs(accept_rel_eps=0.1)
        assert _rank_improving(args, -100.0, [-95.0, -105.0, -111.0]) == [2]

    def test_rejects_unknown_op(self):
        with pytest.raises(ValueError, match="accept_eps_op"):
            OptimizeArgs(accept_eps_op="xor")  # type: ignore[arg-type]


def test_optimize_hands_combine_ranked_improving_candidates():
    """The contract Grammar.combine documents: non-empty, filtered, best first, aligned."""

    class Spy(PiecewiseGrammar):
        def __init__(self, *a, **kw) -> None:
            super().__init__(*a, **kw)
            self.calls: list[tuple[list[float], int]] = []

        def combine(self, base, ranked, improvements, top_k=0):
            self.calls.append((list(improvements), top_k))
            assert len(ranked) == len(improvements) > 0
            return super().combine(base, ranked, improvements, top_k)

    g = Spy(step_target(64), n_initial=1, allow_remove=False)
    args = OptimizeArgs(
        n_steps=100, lr=0.1, clip_grad=None, propose_every=20, visualize_every=0,
        w_simplicity=0.0, accept_abs_eps=1e-9, accept_top_k=3, seed=0,
    )
    optimize(g, args)

    assert g.calls, "combine was never called"
    for improvements, top_k in g.calls:
        assert top_k == 3
        assert all(imp > 1e-9 for imp in improvements)
        assert improvements == sorted(improvements, reverse=True)
