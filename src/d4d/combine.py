"""Greedy acceptance of non-conflicting rewrites.

Every grammar in the upstream tree implements the same algorithm -- filter
proposals that improve on the base, sort by improvement descending, greedily
accept the ones that do not conflict -- and they differ only in what "conflict"
means. Hoisting it here means a grammar declares its conflict rule and inherits
the search.

The four conflict styles this has to subsume, all present upstream:

* **pairwise index** (arclines) -- two rewrites conflict if they touch a shared
  primitive index; ``AddHole`` conflicts with nothing. Accepted rewrites are
  applied to the *original* in one batch.
* **pairwise node id** (tree) -- same shape, keyed on node ids.
* **no conflict test at all** (ur) -- every improving rewrite is a candidate and
  all resolution happens inside ``UR.apply_all_rewrites(rewrites, scores, args)``,
  which sorts by score internally. This is why :meth:`Grammar.apply_all` receives
  ``improvements``: without the scores, ur is inexpressible.
* **stateful** (dice2) -- acceptance depends on the *partially rewritten* object,
  because each acceptance can invalidate a boundary constraint that a later
  rewrite was relying on. Hence the accumulator and ``incremental_apply``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from typing_extensions import TypeVar

if TYPE_CHECKING:
    from .grammar import Grammar

TObject = TypeVar("TObject")
TRewrite = TypeVar("TRewrite")

__all__ = ["DEFAULT_ACCEPT_RULE", "REJECT", "AcceptRule", "greedy_combine"]


class _Reject:
    """Sentinel returned by ``combine_admit`` to veto a rewrite.

    A distinct sentinel rather than ``None``, because ``None`` is a perfectly
    good accumulator value for the many grammars that need no state.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return "REJECT"

    def __bool__(self) -> bool:
        return False


REJECT = _Reject()


@dataclass(frozen=True)
class AcceptRule:
    """When is a proposal good enough to consider?

    Proposal ``i`` is a candidate iff ``base_loss - losses[i]`` exceeds a
    threshold built from an absolute and a relative floor.

    ``combine="min"`` reproduces arclines/tree/ur, which accept when *either*
    floor is cleared::

        better = imp > abs_eps or imp > rel_eps * base_loss

    ``combine="max"`` is the strict reading, requiring both.

    The default -- both floors zero -- reproduces dice2/dice's plain ``imp > 0``.

    Setting a floor is not cosmetic. With a noisy objective, ``abs_eps = 0``
    accepts noise, and because greedy-argmin over P proposals is a maximum-of-P
    estimator it accepts the *luckiest* one, biased by ``O(sigma * sqrt(2 ln P))``.
    If the objective is stochastic, either make evaluation deterministic or set
    ``abs_eps`` above the noise floor.
    """

    abs_eps: float = 0.0
    rel_eps: float = 0.0
    combine: Literal["min", "max"] = "min"

    def threshold(self, base_loss: float) -> float:
        a, r = self.abs_eps, self.rel_eps * abs(base_loss)
        return min(a, r) if self.combine == "min" else max(a, r)

    def better(self, base_loss: float, loss: float) -> bool:
        return (base_loss - loss) > self.threshold(base_loss)


DEFAULT_ACCEPT_RULE = AcceptRule()
"""Plain ``improvement > 0``. Shared because :class:`AcceptRule` is frozen."""


def greedy_combine(
    grammar: Grammar[TObject, Any, TRewrite, Any],
    base: TObject,
    rewrites: Sequence[TRewrite],
    base_loss: float,
    losses: Sequence[float],
    *,
    select_top_k: int = 0,
    rule: AcceptRule = DEFAULT_ACCEPT_RULE,
) -> tuple[TObject, list[TRewrite]]:
    """Accept as many non-conflicting improving rewrites as possible.

    Args:
        base: the object the rewrites were generated from.
        rewrites: candidate rewrites, positionally aligned with ``losses``.
        base_loss: the base object's loss, measured under the same conditions.
        losses: one loss per rewrite.
        select_top_k: stop after this many acceptances; ``0`` means unlimited.
        rule: the improvement threshold.

    Returns:
        ``(new_object, accepted_rewrites)``. When nothing is accepted, returns
        ``base`` itself and an empty list, so callers can test ``if accepted``.

    Ordering matches upstream exactly. Sorting ascending by ``losses[i]`` is
    equivalent to sorting descending by improvement, and Python's stable sort
    breaks ties by ascending index -- reproducing the upstream
    ``candidates.append((-imp, i)); candidates.sort()``.
    """
    if len(rewrites) != len(losses):
        raise ValueError(f"rewrites/losses length mismatch: {len(rewrites)} != {len(losses)}")
    if not rewrites:
        return base, []

    order = [i for i in range(len(rewrites)) if rule.better(base_loss, losses[i])]
    if not order:
        return base, []
    order.sort(key=lambda i: losses[i])

    acc: Any = grammar.combine_init(base)
    partial = base
    accepted: list[TRewrite] = []
    improvements: list[float] = []

    for i in order:
        rewrite = rewrites[i]
        new_acc = grammar.combine_admit(base, partial, acc, rewrite, accepted)
        if new_acc is REJECT:
            continue
        acc = new_acc
        accepted.append(rewrite)
        improvements.append(base_loss - losses[i])
        if grammar.incremental_apply:
            # dice2-style: later admissibility checks read the partial result.
            partial = grammar.apply(partial, rewrite)
        if select_top_k > 0 and len(accepted) >= select_top_k:
            break

    if not accepted:
        return base, []
    if grammar.incremental_apply:
        return partial, accepted
    return grammar.apply_all(base, accepted, improvements), accepted
