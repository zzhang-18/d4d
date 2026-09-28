"""
Shape grammar interface: 

1. **construct** -- ``initial()`` returns the starting object; ``collate()`` packs
   objects into a differentiable batch.
2. **rewrite** -- ``propose()`` enumerates candidate rewrites, ``apply()`` performs rewrites.
3. **combine** -- ``conflicts()`` (or the accumulator hooks) decides which
   accepted rewrites can coexist.
4. **loss** -- ``loss()`` scores a batch differentiably, ``simplicity()`` computes
   program length for the discrete step only.
5. **visualize** -- ``visualize()`` returns a frame for callbacks.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Generic, Literal

import torch
from typing_extensions import TypeVar

from .collection import ListCollection, ListSpec, ObjectCollection
from .combine import DEFAULT_ACCEPT_RULE, REJECT, AcceptRule, greedy_combine

if TYPE_CHECKING:  # keeps numpy out of the runtime dependency set
    import numpy as np

__all__ = ["ExtraMetrics", "Grammar", "Phase", "StepContext"]

ExtraMetrics = Mapping[str, Sequence[float]]
"""Per-object diagnostics returned alongside the loss. Each value has length ``len(batch)``."""

Phase = Literal["step", "proposal", "visualize"]

TObject = TypeVar("TObject")
TCollection = TypeVar("TCollection", bound=ObjectCollection[Any], default=ListCollection[Any])
TRewrite = TypeVar("TRewrite", default=Any)
TState = TypeVar("TState", default=None)


class StepContext:
    """Context of the run at a point in time.

    For the field ``phase``, ``"proposal"`` means the grammar is being
    asked to score candidate rewrites rather than take a real step, which is
    permission to evaluate more cheaply -- fewer sample points, a coarser
    resolution -- and, for a stochastic objective, an instruction to *freeze* its
    randomness so that ``base_loss - proposal_loss`` measures the improvement
    more accurately.
    """

    __slots__ = (
        "compute_extra",
        "lr",
        "n_objects",
        "phase",
        "proposal_step",
        "rewrite",
        "rewrite_index",
        "step",
        "total_steps",
    )

    def __init__(
        self,
        step: int,
        total_steps: int,
        phase: Phase,
        compute_extra: bool = False,
        lr: float = 0.0,
        rewrite: bool = False,
        rewrite_index: int = 0,
        proposal_step: int = 0,
        n_objects: int = 1,
    ) -> None:
        self.step = step
        self.total_steps = total_steps
        self.phase: Phase = phase
        self.compute_extra = compute_extra
        self.lr = lr
        self.rewrite = rewrite
        self.rewrite_index = rewrite_index
        self.proposal_step = proposal_step
        self.n_objects = n_objects

    @property
    def progress(self) -> float:
        """Fraction of the run elapsed in ``[0, 1]``. Handy for annealing schedules."""
        return self.step / self.total_steps if self.total_steps > 0 else 0.0

    def __repr__(self) -> str:
        return (
            f"StepContext(step={self.step}/{self.total_steps}, phase={self.phase!r}, "
            f"lr={self.lr:.3g}, n_objects={self.n_objects})"
        )


class Grammar(ABC, Generic[TObject, TCollection, TRewrite, TState]):
    """Define a grammar by subclassing this and filling in five stages.

    Type parameters, of which only the first two usually need naming::

        class SFGrammar(Grammar[SFAssembly, SFAssemblyBatch]):
            ...

    ``TRewrite`` defaults to ``Any`` and ``TState`` to ``None``, so a grammar with
    untyped rewrites and no state writes ``Grammar[MyObject]``. A grammar that
    also wants its rewrites checked writes all four.

    The minimum viable grammar implements ``initial``, ``propose``, ``apply`` and
    ``loss``, and sets :attr:`list_spec` so the default ``collate`` works.
    """

    # ---- construction ----------------------------------------------------

    list_spec: ListSpec[Any] | None = None
    """Set this to get a working :meth:`collate` for free.

    Two small callables -- how to read an object's tensors, and how to rebuild an
    object around new ones -- are enough for the default
    :class:`~d4d.collection.ListCollection`. Override :meth:`collate` instead when
    a packed representation is worth the code.
    """

    def collate(self, objects: Sequence[TObject]) -> TCollection:
        """Pack objects into one differentiable batch.

        This is the only function for constructing collections.

        The batch must be differentiable end to end: ``parameters()`` are the
        leaves the optimizer steps, and :meth:`loss` must reach them.
        """
        if self.list_spec is None:
            raise NotImplementedError(
                f"{type(self).__name__} must either override collate() or set "
                f"list_spec = ListSpec(params_of=..., with_params=...)"
            )
        return ListCollection.build(list(objects), self.list_spec)  # type: ignore[return-value]

    @abstractmethod
    def initial(self) -> TObject:
        """The starting object -- the ``init_shape`` the run descends from."""

    def object_cost(self, obj: TObject) -> int:
        """Relative cost of holding one object in a batch, for :func:`~d4d.collection.batchify`.

        The default -- parameter element count -- is right when memory scales with
        parameters. It is wrong whenever peak memory is dominated by something
        else: for an SDF grammar the activation cost is ``primitives x sample
        points`` and the parameter count is a constant per object, uncorrelated
        with what actually fills the GPU. Override it there.
        """
        return sum(t.numel() for t in self.collate([obj]).parameters())

    # ---- rewriting -------------------------------------------------------

    @abstractmethod
    def propose(self, obj: TObject, budget: int) -> list[TRewrite]:
        """Enumerate candidate rewrites, at most ``budget`` of them (``0`` = unlimited).

        The budget is passed *down* rather than applied afterwards, so a grammar
        can subsample before doing expensive work, and can split the budget
        across rule families. Upstream, subsampling happened after materializing
        every proposal, which is why four of five grammars had to override it.

        This returns rewrite descriptions only. The optimizer materializes
        candidates by calling :meth:`apply`, so a rewrite that is generated but
        never scored costs nothing.
        """

    @abstractmethod
    def apply(self, obj: TObject, rewrite: TRewrite) -> TObject:
        """Perform one rewrite. Must be pure -- do not mutate ``obj``."""

    def apply_all(
        self, base: TObject, rewrites: Sequence[TRewrite], improvements: Sequence[float]
    ) -> TObject:
        """Apply an accepted *set* of rewrites to ``base``.

        Called once, after the greedy search, when :attr:`incremental_apply` is
        False. The default folds :meth:`apply` left to right.

        Override when rewrites must be resolved jointly -- when indices in one
        rewrite refer to positions the previous one shifted, or when the grammar
        wants to arbitrate among the accepted set itself. ``improvements`` carries
        each rewrite's loss improvement for grammars that need to re-rank
        internally.
        """
        out = base
        for rewrite in rewrites:
            out = self.apply(out, rewrite)
        return out

    incremental_apply: bool = False
    """Whether admissibility depends on the partially-rewritten object.

    False (default): candidates are judged against ``base``, and the accepted set
    is materialized once via :meth:`apply_all`. This is right when rewrites are
    described in terms of the original's indices.

    True: the optimizer folds :meth:`apply` as it goes and hands the running
    result to :meth:`combine_admit`. Needed when accepting one rewrite can
    invalidate another -- for instance when rewrites must preserve a shared
    boundary that an earlier acceptance may already have changed.

    A plain class attribute rather than a ``ClassVar`` so an instance can set it
    from configuration, which also keeps tests from having to mutate the class.
    """

    accept_rule: AcceptRule = DEFAULT_ACCEPT_RULE
    """How much improvement a proposal must show. See :class:`~d4d.combine.AcceptRule`."""

    def conflicts(self, a: TRewrite, b: TRewrite) -> bool:
        """Whether two rewrites can be applied at the same time. Defaults to True.

        The default says everything conflicts, so exactly one rewrite is accepted
        per rewrite step. That is always *correct* and usually slow; a grammar
        that can characterize independence -- disjoint indices, disjoint regions --
        should say so, and gets parallel acceptance for free.
        """
        return True

    def combine_init(self, base: TObject) -> Any:
        """Seed the accumulator threaded through :meth:`combine_admit`."""
        return None

    def combine_admit(
        self,
        base: TObject,
        partial: TObject,
        acc: Any,
        rewrite: TRewrite,
        accepted: Sequence[TRewrite],
    ) -> Any:
        """Admit or veto ``rewrite``, given what has been accepted so far.

        Return the updated accumulator to admit, or
        :data:`~d4d.combine.REJECT` to veto.

        The default runs the pairwise :meth:`conflicts` test against everything
        already accepted. Override for constraints that are not pairwise -- an
        accumulated mask, a running budget, a validity check against ``partial``
        (which is only meaningful when :attr:`incremental_apply` is True).
        """
        for other in accepted:
            if self.conflicts(other, rewrite):
                return REJECT
        return acc

    def combine(
        self,
        base: TObject,
        rewrites: Sequence[TRewrite],
        base_loss: float,
        losses: Sequence[float],
        select_top_k: int = 0,
    ) -> tuple[TObject, list[TRewrite]]:
        """Choose and apply a set of rewrites. Rarely worth overriding.

        The default -- sort by improvement, greedily accept the admissible --
        covers every grammar in the upstream tree via :meth:`conflicts`,
        :meth:`combine_admit` and :meth:`apply_all`.
        """
        return greedy_combine(
            self, base, rewrites, base_loss, losses,
            select_top_k=select_top_k, rule=self.accept_rule,
        )

    # ---- evaluation ------------------------------------------------------

    @abstractmethod
    def loss(
        self, batch: TCollection, ctx: StepContext, state: TState
    ) -> tuple[torch.Tensor, ExtraMetrics]:
        """Score a batch. **Differentiable** -- this is what descent follows.

        Returns ``(losses, extra)`` where ``losses`` has shape ``(len(batch),)``
        -- one independent loss per object, *not* reduced. The optimizer sums them
        so that a batch of candidates trains in one backward pass while staying
        independent.

        ``extra`` maps a diagnostic name to one value per object; it is logged and
        forwarded to callbacks. Respect ``ctx.compute_extra`` and skip anything
        expensive when it is False -- it is False for every proposal evaluation.
        """

    def simplicity(self, batch: TCollection, ctx: StepContext) -> Sequence[float]:
        """Price program length, per object. This is **not** differentiated.

        This is what stops the grammar growing without bound. It enters only two
        places: the ranking of proposals (scaled by ``w_simplicity``), and the
        logged ``$loss``. It contributes no gradient, so it can be a plain count
        -- of primitives, nodes, symbols -- with no smooth relaxation.

        Keeping it out of the gradient is deliberate: a differentiable
        program-length penalty pushes every parameter toward degeneracy, whereas a
        discrete one is paid only when the discrete step actually adds something.
        """
        return [0.0] * len(batch)

    def visualize(
        self, batch: TCollection, ctx: StepContext, state: TState
    ) -> np.ndarray | None:
        """Render a frame for the callbacks to persist, or None for no frame.

        Called on visualization steps and after every rewrite. Returning an
        ``(H, W, 3)`` uint8 array is the convention; callbacks that write PNGs and
        videos assume it.

        Keep it cheap, or raise ``visualize_every``. If the underlying evaluation
        is stochastic, render deterministically -- a frame that resamples its own
        randomness shows noise, not progress.
        """
        return None

    # ---- state and lifecycle ---------------------------------------------

    def init_state(self) -> TState:
        """Mutable per-run state threaded through :meth:`loss` and :meth:`visualize`.

        For anything that evolves with the run but is not a parameter: annealing
        temperatures, resampled point sets, an adaptive multiplier. Grammars
        without state ignore this entirely -- ``TState`` defaults to ``None``.
        """
        return None  # type: ignore[return-value]

    def state_for_proposals(self, state: TState) -> TState:
        """Derive the state used to score proposals, once per rewrite event.

        Called exactly once per rewrite -- not once per proposal and not once per
        batch -- so that every candidate *and the base* are scored under identical
        conditions. That is what makes ``base_loss - proposal_loss`` attributable
        to the rewrite. Any resampling or noise draw that must be shared across
        the comparison belongs here.
        """
        return state

    def step_state(self, state: TState) -> TState:
        """Advance the state by one optimization step."""
        return state

    def cleanup(self, batch: TCollection) -> TCollection:
        """Canonicalize a batch, periodically and before every rewrite.

        The place for housekeeping that would otherwise accumulate: merging
        duplicates, dropping degenerate elements, resolving self-intersections.
        Rewrites are generated from the cleaned object, so this also controls what
        the grammar is even able to see.
        """
        return batch

    # ---- reproducibility -------------------------------------------------

    def config(self) -> Mapping[str, Any]:
        """A JSON-able snapshot of this grammar's hyperparameters.

        Written verbatim into the run's config file. The core has no opinion on
        how a grammar is configured -- it never reads these values -- but a run
        that cannot say what produced it is not reproducible.
        """
        return {}

    def elapsed(self) -> float:
        """Seconds since the run started, for the ``$timestamp`` metric."""
        start = getattr(self, "_d4d_start_time", None)
        if start is None:
            start = time.time()
            object.__setattr__(self, "_d4d_start_time", start)
        return time.time() - start
