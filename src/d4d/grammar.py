"""The grammar interface, in five stages.

1. **construct** -- ``initial()`` returns the starting object; ``collate()`` packs
   objects into a differentiable batch.
2. **rewrite** -- ``propose()`` enumerates candidate rewrites, ``apply()`` performs one.
3. **combine** -- ``conflicts()`` decides which improving rewrites can coexist,
   ``apply_all()`` applies them together; override ``combine()`` for anything else.
4. **loss** -- ``loss()`` scores a batch differentiably, ``simplicity()`` prices
   program size without gradient.
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

if TYPE_CHECKING:  # numpy is not a runtime dependency
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
    """Where the run is when the grammar is called.

    ``phase`` is ``"step"`` for the descent step, ``"proposal"`` while scoring
    candidate rewrites, and ``"visualize"`` for :meth:`Grammar.visualize`.
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
        """Fraction of the run elapsed, in ``[0, 1]``."""
        return self.step / self.total_steps if self.total_steps > 0 else 0.0

    def __repr__(self) -> str:
        return (
            f"StepContext(step={self.step}/{self.total_steps}, phase={self.phase!r}, "
            f"lr={self.lr:.3g}, n_objects={self.n_objects})"
        )


class Grammar(ABC, Generic[TObject, TCollection, TRewrite, TState]):
    """Base class for a grammar. ``initial``, ``propose``, ``apply`` and ``loss`` are abstract.

    Type parameters are ``Grammar[TObject, TCollection, TRewrite, TState]``, with
    defaults ``ListCollection[Any]``, ``Any`` and ``None`` for the last three::

        class MyGrammar(Grammar[MyObject]): ...
        class MyGrammar(Grammar[MyObject, ListCollection[MyObject], MyRewrite, None]): ...

    Set :attr:`list_spec` or override :meth:`collate`.
    """

    # ---- construction ----------------------------------------------------

    list_spec: ListSpec[Any] | None = None
    """Enables the default :meth:`collate`, which builds a :class:`~d4d.collection.ListCollection`."""

    def collate(self, objects: Sequence[TObject]) -> TCollection:
        """Pack objects into one batch; :meth:`loss` must differentiate through its ``parameters()``.

        The optimizer builds every batch through this. The default needs :attr:`list_spec`.
        """
        if self.list_spec is None:
            raise NotImplementedError(
                f"{type(self).__name__} must either override collate() or set "
                f"list_spec = ListSpec(params_of=..., with_params=...)"
            )
        return ListCollection.build(list(objects), self.list_spec)  # type: ignore[return-value]

    @abstractmethod
    def initial(self) -> TObject:
        """The starting object."""

    def object_cost(self, obj: TObject) -> int:
        """Relative cost of one object in a batch, for :func:`~d4d.collection.batchify`.

        Default: the object's parameter element count.
        """
        return sum(t.numel() for t in self.collate([obj]).parameters())

    # ---- rewriting -------------------------------------------------------

    @abstractmethod
    def propose(self, obj: TObject, budget: int) -> list[TRewrite]:
        """Up to ``budget`` candidate rewrites of ``obj`` (``0`` = unlimited).

        The optimizer materializes each one with :meth:`apply`.
        """

    @abstractmethod
    def apply(self, obj: TObject, rewrite: TRewrite) -> TObject:
        """Perform one rewrite. Must not mutate ``obj``."""

    def apply_all(
        self, base: TObject, rewrites: Sequence[TRewrite], improvements: Sequence[float]
    ) -> TObject:
        """Apply a set of rewrites to ``base``; called once by the default :meth:`combine`.

        ``improvements`` is aligned with ``rewrites``. The default folds
        :meth:`apply` over ``rewrites`` in order.
        """
        out = base
        for rewrite in rewrites:
            out = self.apply(out, rewrite)
        return out

    def conflicts(self, a: TRewrite, b: TRewrite) -> bool:
        """Whether ``a`` and ``b`` cannot both be accepted. Default: True, so one rewrite per event."""
        return True

    def combine(
        self,
        base: TObject,
        ranked: Sequence[TRewrite],
        improvements: Sequence[float],
        top_k: int = 0,
    ) -> tuple[TObject, list[TRewrite]]:
        """Pick a compatible subset of ``ranked`` and apply it to ``base``.

        ``ranked`` is non-empty and holds the proposals that cleared the
        acceptance floors, best first; ``improvements`` is aligned with it.
        Returns the new object and the applied rewrites, at most ``top_k`` of them
        (``0`` = unlimited). The default keeps each rewrite that :meth:`conflicts`
        with none already kept, then calls :meth:`apply_all` once.
        """
        accepted: list[TRewrite] = []
        kept: list[float] = []
        for rewrite, imp in zip(ranked, improvements):
            if any(self.conflicts(other, rewrite) for other in accepted):
                continue
            accepted.append(rewrite)
            kept.append(imp)
            if top_k > 0 and len(accepted) >= top_k:
                break
        return self.apply_all(base, accepted, kept), accepted

    # ---- evaluation ------------------------------------------------------

    @abstractmethod
    def loss(
        self, batch: TCollection, ctx: StepContext, state: TState
    ) -> tuple[torch.Tensor, ExtraMetrics]:
        """``(B,)`` differentiable, unreduced per-object losses, and ``extra``.

        ``extra`` maps a name to ``B`` values. It is logged when
        ``ctx.compute_extra`` is True (the step phase) and ignored otherwise.
        """

    def simplicity(self, batch: TCollection, ctx: StepContext) -> Sequence[float]:
        """``B`` program-size prices; not differentiated.

        Weighted by ``OptimizeArgs.w_simplicity`` in proposal ranking and in ``$loss``.
        """
        return [0.0] * len(batch)

    def visualize(
        self, batch: TCollection, ctx: StepContext, state: TState
    ) -> np.ndarray | None:
        """An ``(H, W, 3)`` uint8 frame, or None.

        Called every ``visualize_every`` steps and on rewrite steps.
        """
        return None

    # ---- state and lifecycle ---------------------------------------------

    def init_state(self) -> TState:
        """Initial per-run state passed to :meth:`loss` and :meth:`visualize`. Default: None."""
        return None  # type: ignore[return-value]

    def state_for_proposals(self, state: TState) -> TState:
        """State for scoring the base and every candidate of one rewrite event; called once per event."""
        return state

    def step_state(self, state: TState) -> TState:
        """Advance the state by one optimization step."""
        return state

    def cleanup(self, batch: TCollection) -> TCollection:
        """Canonicalize a batch; called every ``cleanup_every`` steps."""
        return batch

    # ---- reproducibility -------------------------------------------------

    def config(self) -> Mapping[str, Any]:
        """JSON-able hyperparameters, written by ``ConfigWriter``."""
        return {}

    def elapsed(self) -> float:
        """Seconds since the run started, for the ``$timestamp`` metric."""
        start = getattr(self, "_d4d_start_time", None)
        if start is None:
            start = time.time()
            object.__setattr__(self, "_d4d_start_time", start)
        return time.time() - start
