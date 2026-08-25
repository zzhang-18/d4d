"""Batches of objects, and how the optimizer packs them.

The central abstraction the optimizer descends on. An ``ObjectCollection`` is a
batch of ``TObject`` whose parameters are exposed as a flat list of *leaf*
tensors, so that N candidate rewrites can be optimized simultaneously by a
single ``torch.optim`` instance with their losses summed.

Two deliberate departures from the upstream ``d4descent.object_collection``:

1. **No ``Renderable``.** Upstream, ``ObjectCollection`` inherits an ABC with an
   abstract ``rasterize(positions: (...A, 2))`` and concrete 2D SDF
   ``render``/``render01``. That welds 2D rasterization into the algorithm's
   core abstraction, so any non-2D grammar -- 3D assemblies, graphs, programs --
   must implement a meaningless method. Rendering is a *grammar* concern and
   lives in the grammar.

2. **No classmethods.** Upstream ``batchify`` does ``type(self).from_object``
   and ``__getitem__(slice)`` does ``self.__class__.from_objects``, which is
   precisely why grammars need dynamic-subclass ``patch_args`` hacks to thread
   per-collection configuration: the optimizer only ever holds a *class*, so
   there is nowhere to put the config. Here construction goes through
   ``Grammar.collate``, a bound method that closes over the grammar's own
   configuration, and ``batchify`` is a free function taking the grammar.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, Self

import torch
from typing_extensions import TypeVar

if TYPE_CHECKING:
    from .grammar import Grammar

TObject = TypeVar("TObject")

__all__ = ["ListCollection", "ListSpec", "ObjectCollection", "batchify"]

Device = str | torch.device | None



class ObjectCollection(ABC, Sequence[TObject]):
    """A batch of objects whose parameters are contiguous leaf tensors.

    Implementations are free to store whatever packed representation they like
    -- a stacked ``(P, N, D)`` tensor, a global buffer with per-object index
    metadata, or a plain list. The only contract is the methods below.

    ``parameters()`` must return *leaf* tensors (``requires_grad`` set, no
    ``grad_fn``); they are handed straight to ``torch.optim``.
    ``per_object_grads()`` must return exactly ``len(self)`` vectors regardless
    of how parameters are packed -- that is what lets the first-order proposal
    criterion attribute a descent direction to each candidate independently.

    **Invariant.** Whatever ``get(i)`` returns must share parameter tensors with
    (or be rebuilt from) the ones ``parameters()`` exposes. If a subclass
    re-leafs tensors in ``requires_grad_``, it must also refresh the objects, or
    the loss will be computed from stale tensors that receive no gradient.
    """

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def get(self, idx: int, detach: bool = True) -> TObject:
        """Extract a single object. ``detach=True`` severs it from the graph."""

    @abstractmethod
    def parameters(self) -> list[torch.Tensor]:
        """Leaf tensors handed to the torch optimizer."""

    @abstractmethod
    def parameter_names(self) -> list[str]:
        """Human-readable names, positionally aligned with ``parameters()``."""

    @abstractmethod
    def per_object_grads(self) -> list[torch.Tensor]:
        """One flat gradient vector per object. ``len() == len(self)``.

        Only consumed by ``proposal_criterion in ("grad", "grad_only")``, which
        forms the first-order surrogate ``loss - lr * <g, g>``. An object whose
        parameters carry no ``.grad`` must contribute an *empty* vector rather
        than being omitted -- the optimizer zips these against losses
        positionally.
        """

    @abstractmethod
    def requires_grad_(self, requires_grad: bool = True) -> Self:
        """Re-leaf every parameter. Called after every rewrite and cleanup."""

    @abstractmethod
    def clone(self) -> Self: ...

    @abstractmethod
    def to(self, device: Device = None) -> Self: ...

    def scale_grads_(self) -> bool:
        """Rescale gradients in place; return True if anything changed.

        The return value is load-bearing: the optimizer recomputes
        ``per_object_grads`` only when this reports True, so a grammar applying
        per-parameter LR multipliers here must say so.
        """
        return False

    def project_to_valid_(self) -> Self:
        """Project parameters back into the feasible set after an optimizer step.

        A no-op for grammars whose parameters are unconstrained, or already
        reparameterized onto an unbounded domain.
        """
        return self

    # -- Sequence -----------------------------------------------------------
    def __getitem__(self, idx: int) -> TObject:  # type: ignore[override]
        return self.get(idx, True)

    def __iter__(self) -> Iterator[TObject]:
        for i in range(len(self)):
            yield self.get(i, True)


@dataclass(frozen=True)
class ListSpec(Generic[TObject]):
    """How to take one object apart and put it back together.

    Supplying this is the cheapest way to get a working grammar: the default
    ``Grammar.collate`` builds a :class:`ListCollection` from it, so a grammar
    need not define a collection type at all.

    ``with_params`` is not optional. Re-leafing in ``requires_grad_`` creates
    *new* tensors, and without a way to push them back into the object, the
    objects and the optimizer would reference different memory -- the loss would
    be computed from tensors the optimizer never steps.
    """

    params_of: Callable[[TObject], list[torch.Tensor]]
    """Extract the leaf tensors from an object, in a stable order."""

    with_params: Callable[[TObject, list[torch.Tensor]], TObject]
    """Rebuild an object around new tensors. Must not mutate the original."""

    names: tuple[str, ...] = ()
    """Optional names for the tensors, aligned with ``params_of``'s order."""


@dataclass
class ListCollection(ObjectCollection[TObject]):
    """The default collection: a plain list of objects.

    Correct for any grammar, fast for none -- every object keeps its own
    tensors, so a batch of P proposals hands ``P * k`` tensors to the optimizer
    rather than ``k`` stacked ones. That is fine for small grammars and a real
    cost for large ones; a grammar graduates to a hand-written collection when
    the packed representation actually buys throughput.

    It makes no attempt to stack or trace anything. The upstream
    ``StdCollection`` tried to ``torch.jit.trace`` each object's rasterize and
    got it wrong twice over -- it tested one attribute and assigned another, so
    traces rebuilt on every call, and its closure captured the loop variable.
    Nothing in the upstream tree ever used it.
    """

    objects: list[TObject]
    params: list[list[torch.Tensor]]
    spec: ListSpec[TObject]

    @classmethod
    def build(cls, objects: Sequence[TObject], spec: ListSpec[TObject]) -> ListCollection[TObject]:
        objs = list(objects)
        return cls(objects=objs, params=[list(spec.params_of(o)) for o in objs], spec=spec)

    def __len__(self) -> int:
        return len(self.objects)

    def get(self, idx: int, detach: bool = True) -> TObject:
        obj = self.objects[idx]
        if not detach:
            return obj
        return self.spec.with_params(obj, [t.detach().clone() for t in self.params[idx]])

    def parameters(self) -> list[torch.Tensor]:
        return [t for ts in self.params for t in ts]

    def parameter_names(self) -> list[str]:
        out: list[str] = []
        for i, ts in enumerate(self.params):
            for j, _ in enumerate(ts):
                name = self.spec.names[j] if j < len(self.spec.names) else str(j)
                out.append(f"{i}.{name}")
        return out

    def per_object_grads(self) -> list[torch.Tensor]:
        from ._util import safe_cat

        out: list[torch.Tensor] = []
        for ts in self.params:
            device = ts[0].device if ts else torch.device("cpu")
            out.append(safe_cat([t.grad.reshape(-1) for t in ts if t.grad is not None], (), device=device))
        return out

    def _rebuild(self, params: list[list[torch.Tensor]]) -> Self:
        """Rebuild objects around ``params``, preserving the objects<->params invariant."""
        return type(self)(
            objects=[self.spec.with_params(o, ts) for o, ts in zip(self.objects, params)],
            params=params,
            spec=self.spec,
        )

    def requires_grad_(self, requires_grad: bool = True) -> Self:
        return self._rebuild(
            [[t.detach().clone().requires_grad_(requires_grad) for t in ts] for ts in self.params]
        )

    def clone(self) -> Self:
        return self._rebuild([[t.detach().clone() for t in ts] for ts in self.params])

    def to(self, device: Device = None) -> Self:
        return self._rebuild([[t.to(device=device) for t in ts] for ts in self.params])


def batchify(
    grammar: Grammar[TObject, Any, Any, Any],
    objects: Sequence[TObject],
    *,
    cost_budget: int | None,
    batch_size: int | None = None,
    requires_grad: bool = True,
) -> list[Any]:
    """Partition ``objects`` into collections that each fit one forward/backward.

    ``batch_size`` takes precedence when set; otherwise objects are greedily
    accumulated until ``grammar.object_cost`` would exceed ``cost_budget``.

    The greedy rule matches upstream: flush *before* appending the item that
    would exceed the budget, then flush the tail. One consequence is preserved
    deliberately -- a single object costing more than the whole budget still
    produces a batch that exceeds it, because the alternative is refusing to
    evaluate that object at all.

    One deviation, and it is a bug fix: upstream flushes unconditionally, so an
    oversized *first* object emits an empty batch before its own
    (``Collection.cat([])``). Here the flush is guarded on a non-empty
    accumulator, so no empty batch is ever produced. Partitioning is otherwise
    identical whenever every object fits the budget individually.
    """
    if batch_size is not None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        return [
            grammar.collate(objects[i : i + batch_size]).requires_grad_(requires_grad)
            for i in range(0, len(objects), batch_size)
        ]

    if cost_budget is None:
        raise ValueError("Either batch_size or cost_budget must be set.")

    batches: list[Any] = []
    pending: list[TObject] = []
    pending_cost = 0
    for obj in objects:
        cost = grammar.object_cost(obj)
        if pending and pending_cost + cost > cost_budget:
            batches.append(grammar.collate(pending).requires_grad_(requires_grad))
            pending, pending_cost = [], 0
        pending.append(obj)
        pending_cost += cost
    if pending:
        batches.append(grammar.collate(pending).requires_grad_(requires_grad))
    return batches
