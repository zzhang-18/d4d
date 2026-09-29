"""Batches of objects whose parameters the optimizer descends on.

An ``ObjectCollection`` exposes a batch's parameters as a flat list of leaf
tensors, so one ``torch.optim`` instance steps every object in it. Collections
are built by ``Grammar.collate``; :func:`batchify` partitions objects into them.
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
    """A batch of objects whose parameters are leaf tensors.

    Storage is up to the implementation; the methods below are the contract.
    ``get(i)`` must return an object that shares (or is rebuilt from) the
    tensors ``parameters()`` exposes, so a subclass that re-leafs tensors in
    ``requires_grad_`` must also rebuild its objects.
    """

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def get(self, idx: int, detach: bool = True) -> TObject:
        """Extract a single object. ``detach=True`` severs it from the graph."""

    @abstractmethod
    def parameters(self) -> list[torch.Tensor]:
        """Leaf tensors (``requires_grad`` set, any shape) handed to the torch optimizer."""

    @abstractmethod
    def parameter_names(self) -> list[str]:
        """Human-readable names, positionally aligned with ``parameters()``."""

    @abstractmethod
    def per_object_grads(self) -> list[torch.Tensor]:
        """``len(self)`` flat gradients; entry ``i`` is ``(D_i,)``, object ``i``'s ``.grad`` concatenated.

        An object with no ``.grad`` gives an empty ``(0,)`` tensor. Used by
        ``proposal_criterion in ("grad", "grad_only")``.
        """

    @abstractmethod
    def requires_grad_(self, requires_grad: bool = True) -> Self:
        """Re-leaf every parameter. Called after every rewrite and cleanup."""

    @abstractmethod
    def clone(self) -> Self: ...

    @abstractmethod
    def to(self, device: Device = None) -> Self: ...

    def scale_grads_(self) -> bool:
        """Rescale gradients in place; return True if any changed.

        The grad proposal criteria recompute ``per_object_grads`` only when this returns True.
        """
        return False

    def project_to_valid_(self) -> Self:
        """Project parameters into the feasible set; called after every optimizer step. Default: no-op."""
        return self

    # -- Sequence -----------------------------------------------------------
    def __getitem__(self, idx: int) -> TObject:  # type: ignore[override]
        return self.get(idx, True)

    def __iter__(self) -> Iterator[TObject]:
        for i in range(len(self)):
            yield self.get(i, True)


@dataclass(frozen=True)
class ListSpec(Generic[TObject]):
    """How the default ``Grammar.collate`` takes an object's tensors out and puts new ones back."""

    params_of: Callable[[TObject], list[torch.Tensor]]
    """An object's tensors (any shapes), in a stable order."""

    with_params: Callable[[TObject, list[torch.Tensor]], TObject]
    """Rebuild an object around new tensors of the same shapes. Must not mutate the original."""

    names: tuple[str, ...] = ()
    """Optional names for the tensors, aligned with ``params_of``'s order."""


@dataclass
class ListCollection(ObjectCollection[TObject]):
    """The default collection: a list of objects, each keeping its own tensors."""

    objects: list[TObject]
    params: list[list[torch.Tensor]]
    """``params[i]`` is object ``i``'s tensors (any shapes), in ``spec.params_of`` order."""
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
        """Rebuild the objects around ``params``."""
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
    """Partition ``objects``, in order, into collections.

    With ``batch_size``, fixed-size chunks. Otherwise a batch is closed before
    the object that would push its total ``grammar.object_cost`` over
    ``cost_budget``; an object costing more than the budget gets a batch of its own.
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
