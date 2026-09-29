"""Tests for ObjectCollection, ListCollection and batching."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import pytest
import torch

from d4d import Grammar, ListCollection, ListSpec, batchify


@dataclass(frozen=True)
class Blob:
    w: torch.Tensor
    b: torch.Tensor


SPEC = ListSpec(
    params_of=lambda o: [o.w, o.b],
    with_params=lambda o, ts: replace(o, w=ts[0], b=ts[1]),
    names=("w", "b"),
)


class BlobGrammar(Grammar[Blob, ListCollection[Blob], Any, None]):
    list_spec = SPEC

    def __init__(self, cost: Any = None) -> None:
        self._cost = cost

    def initial(self) -> Blob:
        return Blob(torch.zeros(3), torch.zeros(1))

    def propose(self, obj: Blob, budget: int) -> list[Any]:
        return []

    def apply(self, obj: Blob, rewrite: Any) -> Blob:
        return obj

    def conflicts(self, a: Any, b: Any) -> bool:
        return True

    def loss(self, batch, ctx, state):
        return torch.stack([(o.w.sum() + o.b.sum()) ** 2 for o in batch.objects]), {}

    def object_cost(self, obj: Blob) -> int:
        return self._cost(obj) if self._cost else super().object_cost(obj)


def blob(n: int = 3) -> Blob:
    return Blob(torch.arange(n, dtype=torch.float32), torch.ones(1))


class TestListCollection:
    def test_objects_and_params_share_tensors_after_releafing(self):
        """The invariant: re-leafing must refresh the objects, or gradients vanish."""
        g = BlobGrammar()
        c = g.collate([blob(), blob()]).requires_grad_()
        for i, obj in enumerate(c.objects):
            assert obj.w is c.params[i][0], "object holds a stale tensor"
            assert obj.b is c.params[i][1]
        assert all(t.requires_grad and t.is_leaf for t in c.parameters())

    def test_gradients_reach_the_optimizer_parameters(self):
        g = BlobGrammar()
        c = g.collate([blob(), blob()]).requires_grad_()
        losses, _ = g.loss(c, None, None)  # type: ignore[arg-type]
        losses.sum().backward()
        assert all(t.grad is not None for t in c.parameters())

    def test_per_object_grads_has_one_entry_per_object(self):
        g = BlobGrammar()
        c = g.collate([blob(), blob(), blob()]).requires_grad_()
        losses, _ = g.loss(c, None, None)  # type: ignore[arg-type]
        losses.sum().backward()
        grads = c.per_object_grads()
        assert len(grads) == 3
        assert all(gr.numel() == 4 for gr in grads)  # w(3) + b(1)

    def test_per_object_grads_is_empty_not_missing_when_no_grad(self):
        g = BlobGrammar()
        c = g.collate([blob(), blob()]).requires_grad_()
        grads = c.per_object_grads()
        assert len(grads) == 2, "objects without grads must still contribute an entry"
        assert all(gr.numel() == 0 for gr in grads)

    def test_get_detaches_by_default(self):
        g = BlobGrammar()
        c = g.collate([blob()]).requires_grad_()
        assert not c.get(0).w.requires_grad
        assert c.get(0, detach=False).w.requires_grad

    def test_clone_is_independent(self):
        g = BlobGrammar()
        c = g.collate([blob()])
        d = c.clone()
        c.params[0][0].add_(100.0)
        assert not torch.allclose(c.params[0][0], d.params[0][0])

    def test_len_and_iteration(self):
        g = BlobGrammar()
        c = g.collate([blob(), blob(), blob()])
        assert len(c) == 3
        assert len(list(c)) == 3

    def test_parameter_names_are_qualified_per_object(self):
        g = BlobGrammar()
        c = g.collate([blob(), blob()])
        assert c.parameter_names() == ["0.w", "0.b", "1.w", "1.b"]


class TestDefaultCollate:
    def test_missing_list_spec_gives_an_actionable_error(self):
        class NoSpec(BlobGrammar):
            list_spec = None

        with pytest.raises(NotImplementedError, match="override collate|list_spec"):
            NoSpec().collate([blob()])


class TestBatchify:
    def test_batch_size_partitions_evenly(self):
        g = BlobGrammar()
        objs = [blob() for _ in range(7)]
        batches = batchify(g, objs, cost_budget=None, batch_size=3)
        assert [len(b) for b in batches] == [3, 3, 1]

    def test_cost_budget_packs_greedily(self):
        g = BlobGrammar(cost=lambda o: 4)
        objs = [blob() for _ in range(5)]
        batches = batchify(g, objs, cost_budget=10)
        assert [len(b) for b in batches] == [2, 2, 1]  # 4+4=8, +4 would be 12 > 10

    def test_default_object_cost_is_parameter_count(self):
        g = BlobGrammar()
        assert g.object_cost(blob(3)) == 4  # w(3) + b(1)

    def test_oversized_object_gets_its_own_batch_not_an_empty_one(self):
        """Upstream flushed unconditionally and emitted an empty leading batch."""
        g = BlobGrammar(cost=lambda o: 100)
        batches = batchify(g, [blob(), blob()], cost_budget=10)
        assert [len(b) for b in batches] == [1, 1]
        assert all(len(b) > 0 for b in batches)

    def test_every_object_survives_partitioning(self):
        g = BlobGrammar(cost=lambda o: int(o.w.numel()))
        objs = [blob(n) for n in (1, 5, 2, 9, 3, 1)]
        batches = batchify(g, objs, cost_budget=6)
        assert sum(len(b) for b in batches) == len(objs)

    def test_batches_are_leaves_ready_for_the_optimizer(self):
        g = BlobGrammar()
        batches = batchify(g, [blob(), blob()], cost_budget=None, batch_size=1)
        for b in batches:
            assert all(t.requires_grad and t.is_leaf for t in b.parameters())

    def test_requires_either_budget_or_batch_size(self):
        g = BlobGrammar()
        with pytest.raises(ValueError, match="batch_size or cost_budget"):
            batchify(g, [blob()], cost_budget=None)

    def test_rejects_nonsense_batch_size(self):
        g = BlobGrammar()
        with pytest.raises(ValueError, match="batch_size must be"):
            batchify(g, [blob()], cost_budget=None, batch_size=0)

    def test_empty_input_yields_no_batches(self):
        g = BlobGrammar()
        assert batchify(g, [], cost_budget=10) == []
