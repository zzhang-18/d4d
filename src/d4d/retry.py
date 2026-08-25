"""Restarting a run that ran out of GPU memory.

Deliberately *not* part of :func:`~d4d.optimize.optimize`. Retrying restarts the
whole run, which is a driver concern rather than a loop concern; it mutates the
arguments, which the loop must not do to its own inputs; and after a CUDA OOM the
grammar may be holding a half-built optimizer and allocated targets, so a correct
retry has to rebuild the grammar -- which the loop cannot do, because it was
handed one rather than told how to make one.

Hence the factories. Both parameters are callables for a reason: ``make_grammar``
so each attempt starts from clean allocations, and ``make_callbacks`` so each
attempt gets its own writers. Upstream re-bound the frame list inside the ``try``,
so a failed attempt's frames were silently discarded.
"""

from __future__ import annotations

import gc
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any

import torch

from .callbacks import Callback
from .grammar import Grammar
from .optimize import OptimizeArgs, OptimizeResult, optimize

__all__ = ["run_with_oom_backoff"]


def run_with_oom_backoff(
    make_grammar: Callable[[], Grammar[Any, Any, Any, Any]],
    args: OptimizeArgs,
    make_callbacks: Callable[[int], Sequence[Callback]] = lambda _: (),
    retries: int = 5,
) -> OptimizeResult[Any]:
    """Run, halving the batch budget and retrying on CUDA OOM.

    Args:
        make_grammar: builds a fresh grammar for each attempt.
        args: starting hyperparameters; never mutated -- each retry gets a copy.
        make_callbacks: builds callbacks for attempt ``i`` (0-based).
        retries: maximum attempts.

    Raises:
        RuntimeError: if every attempt ran out of memory.
    """
    if retries < 1:
        raise ValueError(f"retries must be >= 1, got {retries}")

    cur = replace(args)
    for attempt in range(retries):
        try:
            return optimize(make_grammar(), cur, list(make_callbacks(attempt)))
        except torch.cuda.OutOfMemoryError:
            if attempt == retries - 1:
                break
            if cur.batch_size is not None:
                if cur.batch_size == 1:
                    raise
                cur = replace(cur, batch_size=cur.batch_size // 2)
            else:
                if cur.cost_budget <= 1:
                    raise
                cur = replace(cur, cost_budget=cur.cost_budget // 2)
            gc.collect()
            torch.cuda.empty_cache()
    raise RuntimeError(f"out of memory after {retries} attempts (last: {cur.batch_size=}, {cur.cost_budget=})")
