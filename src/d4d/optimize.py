"""The optimization loop.

Alternates gradient descent on continuous parameters with discrete rewrites.
Each step:

1. decide whether this is a rewrite step;
2. every ``cleanup_every`` steps, ``cleanup`` and rebuild the optimizer;
3. on a rewrite step, check early stopping, ``propose``, score the candidates and ``combine``;
4. take a descent step;
5. update the moving average, the LR scheduler and the grammar's state.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Generic, Literal

import torch
from torch.optim.lr_scheduler import ExponentialLR, LinearLR, LRScheduler, ReduceLROnPlateau
from torch.optim.optimizer import Optimizer
from typing_extensions import TypeVar

from ._util import MovingAverage, maybe_clamp
from .callbacks import (
    Callback,
    CallbackList,
    RewriteEvent,
    RunEnd,
    RunStart,
    StepEnd,
    StopRun,
    VisualizeEvent,
)
from .collection import ObjectCollection, batchify
from .grammar import ExtraMetrics, Grammar, StepContext
from .scheduler import AdaptiveLRScheduler

__all__ = ["OptimizeArgs", "OptimizeResult", "optimize"]

TObject = TypeVar("TObject")
TCollection = TypeVar("TCollection", bound=ObjectCollection[Any])
TRewrite = TypeVar("TRewrite")
TState = TypeVar("TState")

MetricSeries = dict[str, tuple[float, ...]]


@dataclass
class OptimizeArgs:
    """Hyperparameters for :func:`optimize`."""

    n_steps: int = 4000

    # GD-related
    optimizer: Literal["Adam", "SGD"] = "Adam"
    scheduler: Literal["none", "ReduceLROnPlateau", "AdaptiveLR", "LinearLR", "ExponentialLR"] = "none"
    lr: float = 0.5
    clip_grad: float | None = 2.0
    """Clip gradient values to [-clip_grad, clip_grad]. 'None' disables."""
    clip_grad_mode: Literal["abs", "rel"] = "abs"
    """'abs' clips at 'clip_grad'; 'rel' clips at 'clip_grad / lr'."""
    reduce_lr_factor: float = 0.5
    reduce_lr_patience: int = 2
    reduce_lr_min_lr: float = 1e-4
    increase_lr_patience: int = 2
    reset_lr_after_proposal: bool = False
    """With ReduceLROnPlateau, reset the LR to 'lr' after an accepted rewrite."""
    increase_lr_after_proposal: bool = True
    """With ReduceLROnPlateau, divide the LR by 'reduce_lr_factor' (capped at 'lr') after an accepted rewrite."""

    # Clean up
    cleanup_every: int = 10

    # Proposal (rewrites)
    proposal_trigger: Literal["step", "rel_loss"] = "step"
    """'step' rewrites every 'propose_every' steps. 'rel_loss' rewrites after 'proposal_patience'
    consecutive steps whose moving-average loss fell by less than a 'proposal_rel_loss' fraction."""
    propose_every: int = 50
    proposal_rel_loss: float = 5e-3
    proposal_patience: int = 10
    proposal_criterion: Literal["loss", "grad", "grad_only"] = "loss"
    """How candidates are scored. 'loss' takes 'proposal_steps' descent steps, then evaluates
    the loss. 'grad' uses 'loss - lr * <clip(g), g>' from one backward pass. 'grad_only'
    drops the loss term."""
    proposal_steps: int = 2
    proposal_size: int = 0
    """Candidate budget passed to 'Grammar.propose'. '0' means all."""
    proposal_clip_grad: bool = True
    """Clip the gradient at 'clip_grad' inside the 'grad' criteria."""
    accept_top_k: int = 0
    """Maximum rewrites accepted per rewrite step. '0' means unlimited."""
    accept_abs_eps: float | None = None
    """Floor: accept only if 'base_loss - loss > accept_abs_eps'. 'None' disables."""
    accept_rel_eps: float | None = None
    """Floor: accept only if 'base_loss - loss > accept_rel_eps * |base_loss|'. 'None' disables."""
    accept_eps_op: Literal["and", "or"] = "or"
    """Whether both floors or either must be cleared. With both disabled, the test is 'base_loss - loss > 0'."""

    # Batching
    cost_budget: int = 8192
    """Budget per batch, in units of 'Grammar.object_cost'. Ignored when 'batch_size' is set."""
    batch_size: int | None = None
    """Candidates per batch."""

    # Simplicity
    w_simplicity: float = 1.0
    """Weight on 'Grammar.simplicity' in proposal ranking and '$loss'."""

    # Visualization
    visualize_every: int = 10
    """Call 'Grammar.visualize' every this many steps and on rewrite steps. '0' disables."""

    # Early stopping
    stopping_eps: float = 5e-3
    """A rewrite event stalls unless the moving-average loss is at most '(1 - stopping_eps)' times
    its lowest value at earlier rewrite events."""
    stopping_patience: int | None = None
    """Stop after this many consecutive stalled rewrite events. 'None' disables early stopping."""

    seed: int | None = None

    def __post_init__(self) -> None:
        if self.n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {self.n_steps}")
        if self.propose_every < 1:
            raise ValueError(f"propose_every must be >= 1, got {self.propose_every}")
        if self.cleanup_every < 1:
            raise ValueError(f"cleanup_every must be >= 1, got {self.cleanup_every}")
        if self.accept_top_k < 0:
            raise ValueError(f"accept_top_k must be >= 0, got {self.accept_top_k}")
        if self.accept_eps_op not in ("and", "or"):
            raise ValueError(f"accept_eps_op must be 'and' or 'or', got {self.accept_eps_op!r}")
        if self.proposal_steps < 0:
            raise ValueError(f"proposal_steps must be >= 0, got {self.proposal_steps}")


@dataclass
class OptimizeResult(Generic[TObject]):
    best: TObject
    best_loss: float
    best_step: int
    final: TObject
    metrics: MetricSeries
    """Per-step series; reserved keys are '$'-prefixed. Each entry is measured after that
    step's rewrite, if any, and before its parameter update."""
    stopped_early: bool
    n_steps_run: int
    n_rewrites: int


def _rank_improving(args: OptimizeArgs, base_loss: float, losses: Sequence[float]) -> list[int]:
    """Indices of the proposals that clear the acceptance floors, by ascending loss; ties keep index order."""

    def improves(loss: float) -> bool:
        imp = base_loss - loss
        tests: list[bool] = []
        if args.accept_abs_eps is not None:
            tests.append(imp > args.accept_abs_eps)
        if args.accept_rel_eps is not None:
            tests.append(imp > args.accept_rel_eps * abs(base_loss))
        if not tests:
            return imp > 0
        return any(tests) if args.accept_eps_op == "or" else all(tests)

    return sorted((i for i, loss in enumerate(losses) if improves(loss)), key=lambda i: losses[i])


def _merge_extra(acc: dict[str, list[float]], new: ExtraMetrics, n: int) -> None:
    """Accumulate per-step extras, averaging across objects in the batch."""
    for key, values in new.items():
        seq = list(values)
        if len(seq) != n:
            raise ValueError(f"extra metric {key!r} has length {len(seq)}, expected {n}")
        acc.setdefault(key, []).append(sum(seq) / len(seq) if seq else 0.0)


def optimize(
    grammar: Grammar[TObject, TCollection, TRewrite, TState],
    args: OptimizeArgs,
    callbacks: Callback | Sequence[Callback] | None = None,
) -> OptimizeResult[TObject]:
    """Run grammar-guided hybrid optimization.

    Args:
        grammar: defines the search space, the objective and the rewrite rules.
        args: algorithm hyperparameters.
        callbacks: side effects -- progress, image/video/metric writing,
            checkpointing. The loop itself writes nothing to disk.

    Returns:
        :class:`OptimizeResult` with the best and final objects and the metric series.
    """
    cbs = CallbackList(callbacks)
    if args.seed is not None:
        # TODO: do we need other seeds here?
        torch.manual_seed(args.seed)

    def build_optimizer(
        parameters: list[torch.Tensor], cur_step: int, lr: float
    ) -> tuple[Optimizer, LRScheduler | None]:
        """A fresh optimizer and scheduler over ``parameters`` (leaf tensors, any shape)."""
        if args.optimizer == "Adam":
            opt: Optimizer = torch.optim.Adam(parameters, lr=lr)
        elif args.optimizer == "SGD":
            opt = torch.optim.SGD(parameters, lr=lr)
        else:  # TODO: other optimizers?
            raise ValueError(f"unknown optimizer {args.optimizer}")

        sched: LRScheduler | None = None
        if args.scheduler == "ReduceLROnPlateau":
            sched = ReduceLROnPlateau(  # type: ignore[assignment]
                opt, factor=args.reduce_lr_factor,
                patience=args.reduce_lr_patience, min_lr=args.reduce_lr_min_lr,
            )
        elif args.scheduler == "AdaptiveLR":
            sched = AdaptiveLRScheduler(
                opt, factor=args.reduce_lr_factor,
                reduce_patience=args.reduce_lr_patience,
                increase_patience=args.increase_lr_patience,
                min_lr=args.reduce_lr_min_lr, max_lr=args.lr,
            )
        elif args.scheduler == "LinearLR":
            sched = LinearLR(opt, start_factor=1.0, end_factor=0.0, total_iters=max(args.n_steps - cur_step, 1))
        elif args.scheduler == "ExponentialLR":
            sched = ExponentialLR(opt, gamma=0.01 ** (1 / args.n_steps))
        return opt, sched

    def clip_value(lr: float) -> float | None:
        if args.clip_grad is None:
            return None
        return args.clip_grad if args.clip_grad_mode == "abs" else args.clip_grad / max(lr, 1e-12)

    # Setup
    initial_obj = grammar.initial()
    population: TCollection = grammar.collate([initial_obj]).requires_grad_()
    state: TState = grammar.init_state()

    opt, sched = build_optimizer(population.parameters(), cur_step=0, lr=args.lr)
    cur_lr = float(sched.get_last_lr()[0]) if sched is not None else args.lr

    ma_loss = MovingAverage(window_size=8)
    loss_since_last_rewrite = float("inf")
    stopping_patience = 0
    patience = 0

    series: dict[str, list[float]] = {
        "$loss": [], "$loss_ma": [], "$loss_cont": [],
        "$loss_simp": [], "$lr": [], "$timestamp": [],
    }
    extra_acc: dict[str, list[float]] = {}

    best_obj: TObject = initial_obj
    best_loss = float("inf")
    best_step = 0
    n_rewrites = 0
    stopped_early = False
    step = 0

    result: OptimizeResult[TObject] | None = None
    error: BaseException | None = None

    cbs.on_run_start(RunStart(grammar=grammar, args=args, initial=initial_obj))
    # TODO: is the double try here a good pattern?
    try:
        try:
            for step in range(args.n_steps):
                # -- 1. is this a rewrite step? ----------------------------
                if step == 0:
                    rewrite = False
                elif args.proposal_trigger == "step":
                    rewrite = step % args.propose_every == 0
                elif args.proposal_trigger == "rel_loss":
                    rewrite = patience >= args.proposal_patience
                    if rewrite:
                        patience = 0
                else:
                    raise ValueError(f"Unknown proposal_trigger: {args.proposal_trigger}")

                # -- 2. periodic cleanup -----------------------------------
                if step % args.cleanup_every == 0:
                    population = grammar.cleanup(population).requires_grad_()
                    if not rewrite:
                        # rewrite steps rebuild it in step 3
                        opt, sched = build_optimizer(population.parameters(), step, cur_lr)

                # -- 3. rewrite --------------------------------------------
                if rewrite:
                    # stall check: moving-average loss vs. its lowest value at earlier rewrite events
                    cur_ma = series["$loss_ma"][-1]
                    if cur_ma > loss_since_last_rewrite * (1 - args.stopping_eps):
                        stopping_patience += 1
                    else:
                        stopping_patience = 0
                    loss_since_last_rewrite = min(loss_since_last_rewrite, cur_ma)
                    ma_loss.clear()

                    if args.stopping_patience is not None and stopping_patience >= args.stopping_patience:
                        stopped_early = True
                        break

                    base_obj = population.get(0)
                    rewrites = grammar.propose(base_obj, args.proposal_size)
                    if rewrites:
                        n_rewrites += 1
                        new_obj, changed = _run_rewrite(
                            grammar=grammar, args=args, base_obj=base_obj, rewrites=rewrites,
                            state=state, step=step, cur_lr=cur_lr, rewrite_index=n_rewrites,
                            build_optimizer=build_optimizer, clip_value=clip_value, cbs=cbs,
                        )
                        if changed:
                            population = grammar.collate([new_obj]).requires_grad_()
                            if args.scheduler == "ReduceLROnPlateau":
                                if args.increase_lr_after_proposal:
                                    cur_lr = min(args.lr, cur_lr / args.reduce_lr_factor)
                                if args.reset_lr_after_proposal:
                                    cur_lr = args.lr
                    opt, sched = build_optimizer(population.parameters(), step, cur_lr)

                # -- 4. continuous step ------------------------------------
                ctx = StepContext(
                    step=step, total_steps=args.n_steps, phase="step", compute_extra=True,
                    lr=cur_lr, rewrite=rewrite, rewrite_index=n_rewrites, n_objects=len(population),
                )
                simplicity = list(grammar.simplicity(population, ctx))
                losses, extra = grammar.loss(population, ctx, state)
                loss = losses.sum()

                with_simp = [
                    l + s * args.w_simplicity for l, s in zip(losses.detach().tolist(), simplicity)
                ]
                total = sum(with_simp)
                mean_loss = total / len(with_simp) if with_simp else 0.0

                series["$loss"].append(total)
                series["$loss_cont"].append(float(loss.item()))
                series["$loss_simp"].append(sum(simplicity))
                series["$lr"].append(cur_lr)
                series["$timestamp"].append(grammar.elapsed())
                _merge_extra(extra_acc, extra, len(population))

                if total < best_loss:
                    best_loss, best_step = total, step
                    best_obj = population.get(0)

                if args.visualize_every > 0 and (step % args.visualize_every == 0 or rewrite):
                    vctx = StepContext(
                        step=step, total_steps=args.n_steps, phase="visualize", lr=cur_lr,
                        rewrite=rewrite, rewrite_index=n_rewrites, n_objects=len(population),
                    )
                    image = grammar.visualize(population, vctx, state)
                    if image is not None:
                        cbs.on_visualize(VisualizeEvent(step=step, loss=float(loss.item()), image=image))

                opt.zero_grad()
                loss.backward()
                population.scale_grads_()
                cv = clip_value(cur_lr)
                if cv is not None:
                    torch.nn.utils.clip_grad_value_(population.parameters(), cv)
                opt.step()
                population.project_to_valid_()

                # -- 5. bookkeeping ----------------------------------------
                ma_loss.add(mean_loss)
                series["$loss_ma"].append(ma_loss.mean())

                if sched is not None:
                    if isinstance(sched, (ReduceLROnPlateau, AdaptiveLRScheduler)):
                        # plateau schedulers take the detached loss
                        sched.step(loss.detach())
                    else:
                        sched.step()
                    cur_lr = float(sched.get_last_lr()[0])

                state = grammar.step_state(state)

                ma = series["$loss_ma"]
                if len(ma) >= 2:
                    patience = patience + 1 if (1 - args.proposal_rel_loss) * ma[-2] <= ma[-1] else 0

                cbs.on_step_end(
                    StepEnd(
                        step=step, total_steps=args.n_steps, loss=total,
                        loss_cont=series["$loss_cont"][-1],
                        loss_simplicity=series["$loss_simp"][-1], loss_ma=ma[-1],
                        lr=cur_lr, rewrite=rewrite, elapsed=series["$timestamp"][-1],
                        extra={k: v[-1] for k, v in extra_acc.items()},
                        # evaluated only if a callback calls get_object()
                        _get_object=lambda p=population: p.get(0),
                    )
                )
        except StopRun as stop:
            # a callback raised StopRun: stop early and still return a result
            warnings.warn(f"run stopped at step {step}: {stop}", stacklevel=2)
            stopped_early = True

        metrics: MetricSeries = {k: tuple(v) for k, v in series.items()}
        metrics.update({k: tuple(v) for k, v in extra_acc.items()})
        result = OptimizeResult(
            best=best_obj, best_loss=best_loss, best_step=best_step,
            final=population.get(0), metrics=metrics,
            stopped_early=stopped_early, n_steps_run=step + 1, n_rewrites=n_rewrites,
        )
        return result
    except BaseException as exc:
        error = exc
        raise
    finally:
        # runs on success, error and interruption
        cbs.on_run_end(RunEnd(result=result, error=error))


def _run_rewrite(
    *,
    grammar: Grammar[Any, Any, Any, Any],
    args: OptimizeArgs,
    base_obj: Any,
    rewrites: list[Any],
    state: Any,
    step: int,
    cur_lr: float,
    rewrite_index: int,
    build_optimizer: Callable[..., tuple[Optimizer, LRScheduler | None]],
    clip_value: Callable[[float], float | None],
    cbs: CallbackList,
) -> tuple[Any, bool]:
    """Score candidate rewrites and combine the winners.

    Returns ``(new_object, changed)``.
    """
    # the base goes last, so its score is scored[-1]
    candidates = [grammar.apply(base_obj, r) for r in rewrites] + [base_obj]
    batches = batchify(grammar, candidates, cost_budget=args.cost_budget, batch_size=args.batch_size)

    # one proposal state per rewrite event, shared by every batch
    prop_state = grammar.state_for_proposals(state)

    def make_ctx(inner: int, n: int) -> StepContext:
        return StepContext(
            step=step, total_steps=args.n_steps, phase="proposal", compute_extra=False,
            lr=cur_lr, rewrite=True, rewrite_index=rewrite_index, proposal_step=inner, n_objects=n,
        )

    raw_losses: list[float] = []
    grad_terms: list[float] = []

    if args.proposal_criterion == "loss":
        for batch in batches:
            opt, _ = build_optimizer(batch.parameters(), step, cur_lr)
            for inner in range(args.proposal_steps):
                opt.zero_grad()
                losses, _ = grammar.loss(batch, make_ctx(inner, len(batch)), prop_state)
                losses.sum().backward()
                batch.scale_grads_()
                cv = clip_value(cur_lr)
                if cv is not None:
                    torch.nn.utils.clip_grad_value_(batch.parameters(), cv)
                opt.step()
                batch.project_to_valid_()
        with torch.no_grad():
            for batch in batches:
                losses, _ = grammar.loss(batch, make_ctx(args.proposal_steps, len(batch)), prop_state)
                raw_losses.extend(losses.tolist())

    elif args.proposal_criterion in ("grad", "grad_only"):
        for batch in batches:
            for p in batch.parameters():
                p.grad = None
            losses, _ = grammar.loss(batch, make_ctx(0, len(batch)), prop_state)
            raw_losses.extend(losses.tolist())
            losses.sum().backward()

            grads = batch.per_object_grads()
            scaled = batch.per_object_grads() if batch.scale_grads_() else grads
            lo = -args.clip_grad if (args.clip_grad is not None and args.proposal_clip_grad) else None
            hi = args.clip_grad if args.proposal_clip_grad else None
            # predicted first-order decrease from one step: <clip(scaled_g), g>
            grad_terms.extend(
                float((maybe_clamp(s, min=lo, max=hi) * g).sum().item()) for g, s in zip(grads, scaled)
            )
    else:
        raise ValueError(f"Unknown proposal_criterion: {args.proposal_criterion}")

    simplicity: list[float] = []
    for batch in batches:
        simplicity.extend(grammar.simplicity(batch, make_ctx(0, len(batch))))

    if args.proposal_criterion == "loss":
        scored = [l + s * args.w_simplicity for l, s in zip(raw_losses, simplicity)]
    else:
        loss_w = 0.0 if args.proposal_criterion == "grad_only" else 1.0
        scored = [
            loss_w * l - cur_lr * g + s * args.w_simplicity
            for l, g, s in zip(raw_losses, grad_terms, simplicity)
        ]

    cand_losses, base_loss = scored[:-1], scored[-1]
    ranked = _rank_improving(args, base_loss, cand_losses)
    new_obj, accepted = base_obj, []
    if ranked:
        new_obj, accepted = grammar.combine(
            base_obj, [rewrites[i] for i in ranked], [base_loss - cand_losses[i] for i in ranked],
            top_k=args.accept_top_k,
        )
    cbs.on_rewrite(
        RewriteEvent(
            step=step, rewrite_index=rewrite_index, base_loss=base_loss,
            losses=tuple(cand_losses), rewrites=tuple(rewrites), accepted=tuple(accepted),
            changed=bool(accepted), n_proposals=len(rewrites), elapsed=grammar.elapsed(),
        )
    )
    return new_obj, bool(accepted)
