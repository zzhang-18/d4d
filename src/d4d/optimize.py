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

import math
import warnings
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass
from typing import Any, Generic, Literal

import torch
from torch.optim.lr_scheduler import ExponentialLR, LambdaLR, LinearLR, LRScheduler, ReduceLROnPlateau
from torch.optim.optimizer import Optimizer
from typing_extensions import TypeVar

from ._util import MovingAverage, maybe_clamp, seed_everything
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
    scheduler: Literal["none", "ReduceLROnPlateau", "AdaptiveLR", "LinearLR", "ExponentialLR", "CosineSegment"] = "none"
    """'CosineSegment' restarts the LR at 'lr' at every optimizer rebuild of the main loop (cleanup and rewrite steps)
    and decays it by a cosine to 'lr * segment_end_factor' over the 'min(propose_every, cleanup_every)' steps up to the
    next rebuild (warm restarts per round; held at the end value if no rebuild comes, e.g. with keep_optimizer_state).
    Proposal scoring then starts from 'lr' too, not from the decayed end-of-round value."""
    segment_end_factor: float = 0.1
    """'CosineSegment': the LR at the end of a segment, as a fraction of 'lr'."""
    lr: float = 0.5
    clip_grad: float | None = 2.0
    """Clip gradient values to [-clip_grad, clip_grad]. 'None' disables."""
    clip_grad_mode: Literal["abs", "rel"] = "abs"
    """'abs' clips at 'clip_grad'; 'rel' clips at 'clip_grad / lr'."""
    keep_optimizer_state: bool = False
    """Keep the optimizer across rewrite and cleanup events instead of rebuilding it. An event that changes nothing is
    a no-op for the optimizer. After an accepted rewrite (or a cleanup that changed the population), per-row state is
    carried through 'Grammar.param_correspondence'. Proposal scoring starts from the same state. 'False' rebuilds a
    fresh optimizer at every event (the d4d 0.1.0 behavior)."""
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
    proposal_lr_mult: float = 1.0
    """The scoring optimizer's LR (all 'loss'-criterion scoring passes) as a multiple of the main loop's current LR."""
    proposal_eval: Literal["loss", "grammar"] = "loss"
    """What a candidate's score is after its scoring steps ('loss' criterion only): 'loss' evaluates 'Grammar.loss';
    'grammar' asks 'Grammar.proposal_score' (e.g. a hard metric of the exported object; lower is better). The
    simplicity term is added either way."""
    proposal_size: int = 64
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
    proposal_stage2_steps: int = 0
    """Two-stage scoring ('loss' criterion only). '> 0': the 'proposal_stage2_top' best candidates of each
    'Grammar.rewrite_group' by the first-stage score (improving or not), and the base, are re-scored by a fresh
    scoring pass of this many steps; only they can be accepted, ranked by that score. '0' disables."""
    proposal_stage2_top: int = 8
    """Candidates per group that enter the second stage."""
    accept_verify: bool = False
    """Verified combination ('loss' criterion only). When 'Grammar.combine' accepts two or more rewrites, the prefixes
    of its accepted list (best first) and the base are re-scored by a fresh scoring pass (the second stage's steps
    when two-stage scoring is on); the best prefix is kept, or nothing if no prefix beats the base."""

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
    """Passed to 'seed_everything' at the start of the run. 'None' leaves the RNGs as they are."""

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
        if self.proposal_stage2_steps < 0:
            raise ValueError(f"proposal_stage2_steps must be >= 0, got {self.proposal_stage2_steps}")
        if self.proposal_stage2_top < 1:
            raise ValueError(f"proposal_stage2_top must be >= 1, got {self.proposal_stage2_top}")
        if (self.proposal_stage2_steps > 0 or self.accept_verify) and self.proposal_criterion != "loss":
            raise ValueError("proposal_stage2_steps and accept_verify need proposal_criterion='loss'")
        if self.proposal_lr_mult <= 0:
            raise ValueError(f"proposal_lr_mult must be > 0, got {self.proposal_lr_mult}")
        if self.proposal_eval not in ("loss", "grammar"):
            raise ValueError(f"proposal_eval must be 'loss' or 'grammar', got {self.proposal_eval!r}")
        if self.proposal_eval == "grammar" and self.proposal_criterion != "loss":
            raise ValueError("proposal_eval='grammar' needs proposal_criterion='loss'")
        if not 0.0 <= self.segment_end_factor <= 1.0:
            raise ValueError(f"segment_end_factor must be in [0, 1], got {self.segment_end_factor}")


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


def _identity_maps(params: Sequence[torch.Tensor]) -> list[torch.Tensor | None]:
    return [torch.arange(p.shape[0], device=p.device) if p.dim() > 0 else None for p in params]


def _transfer_state(
    src_opt: Optimizer,
    src_params: Sequence[torch.Tensor],
    dst_opt: Optimizer,
    dst_params: Sequence[torch.Tensor],
    maps: Sequence[torch.Tensor | None] | None,
) -> None:
    """Seed ``dst_opt``'s state for ``dst_params`` from ``src_opt``'s state for ``src_params``, row by row.

    ``maps[i][r]`` is the row of ``src_params[i]`` that row ``r`` of ``dst_params[i]`` continues (-1 = a new row).
    Per-element state (e.g. Adam's ``exp_avg`` / ``exp_avg_sq``, SGD's ``momentum_buffer``) is gathered by row.
    Scalars (e.g. ``step``) are copied. New rows get zero first moments, and second moments equal to the mean over the
    carried rows. A zero second moment under an inherited step count would make their first Adam step ~3x ``lr``.
    Tensors with no map, no state, or no carried rows start fresh.
    """
    if maps is None:
        return
    for p_src, p_dst, m in zip(src_params, dst_params, maps):
        st = src_opt.state.get(p_src)
        if m is None or not st or p_dst.dim() == 0 or p_dst.shape[1:] != p_src.shape[1:]:
            continue
        m = m.to(p_dst.device)
        keep = m >= 0
        if not bool(keep.any()):
            continue
        new: dict[str, Any] = {}
        for k, v in st.items():
            if torch.is_tensor(v) and v.dim() > 0 and v.shape == p_src.shape:
                out = torch.zeros_like(p_dst)
                carried = v[m[keep]]
                out[keep] = carried
                if k in ("exp_avg_sq", "max_exp_avg_sq") and not bool(keep.all()):
                    out[~keep] = carried.mean(0, keepdim=True).expand_as(out[~keep])
                new[k] = out
            elif torch.is_tensor(v):
                new[k] = v.clone()
            else:
                new[k] = v
        dst_opt.state[p_dst] = new


def _carry(
    grammar: Grammar[Any, Any, Any, Any],
    src_opt: Optimizer,
    src_params: Sequence[torch.Tensor],
    old_obj: Any,
    dst_opt: Optimizer,
    dst_params: Sequence[torch.Tensor],
    new_obj: Any,
) -> None:
    """``_transfer_state`` through ``grammar.param_correspondence(old_obj, new_obj)`` (identity when they are the same object)."""
    maps = _identity_maps(dst_params) if new_obj is old_obj else grammar.param_correspondence(old_obj, new_obj)
    _transfer_state(src_opt, src_params, dst_opt, dst_params, maps)


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
        seed_everything(args.seed)

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
        elif args.scheduler == "CosineSegment":
            seg_len, end = max(min(args.propose_every, args.cleanup_every), 1), args.segment_end_factor
            sched = LambdaLR(opt, lambda t: end + (1.0 - end) * 0.5 * (1.0 + math.cos(math.pi * min(t, seg_len) / seg_len)))
        return opt, sched

    def restart_lr() -> float:
        """The LR a main-loop rebuild starts from: 'lr' with warm restarts ('CosineSegment'), else the current one."""
        return args.lr if args.scheduler == "CosineSegment" else cur_lr

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
                    cleaned = grammar.cleanup(population)
                    if not args.keep_optimizer_state:
                        population = cleaned.requires_grad_()
                        if not rewrite:
                            # rewrite steps rebuild it in step 3
                            opt, sched = build_optimizer(population.parameters(), step, restart_lr())
                    elif cleaned is not population:  # changed: carry state row by row; unchanged: keep everything
                        cleaned = cleaned.requires_grad_()
                        new_opt, new_sched = build_optimizer(cleaned.parameters(), step, restart_lr())
                        for i in range(len(cleaned)):
                            _carry(grammar, opt, population.object_parameters(i), population.get(i, detach=False),
                                   new_opt, cleaned.object_parameters(i), cleaned.get(i, detach=False))
                        population, opt, sched = cleaned, new_opt, new_sched

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
                            state=state, step=step, cur_lr=restart_lr(), rewrite_index=n_rewrites,
                            build_optimizer=build_optimizer, clip_value=clip_value, cbs=cbs,
                            live_opt=opt if args.keep_optimizer_state else None,
                            live_params=population.object_parameters(0) if args.keep_optimizer_state else None,
                        )
                        if changed:
                            new_pop = grammar.collate([new_obj]).requires_grad_()
                            if args.keep_optimizer_state:
                                new_opt, new_sched = build_optimizer(new_pop.parameters(), step, restart_lr())
                                _carry(grammar, opt, population.object_parameters(0), base_obj,
                                       new_opt, new_pop.object_parameters(0), new_obj)
                                opt, sched = new_opt, new_sched
                            population = new_pop
                            if args.scheduler == "ReduceLROnPlateau":
                                if args.increase_lr_after_proposal:
                                    cur_lr = min(args.lr, cur_lr / args.reduce_lr_factor)
                                if args.reset_lr_after_proposal:
                                    cur_lr = args.lr
                    if not args.keep_optimizer_state:
                        opt, sched = build_optimizer(population.parameters(), step, restart_lr())

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
    live_opt: Optimizer | None = None,
    live_params: Sequence[torch.Tensor] | None = None,
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

    def descend_and_eval(objs: list[Any], obj_batches: list[Any], n_steps: int) -> list[float]:
        """'loss' criterion: ``n_steps`` of a fresh scoring optimizer per batch, then the raw losses."""
        out: list[float] = []
        offset = 0
        for batch in obj_batches:
            opt, _ = build_optimizer(batch.parameters(), step, cur_lr * args.proposal_lr_mult)
            if live_opt is not None and live_params is not None:  # keep_optimizer_state: start from the live state
                for j in range(len(batch)):
                    _carry(grammar, live_opt, live_params, base_obj, opt, batch.object_parameters(j), objs[offset + j])
            offset += len(batch)
            for inner in range(n_steps):
                opt.zero_grad()
                losses, _ = grammar.loss(batch, make_ctx(inner, len(batch)), prop_state)
                losses.sum().backward()
                batch.scale_grads_()
                cv = clip_value(cur_lr * args.proposal_lr_mult)
                if cv is not None:
                    torch.nn.utils.clip_grad_value_(batch.parameters(), cv)
                opt.step()
                batch.project_to_valid_()
        with torch.no_grad():
            for batch in obj_batches:
                if args.proposal_eval == "grammar":
                    out.extend(float(v) for v in grammar.proposal_score(batch, make_ctx(n_steps, len(batch)), prop_state))
                else:
                    losses, _ = grammar.loss(batch, make_ctx(n_steps, len(batch)), prop_state)
                    out.extend(losses.tolist())
        return out

    def simplicity_of(obj_batches: list[Any]) -> list[float]:
        out: list[float] = []
        for batch in obj_batches:
            out.extend(grammar.simplicity(batch, make_ctx(0, len(batch))))
        return out

    def rescore(objs: list[Any], n_steps: int) -> list[float]:
        """A fresh 'loss' scoring pass over ``objs``: loss + w_simplicity * simplicity."""
        obj_batches = batchify(grammar, objs, cost_budget=args.cost_budget, batch_size=args.batch_size)
        raw = descend_and_eval(objs, obj_batches, n_steps)
        return [l + s * args.w_simplicity for l, s in zip(raw, simplicity_of(obj_batches))]

    if args.proposal_criterion == "loss":
        raw_losses.extend(descend_and_eval(candidates, batches, args.proposal_steps))

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

    simplicity = simplicity_of(batches)

    if args.proposal_criterion == "loss":
        scored = [l + s * args.w_simplicity for l, s in zip(raw_losses, simplicity)]
    else:
        loss_w = 0.0 if args.proposal_criterion == "grad_only" else 1.0
        scored = [
            loss_w * l - cur_lr * g + s * args.w_simplicity
            for l, g, s in zip(raw_losses, grad_terms, simplicity)
        ]

    cand_losses, base_loss = scored[:-1], scored[-1]
    # the scores acceptance ranks by: the first stage's, or the second stage's for its candidates only
    pool, pool_losses, pool_base = list(range(len(rewrites))), list(cand_losses), base_loss
    stage2: list[int] = []
    stage2_losses: list[float] = []
    stage2_base: float | None = None
    if args.proposal_stage2_steps > 0 and rewrites:
        groups: dict[Hashable, list[int]] = {}
        for i, r in enumerate(rewrites):
            groups.setdefault(grammar.rewrite_group(r), []).append(i)
        top = args.proposal_stage2_top
        stage2 = sorted(i for g in groups.values() for i in sorted(g, key=lambda i: cand_losses[i])[:top])
        sc2 = rescore([grammar.apply(base_obj, rewrites[i]) for i in stage2] + [base_obj], args.proposal_stage2_steps)
        stage2_losses, stage2_base = sc2[:-1], sc2[-1]
        pool, pool_losses, pool_base = stage2, stage2_losses, stage2_base
    ranked = [pool[j] for j in _rank_improving(args, pool_base, pool_losses)]
    loss_of = dict(zip(pool, pool_losses))
    new_obj, accepted = base_obj, []
    combined: list[Any] = []
    verify_losses: list[float] = []
    verify_base: float | None = None
    if ranked:
        new_obj, accepted = grammar.combine(
            base_obj, [rewrites[i] for i in ranked], [pool_base - loss_of[i] for i in ranked],
            top_k=args.accept_top_k,
        )
        if args.accept_verify and len(accepted) >= 2:
            combined = list(accepted)
            prefixes = [grammar.apply_all(base_obj, combined[:k]) for k in range(1, len(combined))] + [new_obj]
            vs = rescore(prefixes + [base_obj], args.proposal_stage2_steps or args.proposal_steps)
            verify_losses, verify_base = vs[:-1], vs[-1]
            k = min(range(len(prefixes)), key=lambda j: verify_losses[j])
            if verify_losses[k] < verify_base:
                new_obj, accepted = prefixes[k], combined[: k + 1]
            else:
                new_obj, accepted = base_obj, []
    cbs.on_rewrite(
        RewriteEvent(
            step=step, rewrite_index=rewrite_index, base_loss=base_loss,
            losses=tuple(cand_losses), rewrites=tuple(rewrites), accepted=tuple(accepted),
            changed=bool(accepted), n_proposals=len(rewrites), elapsed=grammar.elapsed(),
            stage2=tuple(stage2), stage2_losses=tuple(stage2_losses), stage2_base_loss=stage2_base,
            combined=tuple(combined), verify_losses=tuple(verify_losses), verify_base_loss=verify_base,
        )
    )
    return new_obj, bool(accepted)
