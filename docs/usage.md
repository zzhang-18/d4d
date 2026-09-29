# Usage notes

## The optimization loop

Where `optimize` calls the grammar's hooks and the callbacks, simplified from
[`src/d4d/optimize.py`](../src/d4d/optimize.py):

```python
def optimize(grammar, args, callbacks):
    obj = grammar.initial()
    batch = grammar.collate([obj])
    state = grammar.init_state()
    on_run_start(...)
    try:
        for step in range(args.n_steps):
            if step % args.cleanup_every == 0:
                batch = grammar.cleanup(batch)

            if is_rewrite_step(step):                   # every propose_every steps, or on a loss plateau
                if stalled_too_long:                    # early stopping
                    break
                base = batch.get(0)
                rewrites = grammar.propose(base, args.proposal_size)
                if rewrites:
                    candidates = [grammar.apply(base, r) for r in rewrites] + [base]
                    prop_state = grammar.state_for_proposals(state)
                    # grammar.loss(phase="proposal") + w_simplicity * grammar.simplicity
                    scores = score(candidates, prop_state)
                    ranked = improving(rewrites, scores)          # best first
                    new_obj, accepted = grammar.combine(base, ranked, ...)  # conflicts, apply_all
                    on_rewrite(...)
                    if accepted:
                        batch = grammar.collate([new_obj])

            losses, extra = grammar.loss(batch, ctx(phase="step"), state)
            if step % args.visualize_every == 0 or is_rewrite_step(step):
                on_visualize(grammar.visualize(batch, ctx, state))   # skipped when it returns None
            descent_step(losses.sum())
            state = grammar.step_state(state)
            on_step_end(...)
    except StopRun:
        pass                                            # stopped early; a result is still returned
    finally:                                            # also on errors, OOM and preemption
        on_run_end(...)
```

## Custom ObjectCollection

Setting `list_spec` gives a grammar the default `ListCollection`, a list of objects that each keep
their own tensors. That is enough to get started, but `loss` then loops over `batch.objects` in
Python, and the optimizer steps `B × k` separate small tensors. When the objects in a batch share a
layout, for example a fixed or padded number of primitives, a custom collection can store the whole
batch as a few stacked tensors such as `(B, P, D)`, so `loss` scores every object in one vectorized
pass. Subclass `ObjectCollection`, return it from `Grammar.collate` instead of setting `list_spec`,
and name it as the grammar's second type parameter: `Grammar[MyObject, MyBatch, MyRewrite, None]`.

The abstract methods, and where the optimizer uses them:

| method | used for |
|---|---|
| `__len__()` | the batch size `B` |
| `get(i, detach=True)` | extracting one object: the base for rewrites, the best and final objects, `StepEnd.get_object()` |
| `parameters()` | the leaf tensors handed to `torch.optim` and to gradient clipping |
| `requires_grad_()` | re-leafing every parameter after each `collate` and `cleanup` |
| `per_object_grads()` | `B` flat `(D_i,)` gradients, for `proposal_criterion="grad"` / `"grad_only"` |
| `parameter_names()`, `clone()`, `to(device)` | part of the contract; the loop itself does not call them |

`get(i)` must return an object built on the same tensors that `parameters()` exposes. Because
`requires_grad_()` replaces every parameter with a fresh leaf, it must rebuild the objects around the
new tensors too; otherwise `loss` reads stale tensors that never receive a gradient.

Two methods are optional: `scale_grads_()` rescales gradients in place before each optimizer step
and returns True if it changed any, and `project_to_valid_()` projects parameters back into their
feasible set after each step.
