# d4d — Design for Descent

Optimize structures described by a *shape grammar*: gradient descent on continuous parameters,
interleaved with discrete rewrites that change the structure itself.

Reference implementation of Stochastic Rewrite Descent (SRD) from Kodnongbua et al., *"Design for
Descent: What Makes a Shape Grammar Easy to Optimize?"*, SIGGRAPH Asia 2025. The algorithm is
grammar-generic: you supply a grammar, `d4d` supplies the optimizer.

## The idea

Many design problems are both discrete and continuous: *how many* parts, and *where* they go.
SRD alternates between the two:

1. **Descend** on the parameters of the current object for `propose_every` steps.
2. **Propose** rewrites from the grammar, score each candidate by optimizing it briefly, and
   accept every non-conflicting candidate that improves on the current object.

This works best when a rewrite is *loss-preserving at the moment it fires*. It should add
degrees of freedom without changing what the object currently represents. The discrete jump is
then free, and descent does the rest.

## Install

```bash
uv sync                                   # torch + typing_extensions only
uv sync --extra progress --extra video    # tqdm, imageio for the built-in callbacks
```

Requires Python 3.11+. pip works as well: `pip install -e .` / `pip install -e '.[progress,video]'`.

## Quickstart: a grammar over strings

The smallest useful grammar is a parametric L-system. An object is a string over `{F, R}` with one
number per symbol. `F(l)` draws forward by `l`, and `R(a)` turns left by `a` radians. The turtle
starts at the origin heading along +x. There is a single production:

```
F(l)  ->  F(l/2) R(0) F(l/2)
```

It draws exactly the same path, so it is loss-preserving. What it adds is a corner that descent can
bend. The goal is to trace a U shape, `(0,0) → (1,0) → (1,1) → (0,1)`, starting from a single
`F`.

The full, runnable version is [`tests/lsystem.py`](tests/lsystem.py). The pieces, in order:

**The object and the rewrite.** Any Python values work. The optimizer only needs to reach the
object's tensors.

```python
@dataclass(frozen=True)
class Turtle:
    program: str            # e.g. "FRFRF"
    params: torch.Tensor    # params[i] belongs to program[i]

@dataclass(frozen=True)
class Expand:
    i: int                  # apply F -> F R F at position i
```

**Construction.** `list_spec` tells the default `collate` how to take an object's tensors out
and put new ones back. `initial` is the starting point.

```python
class TurtleGrammar(Grammar[Turtle]):
    list_spec = ListSpec(
        params_of=lambda o: [o.params],
        with_params=lambda o, ts: replace(o, params=ts[0]),
    )

    def initial(self) -> Turtle:
        return Turtle("F", torch.tensor([1.0]))
```

**Rewriting.** `propose` lists candidate rewrites as cheap descriptions. `apply` performs one of
them, and must not mutate its input.

```python
    def propose(self, obj, budget):
        out = [Expand(i) for i, c in enumerate(obj.program) if c == "F"]
        return out[:budget] if budget > 0 else out

    def apply(self, obj, rewrite):
        i, v = rewrite.i, obj.params.detach()
        half = v[i : i + 1] / 2
        return Turtle(obj.program[:i] + "FRF" + obj.program[i + 1 :],
                      torch.cat([v[:i], half, torch.zeros(1), half, v[i + 1 :]]))
```

**Loss.** The loss is differentiable and returns one value per object in the batch. Here it
interprets the string into a polyline and measures how well that polyline covers the target.
Every term depends only on the drawn path, which is why `Expand` leaves the loss unchanged.

```python
    def loss(self, batch, ctx, state):
        losses = torch.stack([self._loss_one(o) for o in batch.objects])   # (len(batch),)
        return losses, {}          # second value: optional per-object diagnostics
```

Those four methods and `list_spec` are a complete grammar. Two optional hooks make it better:

**Combining.** By default every pair of rewrites conflicts, so only one is accepted per rewrite
step. Expansions at different positions are independent, so the grammar says so and applies
them right to left, which keeps the base indices valid. Several corners can now be accepted at
once.

```python
    def conflicts(self, a, b):
        return False

    def apply_all(self, base, rewrites, improvements):
        for r in sorted(rewrites, key=lambda r: -r.i):
            base = self.apply(base, r)
        return base
```

**Simplicity.** A non-differentiable price on program size. It only affects which rewrites are
accepted, weighted by `OptimizeArgs.w_simplicity`, so it stops the string from growing without
bound.

```python
    def simplicity(self, batch, ctx):
        return [self.w_symbol * len(o.program) for o in batch.objects]
```

**Run it:**

```python
result = optimize(TurtleGrammar(u_target(), w_symbol=1e-3),
                  OptimizeArgs(n_steps=600, lr=0.1, propose_every=50, seed=0))
print(result.best)
```

`uv run python tests/lsystem.py` prints the string at every rewrite event (abridged):

```
step    0  loss 6.4551  F(1.10)
step   50  loss 5.9559  F(0.84) R(+6°) F(0.84)
step  100  loss 0.0633  F(1.47) R(+137°) F(0.93) R(+6°) F(0.93)
step  150  loss 0.0228  F(1.26) R(+116°) F(1.06) R(+67°) F(0.86)
step  250  loss 0.0054  F(1.12) R(+100°) F(1.09) R(+90°) F(1.03)
best     loss 0.0068  F(1.00) R(+90°) F(1.02) R(+92°) F(1.01)
```

Starting from one `F`, each accepted `Expand` adds a corner without changing the loss, and descent
bends it into the U. Once three `F`s can draw the U exactly, further expansions stop paying for
themselves and the string stops growing. (`best` loss includes the `simplicity` term,
`5 symbols × 1e-3`.)

## The grammar interface

A `Grammar` has five stages. Only `initial`, `propose`, `apply` and `loss` are abstract.

| stage | method | notes |
|---|---|---|
| construct | `initial`, `collate` | `collate` is the only way a batch is built; set `list_spec` to get it for free |
| rewrite | `propose`, `apply` | `propose` receives the budget, so subsample before materializing |
| combine | `conflicts`, `apply_all` | the greedy search itself is inherited |
| loss | `loss`, `simplicity` | `loss` is differentiable and per-object; `simplicity` never is |
| visualize | `visualize` | returns an `(H, W, 3)` uint8 frame; callbacks persist it |

Optional hooks, for when a grammar needs them:

- `init_state` / `step_state` / `state_for_proposals`: per-run state such as annealing schedules
  or resampled points. `state_for_proposals` freezes the state so every candidate is scored
  under the same conditions.
- `cleanup`: canonicalize the object periodically, for example by merging duplicates or dropping
  degenerate parts.
- `combine`: replace the greedy search, for admissibility checks that are not pairwise, or that
  depend on the partially rewritten object. It receives the improving rewrites, best first.
- `object_cost`: the memory cost of one object, used to size batches.
- a custom `collate` returning your own `ObjectCollection`, when a packed tensor layout is faster
  than the default list.
- `config`: a JSON-able snapshot of hyperparameters, written by `ConfigWriter`.

[`tests/toy.py`](tests/toy.py) is a second worked example. It fits a piecewise-constant function
with two rule families (`Split` and `Remove`) and shows the trade-off between accuracy and
program size.

## Running the optimizer

`optimize(grammar, OptimizeArgs(...), callbacks)` returns an `OptimizeResult` with fields `best`,
`best_loss`, `best_step`, `final`, `metrics` (per-step series), `n_steps_run`, `n_rewrites` and
`stopped_early`. The arguments you are most likely to tune:

| argument | default | meaning |
|---|---|---|
| `n_steps` | 4000 | total descent steps |
| `lr`, `optimizer` | 0.5, `"Adam"` | continuous step |
| `propose_every` | 50 | steps between rewrite events |
| `proposal_size` | 0 | candidates scored per event (`0` = all) |
| `proposal_criterion`, `proposal_steps` | `"loss"`, 2 | how candidates are scored: brief optimization, or a gradient surrogate |
| `accept_top_k` | 0 | maximum rewrites accepted per event (`0` = unlimited) |
| `accept_abs_eps`, `accept_rel_eps`, `accept_eps_op` | None, None, `"or"` | improvement floors a proposal must clear; `None` disables one, both `None` means `> 0` |
| `w_simplicity` | 1.0 | weight on `Grammar.simplicity` |
| `seed` | None | makes runs reproducible |

### Side effects

The loop writes nothing to disk. Everything observable is a callback:

```python
optimize(grammar, args, [
    TqdmProgress(),
    ImageWriter("out/last.png"),
    VideoWriter("out/run.mp4", fps=5),      # flushes even if the run crashes
    MetricsWriter("out/metrics.json"),
    ConfigWriter("out/config.json"),
    HistoryRecorder(every=50),              # history is opt-in and strided
])
```

`on_run_end` fires from a `finally`, so a run killed by OOM or preemption still produces whatever
its writers had accumulated.

## Development

```bash
uv sync --all-extras                              # dev tools + the optional callback deps
uv run pytest
uv run ruff check src/d4d tests
uv run pyright --pythonpath .venv/bin/python src/d4d
uv run python tests/lsystem.py                    # the quickstart demo
```
