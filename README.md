# d4d — Design for Descent

Grammar-guided hybrid optimization: gradient descent on an object's parameters,
interleaved with discrete rewrites drawn from a grammar you define.

Reference implementation of the algorithm from Kodnongbua et al., *"Design for
Descent: What Makes a Shape Grammar Easy to Optimize?"*, SIGGRAPH Asia 2025,
extracted from `d4descent` so it can be reused with arbitrary grammars.

## The idea

Many design problems are jointly discrete and continuous: how many parts, and
where. Gradient descent handles *where*; it cannot change *how many*. Search
handles *how many*, but combinatorially.

The algorithm interleaves them. Descend on parameters, and every `propose_every`
steps let the grammar offer rewrites that change the structure. Each candidate is
scored by actually optimizing it briefly, and all non-conflicting improvements
are accepted at once.

What makes this work is a property of the *grammar*, not the optimizer: a rule
should fire only when it is **loss-preserving at that moment** — when splitting a
part in two, or adding one of zero size, leaves the represented object unchanged.
The discrete jump then costs nothing, and descent simply continues in a larger
space. A grammar whose rewrites spike the loss will be rejected by the very
search meant to accept them.

## Install

```bash
pip install -e .            # torch + typing_extensions only
pip install -e '.[progress,video]'   # tqdm, imageio for the built-in callbacks
```

Python 3.11+. The core has no numpy, matplotlib or config-library dependency.

## Usage

```python
from d4d import Grammar, ListSpec, OptimizeArgs, optimize

class MyGrammar(Grammar[MyObject, MyBatch]):
    list_spec = ListSpec(params_of=..., with_params=...)   # or override collate()

    def initial(self) -> MyObject: ...
    def propose(self, obj, budget) -> list[MyRewrite]: ...
    def apply(self, obj, rewrite) -> MyObject: ...
    def loss(self, batch, ctx, state): ...        # -> (Tensor (len(batch),), extras)

result = optimize(MyGrammar(), OptimizeArgs(n_steps=2000))
```

`tests/toy.py` is a complete worked example in ~140 lines: fitting a
piecewise-constant function whose segment count the grammar decides.

### The five stages

| stage | method | notes |
|---|---|---|
| construct | `initial`, `collate` | `collate` is the only way a batch is built |
| rewrite | `propose`, `apply` | `propose` receives the budget, so subsample before materializing |
| combine | `conflicts` / `combine_admit`, `apply_all` | the greedy search itself is inherited |
| loss | `loss`, `simplicity` | `loss` is differentiable and per-object; `simplicity` never is |
| visualize | `visualize` | returns a frame; callbacks persist it |

Only `initial`, `propose`, `apply` and `loss` are abstract.

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

`on_run_end` fires from a `finally`, so a run killed by OOM or preemption still
produces whatever its writers had accumulated.

## Differences from `d4descent`

The algorithm is unchanged. The interface is not:

- **`Grammar` replaces `Task`.** One class, five stages. `TCollection`,
  `TRewrite` and `TState` have defaults, so `Grammar[MyObject]` is a valid
  spelling.
- **No `Renderable` on the collection.** 2-D SDF rasterization was welded into
  the algorithm's core abstraction; rendering is a grammar concern.
- **No classmethod constructors.** `collate` is a bound method that closes over
  the grammar's configuration, which removes the need for the dynamic-subclass
  `patch_args` trick used to smuggle per-collection arguments past a bare class.
- **`combine` is inherited, not reimplemented.** All four upstream conflict
  styles — pairwise index, pairwise node id, no-conflict-with-scores, and
  stateful-against-the-partial-result — are expressible through `conflicts`,
  `combine_admit` and `apply_all`.
- **Losses are values, not base classes.** Loss-as-mixin produced MRO diamonds
  and let a *loss* override object initialization. There is no `Objective` type:
  a grammar with one loss implements `loss()`; one with several takes a callable.
- **`accept_top_k: int` replaces `proposal_accept_parallel: bool`,** whose
  disagreement with the integer `combine_proposals` actually took made three
  upstream grammars raise `TypeError` at their first rewrite.
- **`OptimizeResult.best` is the argmin,** not the last step's value.
- **History is opt-in.** Upstream deep-copied the population to CPU every step
  whether or not anyone wanted it.
- **`cost_budget` replaces `batch_param_count`,** with a `Grammar.object_cost`
  hook, because peak memory is not always proportional to parameter count.

## Development

```bash
pytest                                    # 56 tests
ruff check src/d4d tests
pyright --pythonpath $(which python) src/d4d
```
