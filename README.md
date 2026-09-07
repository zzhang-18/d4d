# d4d — Design for Descent

Optimizing structures described through a *shape grammar*.

Reference implementation of Stochastic Rewrite Descent (SRD) from Kodnongbua et al., *"Design for
Descent: What Makes a Shape Grammar Easy to Optimize?"*, SIGGRAPH Asia 2025. Built as an interface
for custom grammars.

## The idea

Many design problems are jointly discrete and continuous: how many parts, and
where. The algorithm interleaves continuous and discrete updates. Descend on parameters, 
and every `propose_every` steps let the grammar offer rewrites that change the structure. 
Each candidate is scored by actually optimizing it briefly, and all non-conflicting improvements
are accepted at once.

## Install

```bash
uv sync                                   # torch + typing_extensions only
uv sync --extra progress --extra video    # tqdm, imageio for the built-in callbacks
```

Python 3.11+

pip works as well:
`pip install -e .` / `pip install -e '.[progress,video]'`.

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
piecewise-constant function with segments.

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

## Development

```bash
uv sync --all-extras                              # dev tools + the optional callback deps
uv run pytest                                     # 56 tests
uv run ruff check src/d4d tests
uv run pyright --pythonpath .venv/bin/python src/d4d
```
