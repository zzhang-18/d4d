# d4d — Design for Descent

[[Paper]](https://www.computationaldesign.group/assets/papers/SIGA-2025-D4Descent.pdf)
[[DOI]](https://doi.org/10.1145/3757377.3764004)
[[Project Page]](https://www.computationaldesign.group/publications/design-for-descent)
[[Original Code]](https://github.com/milmillin/d4descent)

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
pip install d4d
```

```bash
uv add d4d
```

Requires Python 3.11+.

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

The whole grammar, typed. [`tests/lsystem.py`](tests/lsystem.py) is the runnable version, with
`_loss_one` filled in.

```python
import random
from collections.abc import Sequence
from dataclasses import dataclass, replace
import torch

from d4d import ExtraMetrics, Grammar, ListCollection, ListSpec, OptimizeArgs, StepContext, optimize

####################
# The Object
####################

@dataclass(frozen=True)
class Turtle:
    program: str            # e.g. "FRFRF"
    params: torch.Tensor    # (len(program),); params[i] belongs to program[i]

####################
# The Rewrites
####################

@dataclass(frozen=True)
class Expand:
    i: int                  # position of the F to expand


Rewrite = Expand            # Expand | Contract | ... once there are more rules

####################
# The Grammar
####################

# Grammar[TObject, TCollection, TRewrite, TState]
class TurtleGrammar(Grammar[Turtle, ListCollection[Turtle], Rewrite, None]):
    list_spec = ListSpec(
        # How to extract the parameters from the object
        params_of=lambda o: [o.params],
        # How to replace the parameters in the object.
        # This must create a new instance of the object.
        with_params=lambda o, ts: replace(o, params=ts[0]),
    )

    def __init__(self, target: torch.Tensor) -> None:
        self.target = target  # (N, 2) points to trace

    def initial(self) -> Turtle:
        """The starting object."""
        return Turtle("F", torch.tensor([1.0]))

    def propose(self, obj: Turtle, budget: int) -> list[Rewrite]:
        """Sample different rewrites for the object."""
        specs = [Expand(i) for i, c in enumerate(obj.program) if c == "F"]
        # Sample if too many
        if len(specs) > budget and budget > 0:
            specs = random.sample(specs, budget)
        return specs

    def apply(self, obj: Turtle, rewrite: Rewrite) -> Turtle:
        if isinstance(rewrite, Expand):
            i, v = rewrite.i, obj.params.detach()
            half = v[i : i + 1] / 2
            return Turtle(obj.program[:i] + "FRF" + obj.program[i + 1 :],
                          torch.cat([v[:i], half, torch.zeros(1), half, v[i + 1 :]]))
        # elif isinstance(rewrite, Contract):
        #     ...
        else:
            raise NotImplementedError(f"Unknown rewrite {rewrite}")

    def conflicts(self, a: Rewrite, b: Rewrite) -> bool:
        """Whether two rewrites conflict."""
        if isinstance(a, Expand) and isinstance(b, Expand):
            return a.i == b.i
        return False

    def apply_all(self, base: Turtle, rewrites: Sequence[Rewrite]) -> Turtle:
        """Optional. Apply right to left, so the base indices stay valid."""
        for r in sorted(rewrites, key=lambda r: -r.i):
            base = self.apply(base, r)
        return base

    def loss(
        self, batch: ListCollection[Turtle], ctx: StepContext, state: None
    ) -> tuple[torch.Tensor, ExtraMetrics]:
        """(B,) differentiable losses, plus per-object diagnostics (none here)."""
        return torch.stack([self._loss_one(o) for o in batch.objects]), {}

    def _loss_one(self, obj: Turtle) -> torch.Tensor:
        """returns: scalar, distance from the drawn polyline to target."""
        ...

    def simplicity(self, batch: ListCollection[Turtle], ctx: StepContext) -> list[float]:
        """Optional. Usually the program size, weighted by OptimizeArgs.w_simplicity; never differentiated."""
        return [len(o.program) for o in batch.objects]


result = optimize(
    TurtleGrammar(u_target()),
    OptimizeArgs(n_steps=600, lr=0.1, propose_every=50, w_simplicity=1e-3, seed=0),
)
print(result.best)
```

`list_spec`, `initial`, `propose`, `apply`, `conflicts` and `loss` are required. A `conflicts` that
always returns True accepts one rewrite per rewrite event.

Optional hooks, for when a grammar needs them:

- `cleanup`: canonicalize the object periodically, for example by merging duplicates or dropping
  degenerate parts.
- `combine`: replace the greedy search, for admissibility checks that are not pairwise, or that
  depend on the partially rewritten object. It receives the improving rewrites, best first.
- `object_cost`: the memory cost of one object, used to size batches.
- a custom `collate` returning your own `ObjectCollection`, when a packed tensor layout is faster
  than the default list; see [Custom ObjectCollection](docs/usage.md#custom-objectcollection).
- `init_state` / `step_state` / `state_for_proposals`: per-run state such as annealing schedules
  or resampled points. `state_for_proposals` freezes the state so every candidate is scored
  under the same conditions. See [the optimization loop](docs/usage.md#the-optimization-loop) for
  when each is called.
- `config`: a JSON-able snapshot of hyperparameters, used by `ConfigWriter`.

## Running the optimizer

`optimize(grammar, OptimizeArgs(...), callbacks)` returns an `OptimizeResult` with fields `best`,
`best_loss`, `best_step`, `final`, `metrics` (per-step series), `n_steps_run`, `n_rewrites` and
`stopped_early`. Every argument, with its default and meaning, is documented on `OptimizeArgs` in
[`src/d4d/optimize.py`](src/d4d/optimize.py).

## Callbacks

Callbacks observe a run without changing it. The loop itself writes nothing to disk: progress bars,
images, videos, metrics and checkpoints are all callbacks. To write your own, subclass `Callback` and
override any of `on_run_start`, `on_step_end`, `on_visualize`, `on_rewrite` and `on_run_end`. Raising
`StopRun` from a hook ends the run early; any other exception in a callback becomes a warning.

```python
history = HistoryRecorder(every=50)         # keeps every 50th step's object in memory

result = optimize(grammar, args, [
    TqdmProgress(),
    ImageWriter("out/last.png"),
    VideoWriter("out/run.mp4", fps=5),      # flushes even if the run crashes
    MetricsWriter("out/metrics.json"),
    ConfigWriter("out/config.json"),
    history,
])

for step, obj in zip(history.steps, history.objects):
    ...
```

For where `optimize` calls each grammar hook and callback, see
[the optimization loop](docs/usage.md#the-optimization-loop) in `docs/usage.md`.

## Development

```bash
uv sync --extra dev                               # pytest, pyright, ruff
uv run pytest
uv run ruff check src/d4d tests
uv run pyright --pythonpath .venv/bin/python src/d4d
uv run python tests/lsystem.py                    # the quickstart demo
```

## Citation

If you use d4d in your research, please cite:

```bibtex
@inproceedings{kodnongbua2025d4descent,
  author    = {Kodnongbua, Milin and Zhang, Zihan and Sharp, Nicholas and Schulz, Adriana},
  title     = {Design for Descent: What Makes a Shape Grammar Easy to Optimize?},
  year      = {2025},
  isbn      = {9798400721373},
  publisher = {Association for Computing Machinery},
  address   = {New York, NY, USA},
  url       = {https://doi.org/10.1145/3757377.3764004},
  doi       = {10.1145/3757377.3764004},
  booktitle = {Proceedings of the SIGGRAPH Asia 2025 Conference Papers},
  articleno = {172},
  numpages  = {11},
  location  = {Hong Kong, Hong Kong},
  series    = {SA Conference Papers '25},
  keywords  = {optimization, shape grammar, procedural modeling},
}
```

## License

d4d is licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE). It permits use,
modification and distribution for noncommercial purposes only; see [`LICENSE`](LICENSE) for the terms.
