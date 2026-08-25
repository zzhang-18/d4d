"""d4d -- Design for Descent.

Grammar-guided hybrid discrete/continuous optimization: gradient descent on an
object's parameters, interleaved with discrete rewrites drawn from a grammar you
define.

Define a grammar, hand it to :func:`optimize`::

    from d4d import Grammar, ListSpec, OptimizeArgs, optimize

    class MyGrammar(Grammar[MyObject, MyBatch]):
        def initial(self) -> MyObject: ...
        def collate(self, objects) -> MyBatch: ...
        def propose(self, obj, budget) -> list[MyRewrite]: ...
        def apply(self, obj, rewrite) -> MyObject: ...
        def loss(self, batch, ctx, state): ...

    result = optimize(MyGrammar(), OptimizeArgs(n_steps=2000))

Reference: Kodnongbua et al., "Design for Descent: What Makes a Shape Grammar
Easy to Optimize?", SIGGRAPH Asia 2025.
"""

from ._util import MovingAverage, maybe_clamp, safe_cat, safe_stack
from .callbacks import (
    BestObjectWriter,
    Callback,
    CallbackList,
    CheckpointWriter,
    ConfigWriter,
    DebugPrinter,
    EarlyStopOnNaN,
    HistoryRecorder,
    ImageWriter,
    MetricsWriter,
    RewriteEvent,
    RunEnd,
    RunStart,
    StepEnd,
    StopRun,
    TqdmProgress,
    VideoWriter,
    VisualizeEvent,
)
from .collection import ListCollection, ListSpec, ObjectCollection, batchify
from .combine import DEFAULT_ACCEPT_RULE, REJECT, AcceptRule, greedy_combine
from .grammar import ExtraMetrics, Grammar, Phase, StepContext
from .optimize import OptimizeArgs, OptimizeResult, optimize
from .retry import run_with_oom_backoff
from .scheduler import AdaptiveLRScheduler
from .serialize import to_jsonable

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # core
    "Grammar", "StepContext", "Phase", "ExtraMetrics",
    "ObjectCollection", "ListCollection", "ListSpec", "batchify",
    "AcceptRule", "DEFAULT_ACCEPT_RULE", "REJECT", "greedy_combine",
    "OptimizeArgs", "OptimizeResult", "optimize",
    "AdaptiveLRScheduler", "run_with_oom_backoff", "to_jsonable",
    # callbacks
    "Callback", "CallbackList", "StopRun",
    "RunStart", "StepEnd", "VisualizeEvent", "RewriteEvent", "RunEnd",
    "TqdmProgress", "ImageWriter", "VideoWriter", "MetricsWriter",
    "BestObjectWriter", "HistoryRecorder", "CheckpointWriter",
    "ConfigWriter", "DebugPrinter", "EarlyStopOnNaN",
    # utils
    "MovingAverage", "maybe_clamp", "safe_cat", "safe_stack",
]
