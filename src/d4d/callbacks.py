"""Run events and the callbacks that observe them.

``on_run_end`` fires from a ``finally``, including when the run raises.
Object-typed fields are ``Any``.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

    from .optimize import OptimizeArgs, OptimizeResult

__all__ = [
    "BestObjectWriter",
    "Callback",
    "CallbackList",
    "CheckpointWriter",
    "ConfigWriter",
    "DebugPrinter",
    "EarlyStopOnNaN",
    "HistoryRecorder",
    "ImageWriter",
    "MetricsWriter",
    "RewriteEvent",
    "RunEnd",
    "RunStart",
    "StepEnd",
    "StopRun",
    "TqdmProgress",
    "VideoWriter",
    "VisualizeEvent",
]


class StopRun(Exception):
    """Raised by a callback to halt the run cleanly. The run-end hook still fires."""


# -- events ----------------------------------------------------------------


@dataclass(frozen=True)
class RunStart:
    grammar: Any
    args: OptimizeArgs
    initial: Any


@dataclass(frozen=True)
class StepEnd:
    """One completed optimization step."""

    step: int
    total_steps: int
    loss: float
    """Total loss including the simplicity term -- the ``$loss`` series."""
    loss_cont: float
    """Loss without the simplicity term."""
    loss_simplicity: float
    loss_ma: float
    lr: float
    rewrite: bool
    elapsed: float
    extra: Mapping[str, float]
    _get_object: Callable[[], Any] = field(repr=False, default=lambda: None)

    def get_object(self) -> Any:
        """Materialize the current object; evaluated only when called."""
        return self._get_object()


@dataclass(frozen=True)
class VisualizeEvent:
    step: int
    loss: float
    image: np.ndarray
    """``(H, W, 3)`` uint8 frame from ``Grammar.visualize``."""


@dataclass(frozen=True)
class RewriteEvent:
    """One discrete rewrite step, after candidates were scored and combined."""

    step: int
    rewrite_index: int
    base_loss: float
    losses: tuple[float, ...]
    rewrites: tuple[Any, ...]
    accepted: tuple[Any, ...]
    changed: bool
    n_proposals: int
    elapsed: float

    def improvements(self) -> list[float]:
        """Per-candidate loss improvement over the base. Negative means worse."""
        return [self.base_loss - l for l in self.losses]


@dataclass(frozen=True)
class RunEnd:
    result: OptimizeResult[Any] | None
    error: BaseException | None
    """Set when the run raised; ``result`` is then None."""


# -- protocol --------------------------------------------------------------


class Callback:
    """Observe a run. Override only the hooks you need."""

    def on_run_start(self, ev: RunStart) -> None: ...
    def on_step_end(self, ev: StepEnd) -> None: ...
    def on_visualize(self, ev: VisualizeEvent) -> None: ...
    def on_rewrite(self, ev: RewriteEvent) -> None: ...
    def on_run_end(self, ev: RunEnd) -> None: ...


class CallbackList(Callback):
    """Fan out to several callbacks. Exceptions become warnings, except :class:`StopRun`, which propagates."""

    def __init__(self, callbacks: Callback | Sequence[Callback] | None = None) -> None:
        if callbacks is None:
            self.callbacks: list[Callback] = []
        elif isinstance(callbacks, Callback):
            self.callbacks = [callbacks]
        else:
            self.callbacks = list(callbacks)

    def _fan(self, hook: str, ev: Any) -> None:
        for cb in self.callbacks:
            try:
                getattr(cb, hook)(ev)
            except StopRun:
                raise
            except Exception as exc:  # noqa: BLE001
                warnings.warn(f"{type(cb).__name__}.{hook} raised {type(exc).__name__}: {exc}", stacklevel=2)

    def on_run_start(self, ev: RunStart) -> None: self._fan("on_run_start", ev)
    def on_step_end(self, ev: StepEnd) -> None: self._fan("on_step_end", ev)
    def on_visualize(self, ev: VisualizeEvent) -> None: self._fan("on_visualize", ev)
    def on_rewrite(self, ev: RewriteEvent) -> None: self._fan("on_rewrite", ev)
    def on_run_end(self, ev: RunEnd) -> None: self._fan("on_run_end", ev)


# -- built-ins -------------------------------------------------------------

SaveFn = Callable[[Any, Path], None]
"""Save one object to a path."""


class TqdmProgress(Callback):
    """Progress bar. Needs the ``progress`` extra; warns and does nothing without it."""

    def __init__(self, **kwargs: Any) -> None:
        self._kwargs = kwargs
        self._bar: Any = None

    def on_run_start(self, ev: RunStart) -> None:
        try:
            from tqdm.auto import tqdm
        except ImportError:
            warnings.warn("TqdmProgress needs tqdm: pip install 'd4d[progress]'", stacklevel=2)
            return
        self._bar = tqdm(total=ev.args.n_steps, **self._kwargs)

    def on_step_end(self, ev: StepEnd) -> None:
        if self._bar is None:
            return
        self._bar.update(1)
        self._bar.set_description(f"loss={ev.loss:.3e} lr={ev.lr:.2e}")

    def on_run_end(self, ev: RunEnd) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None


class ImageWriter(Callback):
    """Write the latest frame to a fixed path, overwriting it each time."""

    def __init__(self, path: str | Path, save_fn: Callable[[np.ndarray, Path], None] | None = None) -> None:
        self.path = Path(path)
        self.save_fn = save_fn or _default_save_image

    def on_visualize(self, ev: VisualizeEvent) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.save_fn(ev.image, self.path)


def _default_save_image(image: np.ndarray, path: Path) -> None:
    try:
        import imageio.v3 as iio
    except ImportError as exc:
        raise RuntimeError("ImageWriter needs imageio: pip install 'd4d[video]'") from exc
    iio.imwrite(path, image)


class VideoWriter(Callback):
    """Collect frames and encode them to a video at run end, including when the run raised."""

    def __init__(self, path: str | Path, fps: int = 5, max_frames: int | None = None) -> None:
        self.path = Path(path)
        self.fps = fps
        self.max_frames = max_frames
        self.frames: list[np.ndarray] = []

    def on_visualize(self, ev: VisualizeEvent) -> None:
        if self.max_frames is not None and len(self.frames) >= self.max_frames:
            return
        self.frames.append(ev.image)

    def on_run_end(self, ev: RunEnd) -> None:
        if not self.frames:
            return
        try:
            import imageio.v3 as iio
        except ImportError:
            warnings.warn("VideoWriter needs imageio: pip install 'd4d[video]'", stacklevel=2)
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(self.path, self.frames, fps=self.fps, codec="libx264")


class MetricsWriter(Callback):
    """Persist the metric series at run end. Defaults to JSON; pass ``torch.save`` for ``.pt``."""

    def __init__(self, path: str | Path, save_fn: Callable[[Any, Path], None] | None = None) -> None:
        self.path = Path(path)
        self.save_fn = save_fn or _default_save_json

    def on_run_end(self, ev: RunEnd) -> None:
        if ev.result is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.save_fn(dict(ev.result.metrics), self.path)


def _default_save_json(obj: Any, path: Path) -> None:
    from .serialize import to_jsonable

    path.write_text(json.dumps(to_jsonable(obj), indent=2))


class BestObjectWriter(Callback):
    """Persist the best-scoring object at run end. ``save_fn`` is required."""

    def __init__(self, path: str | Path, save_fn: SaveFn) -> None:
        self.path = Path(path)
        self.save_fn = save_fn

    def on_run_end(self, ev: RunEnd) -> None:
        if ev.result is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.save_fn(ev.result.best, self.path)


class HistoryRecorder(Callback):
    """Keep the object of every ``every``-th step in memory."""

    def __init__(self, every: int = 1) -> None:
        if every < 1:
            raise ValueError(f"every must be >= 1, got {every}")
        self.every = every
        self.steps: list[int] = []
        self.objects: list[Any] = []

    def on_step_end(self, ev: StepEnd) -> None:
        if ev.step % self.every == 0:
            self.steps.append(ev.step)
            self.objects.append(ev.get_object())


class CheckpointWriter(Callback):
    """Save the current object every ``every`` steps."""

    def __init__(self, directory: str | Path, every: int, save_fn: SaveFn,
                 name: str = "step_{step:07d}") -> None:
        if every < 1:
            raise ValueError(f"every must be >= 1, got {every}")
        self.directory = Path(directory)
        self.every = every
        self.save_fn = save_fn
        self.name = name

    def on_step_end(self, ev: StepEnd) -> None:
        if ev.step % self.every != 0:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        self.save_fn(ev.get_object(), self.directory / self.name.format(step=ev.step))


class ConfigWriter(Callback):
    """Write the args, grammar config, argv and start time as JSON at run start."""

    def __init__(self, path: str | Path, extra: Mapping[str, Any] | None = None) -> None:
        self.path = Path(path)
        self.extra = dict(extra or {})

    def on_run_start(self, ev: RunStart) -> None:
        import sys
        import time

        from .serialize import to_jsonable

        payload = {
            "optimize": to_jsonable(ev.args),
            "grammar_type": f"{type(ev.grammar).__module__}.{type(ev.grammar).__qualname__}",
            "grammar": to_jsonable(ev.grammar.config()),
            "argv": list(sys.argv),
            "started_at": time.time(),
            **{k: to_jsonable(v) for k, v in self.extra.items()},
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, indent=2))


class DebugPrinter(Callback):
    """Print each rewrite event's ``top_k`` candidates, best improvement first."""

    def __init__(self, every: int = 1, top_k: int = 10) -> None:
        self.every = every
        self.top_k = top_k

    def on_rewrite(self, ev: RewriteEvent) -> None:
        if self.every < 1 or ev.rewrite_index % self.every != 0:
            return
        ranked = sorted(zip(ev.improvements(), ev.rewrites), key=lambda x: -x[0])
        print(f"[rewrite {ev.rewrite_index} @ step {ev.step}] "
              f"{ev.n_proposals} proposals, {len(ev.accepted)} accepted, base={ev.base_loss:.4e}")
        for imp, rw in ranked[: self.top_k]:
            print(f"   {imp:+.3e}  {rw}")


class EarlyStopOnNaN(Callback):
    """Raise :class:`StopRun` when the loss is non-finite."""

    def on_step_end(self, ev: StepEnd) -> None:
        if ev.loss != ev.loss or ev.loss in (float("inf"), float("-inf")):
            raise StopRun(f"non-finite loss {ev.loss} at step {ev.step}")
