"""Best-effort conversion of a config tree to something JSON can hold.

The core has no opinion on how a grammar is configured -- it never reads a
grammar's hyperparameters -- but a run that cannot say what produced it is not
reproducible. This is deliberately dependency-free: no yaml, no config library,
nothing that would make the core harder to install into an existing environment.
"""

from __future__ import annotations

import dataclasses
import enum
from pathlib import Path
from typing import Any

__all__ = ["to_jsonable"]

_MAX_DEPTH = 32


def to_jsonable(obj: Any, _depth: int = 0) -> Any:
    """Convert ``obj`` into JSON-serializable form. Never raises.

    Dataclasses become dicts, enums their values, paths strings, tensors a
    ``{shape, dtype}`` summary; anything else falls back to ``repr()``. Falling
    back rather than failing is the point -- a config dump must not be the thing
    that kills a twelve-hour run.
    """
    if _depth > _MAX_DEPTH:
        return "<max depth>"
    try:
        if obj is None or isinstance(obj, (bool, int, float, str)):
            return obj
        if isinstance(obj, enum.Enum):
            return to_jsonable(obj.value, _depth + 1)
        if isinstance(obj, Path):
            return str(obj)
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return {f.name: to_jsonable(getattr(obj, f.name, None), _depth + 1)
                    for f in dataclasses.fields(obj)}
        if isinstance(obj, dict):
            return {str(k): to_jsonable(v, _depth + 1) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set, frozenset)):
            return [to_jsonable(v, _depth + 1) for v in obj]
        # Duck-typed so importing torch is not required to serialize a config.
        if hasattr(obj, "shape") and hasattr(obj, "dtype"):
            return {"shape": list(getattr(obj, "shape", ())), "dtype": str(getattr(obj, "dtype", ""))}
        if isinstance(obj, type):
            return f"{obj.__module__}.{obj.__qualname__}"
        return repr(obj)
    except Exception as exc:  # noqa: BLE001 -- a config dump must never kill a run
        return f"<unserializable: {type(exc).__name__}>"
