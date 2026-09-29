"""Best-effort conversion of a config tree to JSON-serializable values."""

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
    ``{shape, dtype}`` summary, classes their qualified name; anything else
    falls back to ``repr()``.
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
        # tensors and arrays, duck-typed
        if hasattr(obj, "shape") and hasattr(obj, "dtype"):
            return {"shape": list(getattr(obj, "shape", ())), "dtype": str(getattr(obj, "dtype", ""))}
        if isinstance(obj, type):
            return f"{obj.__module__}.{obj.__qualname__}"
        return repr(obj)
    except Exception as exc:  # noqa: BLE001
        return f"<unserializable: {type(exc).__name__}>"
