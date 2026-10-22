"""Disk cache for slow solves, keyed by function, arguments and package version."""

from __future__ import annotations

import functools
import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from qubitlab import __version__

CACHE_DIR = Path(os.environ.get("QUBITLAB_CACHE", Path.cwd() / ".cache" / "qubitlab"))


def disk_cached[F: Callable[..., Any]](fn: F) -> F:
    """Cache a function whose arguments and result are JSON-serializable."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        payload = json.dumps(
            {"fn": fn.__qualname__, "v": __version__, "args": args, "kwargs": kwargs},
            sort_keys=True,
            default=str,
        )
        key = hashlib.sha256(payload.encode()).hexdigest()[:24]
        path = CACHE_DIR / f"{fn.__name__}-{key}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        result = fn(*args, **kwargs)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(result), encoding="utf-8")
        tmp.replace(path)
        return result

    return wrapper  # type: ignore[return-value]
