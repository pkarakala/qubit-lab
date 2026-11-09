"""Disk cache for slow solves, keyed by function, arguments and the code that ran them.

The cache lives in $QUBITLAB_CACHE, else $XDG_CACHE_HOME/qubitlab, else ~/.cache/qubitlab,
so every folder `qubitlab serve` starts from shares it. A result is reused only by the same
package version and the same code, so an edited checkout or a new release never serves answers
from other code. Old entries are never read again; deleting the folder is always safe.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from qubitlab import __version__
from qubitlab.snapshot import fingerprint

CACHE_DIR = Path(
    os.environ.get("QUBITLAB_CACHE")
    or Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "qubitlab"
)
# Every file that can change a cached answer: the modules and the architecture scenes.
CODE = fingerprint(("*.py", "theory/*.py", "sim/*.py", "checks/*.py", "data/architectures/*"))
_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _lock(key: str) -> threading.Lock:
    with _locks_guard:
        return _locks.setdefault(key, threading.Lock())


def disk_cached[F: Callable[..., Any]](fn: F) -> F:
    """Cache a function whose arguments and result are JSON-serializable."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        payload = json.dumps(
            {"fn": fn.__qualname__, "v": __version__, "code": CODE, "args": args, "kwargs": kwargs},
            sort_keys=True,
            default=str,
        )
        key = hashlib.sha256(payload.encode()).hexdigest()[:24]
        path = CACHE_DIR / f"{fn.__name__}-{key}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        # One solve per key: a second caller (the warm-up and a click) waits for the first.
        with _lock(key):
            if path.exists():
                return json.loads(path.read_text(encoding="utf-8"))
            result = fn(*args, **kwargs)
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(f".{os.getpid()}.tmp")  # other processes may share the dir
            tmp.write_text(json.dumps(result), encoding="utf-8")
            tmp.replace(path)
        return result

    return wrapper  # type: ignore[return-value]
