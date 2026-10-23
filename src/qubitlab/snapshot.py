"""Shipped check results, so the Checks page opens at once instead of re-running every solve.

`scripts/make_reference.py` writes data/checks.json with the results and a fingerprint of the
code that produced them (every file in theory/, sim/ and checks/). The API serves the snapshot
only when the fingerprint matches the code it runs; otherwise it computes the checks live.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

PACKAGE = Path(__file__).resolve().parent
SNAPSHOT = PACKAGE / "data" / "checks.json"
SOURCES = ("theory", "sim", "checks")


def fingerprint() -> str:
    h = hashlib.sha256()
    for folder in SOURCES:
        for path in sorted((PACKAGE / folder).glob("*.py")):
            h.update(path.name.encode())
            h.update(path.read_bytes())
    return h.hexdigest()[:16]


def write(results: list[dict[str, Any]], path: Path = SNAPSHOT) -> None:
    data = {"fingerprint": fingerprint(), "results": results}
    path.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")


def load(path: Path = SNAPSHOT) -> list[dict[str, Any]] | None:
    """The shipped results, or None if missing or made by different code."""
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("fingerprint") != fingerprint():
        return None
    results: list[dict[str, Any]] = data["results"]
    return results
