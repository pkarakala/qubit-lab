"""Shipped check results are served only when they match the code."""

import json
from pathlib import Path

from qubitlab import snapshot


def test_round_trip_and_stale_snapshot_is_ignored(tmp_path: Path) -> None:
    path = tmp_path / "checks.json"
    rows = [{"key": "x", "verdict": "AGREE", "expected": "AGREE"}]
    snapshot.write(rows, path)
    assert snapshot.load(path) == rows
    data = json.loads(path.read_text())
    data["fingerprint"] = "stale"
    path.write_text(json.dumps(data))
    assert snapshot.load(path) is None
    assert snapshot.load(tmp_path / "missing.json") is None


def test_shipped_snapshot_has_no_surprises() -> None:
    rows = snapshot.load()
    if rows is None:  # stale or missing: the API computes live, so nothing to check here
        return
    assert all(r["verdict"] == r["expected"] for r in rows)
