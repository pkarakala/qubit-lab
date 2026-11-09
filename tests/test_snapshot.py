"""Shipped results (checks, slow solves) are served only when they match the code."""

import json
from pathlib import Path

import pytest

from qubitlab import api, snapshot


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


def test_a_shipped_solve_is_served_as_stored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "solves.json"
    key = api.solve_key("memory", "surface", 3, 0.001, 100)
    run = {"code": "surface", "distance": 3, "p": 0.001, "rounds": 3, "shots": 100}
    snapshot.write([{"key": key, "result": run | {"failures": 42}}], path)
    monkeypatch.setattr(api, "SHIPPED", path)
    api._shipped.cache_clear()
    try:
        assert api._memory("surface", 3, 0.001, 100).failures == 42  # Stim would find ~0
    finally:
        api._shipped.cache_clear()


def test_every_slow_solve_the_views_ask_for_is_shipped() -> None:
    rows = snapshot.load(api.SHIPPED)
    if rows is None:  # stale or missing: the API solves live
        return
    keys = {r["key"] for r in rows}
    for n, ratio in api.CNOT_GRID:
        k1, _, times = api.cnot_times(n, ratio)
        assert {api.solve_key("cat_cnot", n, k1, T) for T in times} <= keys
    for code, d in api.QEC_CURVES:
        assert {api.solve_key("memory", code, d, p, api.QEC_SHOTS) for p in api.P_GRID} <= keys
