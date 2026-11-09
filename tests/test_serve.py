"""`qubitlab serve`: the explorer and the API share one port."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from qubitlab import warm
from qubitlab.serve import app_with_explorer


def test_explorer_and_api_on_one_port(tmp_path: Path) -> None:
    (tmp_path / "index.html").write_text("<title>Qubit Lab</title>", encoding="utf-8")
    client = TestClient(app_with_explorer(tmp_path))
    assert "Qubit Lab" in client.get("/").text
    assert client.get("/api/health").json()["status"] == "ok"


def test_missing_explorer_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="build:package"):
        app_with_explorer(tmp_path)


def test_requests_hold_the_warm_up(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The background warm-up waits between its steps while a visitor's request is answered.
    calls: list[str] = []
    monkeypatch.setattr(warm, "request_started", lambda: calls.append("started"))
    monkeypatch.setattr(warm, "request_finished", lambda: calls.append("finished"))
    (tmp_path / "index.html").write_text("<title>Qubit Lab</title>", encoding="utf-8")
    TestClient(app_with_explorer(tmp_path)).get("/api/health")
    assert calls == ["started", "finished"]
