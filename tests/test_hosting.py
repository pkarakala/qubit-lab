"""The public-hosting switches in qubitlab.api, applied to `qubitlab serve`."""

import importlib
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import qubitlab.api
from qubitlab.serve import app_with_explorer


@pytest.fixture
def hosted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    (tmp_path / "index.html").write_text("<title>Qubit Lab</title>")
    monkeypatch.setenv("QUBITLAB_RATE", "3/minute")
    monkeypatch.setenv("QUBITLAB_SOLVES", "2")
    importlib.reload(qubitlab.api)
    with TestClient(app_with_explorer(tmp_path)) as client:
        yield client
    monkeypatch.undo()
    importlib.reload(qubitlab.api)


def test_serves_the_explorer_and_the_api(hosted: TestClient) -> None:
    assert "Qubit Lab" in hosted.get("/").text
    assert hosted.get("/api/health").status_code == 200


def test_rate_limit_answers_429(hosted: TestClient) -> None:
    codes = [hosted.get("/api/health").status_code for _ in range(4)]
    assert codes == [200, 200, 200, 429]


def test_warm_runs_in_the_background_on_start(monkeypatch: pytest.MonkeyPatch) -> None:
    import threading

    import qubitlab.warm

    started = threading.Event()
    monkeypatch.setattr(qubitlab.warm, "warm", started.set)
    monkeypatch.setenv("QUBITLAB_WARM", "1")
    importlib.reload(qubitlab.api)
    try:
        with TestClient(qubitlab.api.app) as client:
            assert client.get("/api/health").status_code == 200
        assert started.wait(5)
    finally:
        monkeypatch.undo()
        importlib.reload(qubitlab.api)
