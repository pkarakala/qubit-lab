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
