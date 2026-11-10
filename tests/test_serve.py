"""`qubitlab serve`: the explorer and the API share one port."""

import asyncio
import json
import zlib
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from qubitlab import warm
from qubitlab.serve import app_with_explorer


def test_explorer_and_api_on_one_port(tmp_path: Path) -> None:
    (tmp_path / "index.html").write_text("<title>Qubit Lab</title>", encoding="utf-8")
    client = TestClient(app_with_explorer(tmp_path))
    assert "Qubit Lab" in client.get("/").text
    assert client.get("/api/health").json()["status"] == "ok"


def test_answers_and_scripts_are_gzipped(tmp_path: Path) -> None:
    (tmp_path / "index.html").write_text("<title>Qubit Lab</title>" + " " * 1000, encoding="utf-8")
    client = TestClient(app_with_explorer(tmp_path))
    for path in ("/", "/api/formulas"):
        r = client.get(path, headers={"Accept-Encoding": "gzip"})
        assert r.headers.get("content-encoding") == "gzip", path


def test_gzipped_flip_rates_still_arrive_row_by_row(tmp_path: Path) -> None:
    # The explorer draws each cat size as its line arrives, so gzip must not hold lines back.
    # A test client gathers the whole body first, so this speaks ASGI to the app directly.
    (tmp_path / "index.html").write_text("<title>Qubit Lab</title>", encoding="utf-8")
    app = app_with_explorer(tmp_path)
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/cat/rates/stream",
        "query_string": b"",
        "headers": [(b"accept-encoding", b"gzip"), (b"content-type", b"application/json")],
    }
    request = {"kappa_1": 0.1, "alpha_sq_values": [1.0, 2.0]}
    bodies: list[bytes] = []

    async def exchange() -> None:
        inbox: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        inbox.put_nowait({"type": "http.request", "body": json.dumps(request).encode()})

        async def send(message: Any) -> None:
            if message["type"] == "http.response.body":
                bodies.append(message["body"])

        await app(scope, inbox.get, send)

    asyncio.run(exchange())
    gunzip = zlib.decompressobj(16 + zlib.MAX_WBITS)
    # The header line, one line per cat size, then the end of the gzip stream.
    assert [gunzip.decompress(body).count(b"\n") for body in bodies] == [1, 1, 1, 0]


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
