"""Every Copy-as-Python script runs, with inputs as the explorer sends them."""

import contextlib
import io

import pytest
from fastapi.testclient import TestClient

from qubitlab.api import app

client = TestClient(app)

# Small inputs, so the whole file runs in seconds.
CASES = {
    "phase": {"alpha_sq": 2, "kappa_1": 0.1, "t_final": 1, "n_frames": 3, "grid": 21},
    "rates": {"kappa_1": 0.1, "alpha_sq_values": [1, 2]},
    "transmon": {"E_J": 15, "E_C": 0.3},
    "qubit": {
        "experiment": "pulse",
        "alpha_ghz": -0.2,
        "duration_ns": 6,
        "drag": 1,
        "n_points": 21,
    },
    "pulse": {"duration_ns": 4, "steps": 10},
    "qec": {"p": 0.005, "p_phase": 0.01, "p_bit": 1e-5},
    "compare": {"t_ns": 25},
    "gates": {"g_mhz": 20, "n": 2, "log_kappa_ratio": -2},
}


@pytest.mark.parametrize("view", CASES)
def test_script_runs_and_prints(view: str) -> None:
    r = client.post(f"/api/python/{view}", json=CASES[view])
    assert r.status_code == 200, r.text
    code = r.json()["python"]
    assert "pip install qubitlab" in code
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(code, f"<{view} script>", "exec"), {})
    assert out.getvalue().strip()


def test_script_inputs_are_validated() -> None:
    assert client.post("/api/python/pulse", json={"duration_ns": 1e6}).status_code == 422
    assert client.post("/api/python/nope", json={}).status_code == 422
