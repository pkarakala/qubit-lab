import pytest
from fastapi.testclient import TestClient

from catlab.api import app

client = TestClient(app)


def test_formulas_carry_latex_and_sources() -> None:
    data = client.get("/api/formulas").json()
    assert data and all(f["latex"] and f["sources"] for f in data)


def test_evolve_keeps_theory_and_sim_apart() -> None:
    body = {"alpha_sq": 2.0, "kappa_1": 0.05, "t_final": 1.0, "n_frames": 3, "grid": 21}
    data = client.post("/api/cat/evolve", json=body).json()
    assert set(data) == {"sim", "theory"}
    assert len(data["sim"]["wigner"]) == 3
    assert len(data["theory"]["even_cat_wigner"]) == 21
    assert data["theory"]["parity_decay_rate"] == pytest.approx(2 * 0.05 * 2.0)


def test_rates_refuse_unresolved_sizes() -> None:
    assert client.post("/api/cat/rates", json={"alpha_sq_values": [9.0]}).status_code == 422


def test_transmon_bands() -> None:
    data = client.get("/api/transmon/bands", params={"E_J": 50, "E_C": 1, "n_points": 21}).json()
    assert data["theory"]["e01"] == pytest.approx(19.0)
    assert abs(data["sim"]["e01_mean"] - 19.0) / 19.0 < 0.01


def test_bad_input_is_rejected() -> None:
    assert client.post("/api/cat/evolve", json={"alpha_sq": -1}).status_code == 422
