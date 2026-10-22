import pytest
from fastapi.testclient import TestClient

from qubitlab.api import app

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


def test_qubit_t1_run_has_theory_curve() -> None:
    body = {"experiment": "t1", "t1_ns": 100.0, "t2_ns": 150.0, "t_max_ns": 100.0, "n_points": 11}
    data = client.post("/api/qubit/run", json=body).json()
    assert set(data) == {"sim", "theory"}
    assert data["sim"]["populations"][-1][1] == pytest.approx(0.3679, abs=1e-3)
    assert data["theory"]["curve"][-1] == pytest.approx(0.3679, abs=1e-3)


def test_qubit_rejects_t2_above_twice_t1() -> None:
    body = {"experiment": "t1", "t1_ns": 10.0, "t2_ns": 30.0}
    assert client.post("/api/qubit/run", json=body).status_code == 422


def test_cat_repetition_code_has_an_optimal_size() -> None:
    body = {"p_phase": 0.01, "p_bit": 1e-4, "shots": 20000}
    rows = client.post("/api/qec/cat", json=body).json()["rows"]
    assert [r["qubits"] for r in rows[:3]] == [1, 5, 9]
    floors = [r["theory"]["bit_flip_floor"] for r in rows]
    assert floors == sorted(floors)  # the uncorrected part only grows with d
