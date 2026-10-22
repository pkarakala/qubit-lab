"""The playground: a typed one-mode model must match the built-in solver and simple hand calcs."""

import math

import numpy as np
import pytest
from fastapi.testclient import TestClient

from qubitlab.api import app
from qubitlab.sim import cat as cat_sim
from qubitlab.sim.custom import Jump, Model, ModelError, build, evolve, python_script

client = TestClient(app)


def test_two_photon_preset_matches_the_cat_solver() -> None:
    p = cat_sim.CatParams(alpha_sq=4.0, kappa_1=0.1)
    ref = cat_sim.evolve(p, "vacuum", t_final=1.0, n_frames=5, grid=21, extent=4.0)
    m = Model(
        "0",
        (Jump("a**2 - alpha2", "1"), Jump("a", "kappa_1")),
        {"alpha2": 4.0, "kappa_1": 0.1},
        dim=p.n_fock,
    )
    ev = evolve(m, "vacuum", 4.0, t_final=1.0, n_frames=5, grid=21, extent=4.0)
    np.testing.assert_allclose(ev.photon_number, ref.photon_number, atol=1e-10)
    np.testing.assert_allclose(ev.wigner, ref.wigner, atol=1e-10)


def test_photon_loss_from_a_coherent_state() -> None:
    # Hand calc: under D[sqrt(kappa) a], <n>(t) = |alpha|^2 exp(-kappa t). Here 4 e^-1 = 1.4715.
    m = Model("0", (Jump("a", "kappa"),), {"kappa": 1.0}, dim=30)
    ev = evolve(m, "coherent_plus", 4.0, t_final=1.0, n_frames=2, grid=21, extent=4.0)
    assert ev.photon_number[-1] == pytest.approx(4 * math.exp(-1), rel=1e-6)


def test_driven_damped_oscillator_steady_state() -> None:
    # Hand calc: d<a>/dt = -i eps - (kappa/2)<a> = 0 gives |<a>|^2 = (2 eps / kappa)^2 = 1.
    m = Model("eps*(a + ad)", (Jump("a", "kappa"),), {"eps": 0.5, "kappa": 1.0}, dim=20)
    ev = evolve(m, "vacuum", 0.0, t_final=30.0, n_frames=2, grid=21, extent=4.0)
    assert ev.photon_number[-1] == pytest.approx(1.0, rel=1e-4)


def test_operator_algebra() -> None:
    b = build(Model("ad*a", (), {}, dim=8))
    np.testing.assert_allclose(np.asarray(b.H.to_jax()), np.diag(np.arange(8.0)), atol=1e-12)


@pytest.mark.parametrize(
    "text",
    [
        "__import__('os')",
        "a.dag()",
        "open('x')",
        "[a, a]",
        "a if 1 else a",
        "lambda: a",
        "a**k",
        "a**99",
        "1e300**8 * a",
        "ad**2 - a**2",  # not Hermitian
        "x * a",  # unknown name
    ],
)
def test_parser_rejects(text: str) -> None:
    with pytest.raises(ModelError):
        build(Model(text, (), {"k": 2.0}, dim=8))


def test_reserved_parameter_names_are_rejected() -> None:
    with pytest.raises(ModelError):
        Model("0", (), {"a": 1.0})


def test_python_script_reproduces_the_run() -> None:
    m = Model(
        "-K*ad**2*a**2 + eps2*(ad**2 + a**2)",
        (Jump("a", "kappa"),),
        {"K": 1.0, "eps2": 2.0, "kappa": 0.1},
        20,
    )
    ev = evolve(m, "vacuum", 2.0, t_final=1.0, n_frames=3, grid=21, extent=3.0)
    scope: dict[str, object] = {}
    exec(python_script(m, "vacuum", 2.0, 1.0, 3), scope)  # our own generated code
    result = scope["result"]
    assert float(result.expects[0, -1].real) == pytest.approx(ev.photon_number[-1], rel=1e-8)  # type: ignore[attr-defined]


def test_api_playground() -> None:
    body = {
        "hamiltonian": "0",
        "jumps": [{"op": "a", "rate": "1"}],
        "initial": "coherent_plus",
        "alpha_sq": 4.0,
        "t_final": 1.0,
        "n_frames": 2,
        "grid": 21,
        "dim": 30,
    }
    data = client.post("/api/playground", json=body).json()
    assert data["sim"]["photon_number"][-1] == pytest.approx(4 * math.exp(-1), rel=1e-4)
    assert "dq.mesolve" in data["python"]
    bad = {**body, "hamiltonian": "__import__('os')"}
    assert client.post("/api/playground", json=bad).status_code == 422


def test_api_reports_a_stiff_model_cleanly() -> None:
    body = {
        "hamiltonian": "K*ad**2*a**2",
        "params": {"K": 1e7},
        "initial": "coherent_plus",
        "t_final": 100.0,
        "n_frames": 2,
        "grid": 21,
    }
    r = client.post("/api/playground", json=body)
    assert r.status_code == 422
    assert "solver gave up" in r.json()["detail"]
