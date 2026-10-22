import math

import numpy as np
import pytest
import scqubits

from qubitlab.sim import transmon
from qubitlab.sim import transmon_dynamics as dyn


def test_charge_basis_solver_matches_scqubits() -> None:
    """Independent cross-check: scqubits diagonalizes the same Hamiltonian its own way."""
    ours = transmon.levels(
        transmon.TransmonParams(E_J=50.0, E_C=1.0, n_cut=30), n_g=0.0, n_levels=3
    )
    ref = scqubits.Transmon(EJ=50.0, EC=1.0, ng=0.0, ncut=30, truncated_dim=3).eigenvals(3)
    np.testing.assert_allclose(np.diff(ours), np.diff(ref), rtol=1e-9)


def test_t2_cannot_exceed_twice_t1() -> None:
    with pytest.raises(ValueError):
        dyn.TransmonQubit(t1_ns=10.0, t2_ns=25.0)


def test_pi_pulse_area() -> None:
    t = np.linspace(0, 20.0, 401)
    ox, oy = dyn.gaussian_pulse(t, 20.0, math.pi, drag=0.0, alpha_ghz=-0.3)
    assert 2 * math.pi * np.trapezoid(ox, t) == pytest.approx(math.pi, rel=1e-2)
    assert not oy.any()


def test_drag_cuts_leakage() -> None:
    q = dyn.TransmonQubit(alpha_ghz=-0.2, t1_ns=1e9, t2_ns=1e9)
    plain = dyn.run("pulse", q, duration_ns=8.0, drag=0.0).leakage[-1]
    drag = dyn.run("pulse", q, duration_ns=8.0, drag=1.0).leakage[-1]
    assert drag < plain / 10


def test_populations_sum_to_one() -> None:
    traj = dyn.run("rabi", dyn.TransmonQubit(t1_ns=50.0, t2_ns=60.0), t_max_ns=50.0, n_points=11)
    np.testing.assert_allclose(traj.populations.sum(axis=1), 1.0, atol=1e-9)
    assert np.all(np.linalg.norm(traj.bloch, axis=1) <= 1 + 1e-9)
