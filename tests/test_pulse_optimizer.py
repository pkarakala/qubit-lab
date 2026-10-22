import jax.numpy as jnp
import numpy as np
import pytest

from qubitlab.sim import pulse_optimizer as po
from qubitlab.sim import transmon_dynamics as qb

Q = qb.TransmonQubit(alpha_ghz=-0.2)


def test_subspace_fidelity_limits() -> None:
    x = jnp.asarray(po.X_GATE)
    assert float(po.subspace_fidelity(x.conj().T @ x)) == pytest.approx(1.0)
    # Doing nothing when X was asked for: Tr(M M+) = 2, Tr M = 0, so F = 2/6.
    assert float(po.subspace_fidelity(x.conj().T @ jnp.eye(2))) == pytest.approx(1 / 3)


def test_pedersen_matches_six_state_simulation() -> None:
    # Eq. (3) and the 2-design average are two routes to the same number in a closed system.
    edges, ox, oy = po.drag_slices(Q, 6.0, 24)
    error, _ = po.gate_error(Q, edges, ox, oy)
    closed = qb.TransmonQubit(alpha_ghz=-0.2, t1_ns=1e12, t2_ns=1e12)
    H = qb.driven_h(closed, 0.0, edges, ox, oy)
    f = qb.average_gate_fidelity(closed, H, 6.0, po.X_GATE)
    assert 1 - f == pytest.approx(error, rel=1e-4)


def test_optimizer_beats_drag_within_its_amplitude_limit() -> None:
    edges, ox, oy = po.drag_slices(Q, 10.0, 40)
    drag_error, _ = po.gate_error(Q, edges, ox, oy)
    r = po.optimize(Q, 10.0, slices=40, omega_max_ghz=0.3)
    assert r.error < 1e-6 < drag_error / 1000
    assert max(np.abs(r.omega_x).max(), np.abs(r.omega_y).max()) <= 0.3
    assert r.history[0] == pytest.approx(drag_error, rel=1e-6)


def test_run_pulse_ends_in_one() -> None:
    r = po.optimize(Q, 10.0, slices=40)
    traj = qb.run_pulse(Q, r.edges_ns, r.omega_x, r.omega_y)
    assert traj.populations[-1, 1] == pytest.approx(1.0, abs=1e-3)  # T1 = 50 us costs ~1e-4
