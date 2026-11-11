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


@pytest.fixture(scope="module")
def nominal() -> po.Optimized:
    return po.optimize(Q, 10.0, slices=40)


def test_zero_detuning_is_the_old_model() -> None:
    edges, ox, oy = po.drag_slices(Q, 6.0, 24)
    assert po.gate_error(Q, edges, ox, oy, 0.0) == po.gate_error(Q, edges, ox, oy)


def _even(errors: list[float]) -> float:
    # Mean over +x and -x drops the cross term with the pulse's own error, which is odd in x
    # (docs/hand_calcs/qubit_decoherence.py, step 8).
    return sum(errors) / 2


def test_mismatch_error_grows_quadratically(nominal: po.Optimized) -> None:
    # The pulse is at F ~ 1, a maximum, so 1 - F is quadratic in any small model error:
    # doubling the mismatch multiplies the even part by 4. Step 8 gives the size for detuning,
    # (2/3)(pi Delta tau)^2 with tau of a few ns here, so at least 1e-5 at 1 MHz.
    # Finding: with alpha off by 10 and 20 MHz (5% and 10%) the ratio came out 4.24, so the
    # quartic term is ~6% there. It scales as the mismatch squared, so at 5 and 10 MHz it
    # should be ~1.5%, inside the 5% band.
    r = nominal

    def at(alpha: float, delta: float) -> float:
        q = qb.TransmonQubit(alpha_ghz=alpha)
        return po.gate_error(q, r.edges_ns, r.omega_x, r.omega_y, delta)[0]

    d1 = _even([at(-0.2, 0.001), at(-0.2, -0.001)])
    d2 = _even([at(-0.2, 0.002), at(-0.2, -0.002)])
    a1 = _even([at(-0.195, 0.0), at(-0.205, 0.0)])
    a2 = _even([at(-0.19, 0.0), at(-0.21, 0.0)])
    assert d1 > 1e-5 > r.error * 10
    assert a1 > r.error * 10
    assert d2 / d1 == pytest.approx(4.0, rel=0.05)
    assert a2 / a1 == pytest.approx(4.0, rel=0.05)


def test_ensemble_of_one_is_the_nominal_optimizer(nominal: po.Optimized) -> None:
    r = po.optimize(Q, 10.0, slices=40, ensemble=[(-0.2, 0.0)])
    np.testing.assert_array_equal(r.history, nominal.history)


@pytest.mark.slow
def test_ensemble_pulse_has_lower_worst_case_error() -> None:
    # kobzar2004 Eq. (1): average the cost over the models. At +-2 MHz the nominal pulse loses
    # about (2/3)(pi 0.002 tau)^2 (step 8), ~2e-3 at 20 ns with tau ~ 8 ns. With 300 MHz of
    # drive the pulse has room to cancel the first-order detuning term, so we expect the
    # worst case to drop by at least 10x.
    # Finding: first written at 10 ns. There the ensemble cost plateaus near 1.5e-4 (also
    # after 1500 steps) and the worst case drops only from 6.0e-4 to 4.5e-4. At 20 ns it
    # keeps falling.
    models = [(-0.2, -0.002), (-0.2, 0.0), (-0.2, 0.002)]
    nominal = po.optimize(Q, 20.0, slices=80)
    robust = po.optimize(Q, 20.0, slices=80, ensemble=models)

    def worst(r: po.Optimized) -> float:
        return max(
            po.gate_error(qb.TransmonQubit(alpha_ghz=a), r.edges_ns, r.omega_x, r.omega_y, d)[0]
            for a, d in models
        )

    assert worst(robust) < worst(nominal) / 10
