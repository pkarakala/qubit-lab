"""Heterodyne readout records. Expected values are from docs/hand_calcs/readout.py, step 4-5."""

import math
from dataclasses import replace

import numpy as np
import pytest

from qubitlab.sim import readout as sim


@pytest.mark.parametrize("eta", [1.0, 0.5])
def test_noise_is_unit_white_noise_in_each_quadrature(eta: float) -> None:
    # No drive: the box-car signal is pure noise of variance tau per quadrature, for any eta.
    r = sim.Readout(chi=0.5, kappa=1.0, eps=0.0, tau=5.0, eta=eta, photons=3, dt=0.01)
    s = sim.records(r, excited=False, shots=2000)[:, 0]
    assert s.real.var() == pytest.approx(5.0, rel=0.1)
    assert s.imag.var() == pytest.approx(5.0, rel=0.1)
    assert abs(s.mean()) < 0.2  # 3 sigma of the sample mean is 3 sqrt(5 / 2000) = 0.15


@pytest.mark.parametrize(
    ("delta_r", "expected"),
    [
        (0.0, math.sqrt(2) * (0.5 - 0.5j)),  # alpha_g = -i eps / (kappa/2 - i chi)
        (0.5, math.sqrt(2) * -1j),  # drive on the |g> resonance: alpha_g = -2i eps / kappa
    ],
)
def test_late_record_sits_at_the_steady_state_field(delta_r: float, expected: complex) -> None:
    # Moving the drive (a resonator that drifted by delta_r) moves the cloud as step 4 says:
    # per unit time the record is sqrt(2 eta kappa) alpha_g.
    r = sim.Readout(chi=0.5, kappa=1.0, eps=0.5, tau=20.0, delta_r=delta_r, photons=10, dt=0.01)
    m = sim.records(r, excited=False, shots=500, bins=2)
    late = m[:, 1].mean() / 10.0  # t from 10 to 20 / kappa, long after the ring-up
    assert abs(late - expected) < 0.05  # sampling error 0.014 per quadrature


def test_t1_decay_pulls_excited_shots_toward_ground() -> None:
    # A shot that decays at t_d looks like |e> for min(t_d, tau) of the window. Averaged over
    # exponential t_d with T1 = tau, that is (1 - e^-1) tau, so the |e> cloud's mean moves a
    # fraction e^-1 = 0.37 of the way from |e> to |g>. The resonator's lag after a decay
    # (~1.4 / kappa) shifts this by about 1.4 / 40, under 0.02.
    r = sim.Readout(chi=0.5, kappa=1.0, eps=0.5, tau=40.0, photons=8)
    g = sim.records(r, excited=False, shots=300)[:, 0].mean()
    e = sim.records(r, excited=True, shots=300)[:, 0].mean()
    e1 = sim.records(replace(r, t1=40.0), excited=True, shots=300)[:, 0].mean()
    assert abs(e1 - e) / abs(g - e) == pytest.approx(math.exp(-1), abs=0.05)


def test_bad_parameters_raise() -> None:
    with pytest.raises(ValueError):
        sim.Readout(chi=0.5, kappa=1.0, eps=0.1, tau=1.0, eta=0.0)
    with pytest.raises(ValueError):
        sim.Readout(chi=0.5, kappa=0.0, eps=0.1, tau=1.0)


def test_fidelity_of_separated_and_identical_clouds() -> None:
    rng = np.random.default_rng(0)
    g = rng.normal(size=5000) + 1j * rng.normal(size=5000)
    assert sim.assignment_fidelity(g, g + 20.0) == pytest.approx(1.0)
    assert abs(sim.assignment_fidelity(g[:2500], g[2500:])) < 0.06
