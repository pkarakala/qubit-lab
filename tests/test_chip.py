"""Chip layer: the closed forms against independent numbers, and the Palace data file."""

import numpy as np
import pytest
from scipy.special import ellipk

from qubitlab.sim import chip as sim
from qubitlab.theory import chip as th

K = th.CONSTANTS


def test_cpw_capacitance_matches_scipy_elliptic_k() -> None:
    # JAX has no ellipk; the registry computes it with the AGM. scipy's ellipk takes m = k^2.
    w, s, eps_eff = 10e-6, 6e-6, (1 + 11.45) / 2
    k2 = (w / (w + 2 * s)) ** 2
    expected = 4 * K["epsilon_0"] * eps_eff * ellipk(k2) / ellipk(1 - k2)
    got = th.cpw_capacitance_per_length(epsilon_0=K["epsilon_0"], epsilon_eff=eps_eff, w=w, s=s)
    assert float(got) == pytest.approx(expected, rel=1e-12)
    assert expected == pytest.approx(163.5e-12, rel=1e-3)  # docs/hand_calcs/chip.py


def test_half_wave_resonator_hand_numbers() -> None:
    # 8 mm on silicon: c / (2 l sqrt(6.225)) = 7.51 GHz.
    f0 = th.half_wave_frequency(c=K["c"], epsilon_eff=6.225, l=8e-3)
    assert float(f0) == pytest.approx(7.513e9, rel=1e-3)


def test_energies_and_harmonic_frequency_are_consistent() -> None:
    # sqrt(8 E_J E_C) = 1 / (2 pi sqrt(L_J C)) exactly: koch2007 Eq. (2.11).
    L_J, C = 12e-9, 90e-15
    E_C = float(th.charging_energy(e=K["e"], C_Sigma=C, h=K["h"]))
    E_J = float(th.josephson_energy(h=K["h"], e=K["e"], L_J=L_J))
    f_p = float(th.harmonic_frequency(L_J=L_J, C_Sigma=C))
    assert E_C == pytest.approx(0.2152e9, rel=1e-3)
    assert E_J == pytest.approx(13.62e9, rel=1e-3)
    assert np.sqrt(8 * E_J * E_C) == pytest.approx(f_p, rel=1e-12)


def test_coupling_scales_with_the_capacitance_ratio() -> None:
    g = th.coupling_g(C_g=5e-15, f_q=6e9, f_r=6e9, C_Sigma=90e-15, C_r=650e-15)
    # (C_g/2) f / sqrt(C_Sigma C_r) = 2.5 * 6e9 / sqrt(90 * 650) = 62 MHz.
    assert float(g) == pytest.approx(62.0e6, rel=1e-2)


def test_palace_data_is_self_consistent() -> None:
    caps = sim.capacitances()
    assert 0 < caps.C_g < caps.C_sigma
    m = np.array(sim.load()["device"]["maxwell_F"])
    assert np.allclose(m, m.T, rtol=1e-3, atol=1e-18)  # the Maxwell matrix is symmetric
    L_J, f_q, f_r = sim.gallery_modes()
    assert f_q < f_r  # the gallery junction keeps the qubit below the resonator
    cross = sim.crossing()
    assert cross.f_lower.min() < cross.f_r < cross.f_upper.max()
    assert sim.provenance()["version"] != "unknown"


def test_gallery_scene_uses_the_palace_capacitance() -> None:
    from qubitlab.architectures import scenes

    p = scenes()["simulated_transmon"].physics
    C = sim.capacitances().C_sigma
    E_C = float(th.charging_energy(e=K["e"], C_Sigma=C, h=K["h"])) / 1e9
    E_J = float(th.josephson_energy(h=K["h"], e=K["e"], L_J=sim.gallery_modes()[0])) / 1e9
    assert p["E_C"] == pytest.approx(E_C, rel=2e-3)
    assert p["E_J"] == pytest.approx(E_J, rel=2e-3)
