"""Expected values come from the hand calcs and the papers, never from running the code."""

import math

import pytest

from qubitlab.theory import FORMULAS, cat, transmon


def test_coherent_overlap() -> None:
    assert float(cat.coherent_overlap(alpha=1.0)) == pytest.approx(math.exp(-4))


def test_coherent_wigner_peak_is_two_over_pi() -> None:
    assert float(cat.wigner_coherent(x=1.5, p=0.0, alpha=1.5)) == pytest.approx(2 / math.pi)


def test_cat_parity_shows_at_the_origin() -> None:
    # W(0) = (2/pi) <P>: +2/pi for the even cat, -2/pi for the odd cat.
    assert float(cat.wigner_even_cat(x=0.0, p=0.0, alpha=2.0)) == pytest.approx(2 / math.pi)
    assert float(cat.wigner_odd_cat(x=0.0, p=0.0, alpha=2.0)) == pytest.approx(-2 / math.pi)


def test_stabilized_photon_number_matches_hand_calc() -> None:
    # docs/hand_calcs/cat_steady_state.py
    value = cat.stabilized_photon_number(alpha=2.0, kappa_1=0.1, kappa_2=1.0)
    assert float(value) == pytest.approx(3.95)


def test_bit_flip_suppression_is_e_squared_per_photon() -> None:
    assert float(cat.bit_flip_suppression(Delta_n=1.0)) == pytest.approx(math.exp(-2))


def test_transmon_numbers_from_koch() -> None:
    # E_J/E_C = 50: sqrt(8 * 50) - 1 = 19.
    assert float(transmon.qubit_frequency(E_J=50.0, E_C=1.0)) == pytest.approx(19.0)
    assert float(transmon.anharmonicity(E_C=0.3)) == pytest.approx(-0.3)
    # Eq. (2.11), m = 0: -E_J + sqrt(8 E_J E_C)/2 - E_C/4.
    assert float(transmon.level_energy(m=0.0, E_J=50.0, E_C=1.0)) == pytest.approx(-50 + 10 - 0.25)


def test_charge_dispersion_sign_alternates() -> None:
    e0 = float(transmon.charge_dispersion(m=0.0, E_J=50.0, E_C=1.0))
    e1 = float(transmon.charge_dispersion(m=1.0, E_J=50.0, E_C=1.0))
    assert e0 > 0 > e1


def test_latex_is_generated_for_every_formula() -> None:
    for formula in FORMULAS.values():
        assert "=" in formula.latex
        assert formula.describe()["sources"]


def test_unknown_argument_is_rejected() -> None:
    with pytest.raises(TypeError):
        cat.coherent_overlap(beta=1.0)
