import numpy as np
import pytest

from qubitlab.sim import cat, transmon


def test_evolve_shapes_and_trace() -> None:
    ev = cat.evolve(cat.CatParams(alpha_sq=2.0, kappa_1=0.05), t_final=1.0, n_frames=5, grid=21)
    assert ev.wigner.shape == (5, 21, 21)
    # The Wigner function integrates to 1.
    dx = ev.xvec[1] - ev.xvec[0]
    assert ev.wigner[-1].sum() * dx * dx == pytest.approx(1.0, abs=1e-3)
    assert ev.purity[0] == pytest.approx(1.0)


def test_wigner_axes_are_x_then_p() -> None:
    ev = cat.evolve(cat.CatParams(alpha_sq=4.0), "coherent_plus", t_final=0.0, n_frames=1, grid=41)
    i, j = np.unravel_index(np.argmax(ev.wigner[0]), ev.wigner[0].shape)
    assert ev.xvec[i] == pytest.approx(2.0, abs=0.2)
    assert ev.xvec[j] == pytest.approx(0.0, abs=0.2)


@pytest.mark.slow
def test_bit_flip_rate_is_converged_in_fock_space() -> None:
    p = cat.CatParams(alpha_sq=6.0, kappa_1=0.1)
    bigger = cat.CatParams(alpha_sq=6.0, kappa_1=0.1, dim=p.n_fock + 10)
    a, b = cat.liouvillian_rates(p).z_decay, cat.liouvillian_rates(bigger).z_decay
    assert a == pytest.approx(b, rel=0.02)


def test_negative_rates_are_rejected() -> None:
    with pytest.raises(ValueError):
        cat.CatParams(kappa_1=-1.0)


def test_transmon_bands_are_periodic_in_offset_charge() -> None:
    b = transmon.bands(transmon.TransmonParams(E_J=10.0, E_C=1.0), n_points=41)
    assert np.allclose(b.energies[0], b.energies[-1])
    assert b.dispersion(1) > b.dispersion(0)
