"""Energy levels of the gallery's qubit designs, from scqubits (and NumPy for the Kerr cat).

Energies in GHz (MHz for the Kerr cat), measured from the ground state. Nothing here uses a
closed-form result.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scqubits
from numpy.typing import NDArray

scqubits.settings.PROGRESSBAR_DISABLED = True


@dataclass(frozen=True)
class Spectrum:
    x: NDArray[np.float64]  # the swept parameter
    levels: NDArray[np.float64]  # [x, level], relative to the ground state at each x
    x_label: str
    unit: str


def _relative(e: NDArray[np.float64]) -> NDArray[np.float64]:
    return e - e[:, :1]


def tunable_transmon(E_J_max: float, E_C: float, d: float, n: int = 4) -> Spectrum:
    flux = np.linspace(-0.6, 0.6, 61)
    q = scqubits.TunableTransmon(EJmax=E_J_max, EC=E_C, d=d, flux=0.0, ng=0.0, ncut=31)
    e = q.get_spectrum_vs_paramvals("flux", flux, evals_count=n).energy_table
    return Spectrum(flux, _relative(np.asarray(e)), "flux through the SQUID (Phi_0)", "GHz")


def transmon(E_J: float, E_C: float, n: int = 4) -> Spectrum:
    ng = np.linspace(-1.0, 1.0, 61)
    q = scqubits.Transmon(EJ=E_J, EC=E_C, ng=0.0, ncut=31)
    e = q.get_spectrum_vs_paramvals("ng", ng, evals_count=n).energy_table
    return Spectrum(ng, _relative(np.asarray(e)), "offset charge n_g", "GHz")


def fluxonium(E_J: float, E_C: float, E_L: float, n: int = 5) -> Spectrum:
    flux = np.linspace(0.0, 1.0, 81)
    q = scqubits.Fluxonium(EJ=E_J, EC=E_C, EL=E_L, flux=0.0, cutoff=110)
    e = q.get_spectrum_vs_paramvals("flux", flux, evals_count=n).energy_table
    return Spectrum(flux, _relative(np.asarray(e)), "flux through the loop (Phi_0)", "GHz")


def kerr_cat(K: float, eps2_max: float, n: int = 6, dim: int = 40) -> Spectrum:
    """Top n levels of H = -K a+^2 a^2 + eps2 (a+^2 + a^2), measured down from the highest.

    With this sign the cat pair |C+-> sits at the top of the spectrum and becomes degenerate as
    eps2 grows (theory: cat.kerr_cat_amplitude, alpha^2 = eps2/K).
    """
    a = np.diag(np.sqrt(np.arange(1, dim)), 1)
    ad = a.T
    eps = np.linspace(0.0, eps2_max, 61)
    out = []
    for e2 in eps:
        H = -K * ad @ ad @ a @ a + e2 * (ad @ ad + a @ a)
        w = np.linalg.eigvalsh(H)[::-1][:n]
        out.append(w[0] - w)  # distance below the top level
    return Spectrum(eps, np.asarray(out), "squeezing drive eps_2 / 2 pi (MHz)", "MHz")
