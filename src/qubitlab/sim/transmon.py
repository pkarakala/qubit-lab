"""Exact transmon spectrum, from scqubits.

scqubits diagonalizes H = 4 E_C (n - n_g)^2 - E_J cos(phi) in the charge basis, truncated at
|n| <= n_cut. That is exact to machine precision once n_cut is a few times (E_J / E_C)^(1/4).
This module only wraps it in the shapes the checks and the explorer use.

Nothing here uses a closed-form result. `qubitlab.checks` compares these numbers to Koch et al.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scqubits
from numpy.typing import NDArray

scqubits.settings.PROGRESSBAR_DISABLED = True


@dataclass(frozen=True)
class TransmonParams:
    E_J: float = 15.0  # Josephson energy, GHz
    E_C: float = 0.3  # charging energy, GHz
    n_cut: int = 25

    def __post_init__(self) -> None:
        if self.E_J <= 0 or self.E_C <= 0:
            raise ValueError("E_J and E_C must be positive")

    def device(self, n_g: float = 0.0, n_levels: int = 4) -> scqubits.Transmon:
        return scqubits.Transmon(
            EJ=self.E_J, EC=self.E_C, ng=n_g, ncut=self.n_cut, truncated_dim=n_levels
        )


def levels(params: TransmonParams, n_g: float = 0.0, n_levels: int = 4) -> NDArray[np.float64]:
    """Energies E_0..E_{n_levels-1} at offset charge n_g, in the units of E_J and E_C."""
    return np.asarray(params.device(n_g, n_levels).eigenvals(evals_count=n_levels))


@dataclass(frozen=True)
class Bands:
    n_g: NDArray[np.float64]
    energies: NDArray[np.float64]  # [i_ng, level], relative to the minimum of E_0

    def transition(self, lo: int, hi: int) -> NDArray[np.float64]:
        return self.energies[:, hi] - self.energies[:, lo]

    def dispersion(self, level: int) -> float:
        """Peak-to-peak variation of E_level over n_g."""
        e = self.energies[:, level]
        return float(e.max() - e.min())


def bands(params: TransmonParams, n_points: int = 101, n_levels: int = 4) -> Bands:
    """Energy bands over one period of offset charge, n_g in [-1, 1]."""
    ng = np.linspace(-1.0, 1.0, n_points)
    spectrum = params.device(0.0, n_levels).get_spectrum_vs_paramvals(
        "ng", ng, evals_count=n_levels
    )
    e = np.asarray(spectrum.energy_table)
    return Bands(n_g=ng, energies=e - e[:, 0].min())
