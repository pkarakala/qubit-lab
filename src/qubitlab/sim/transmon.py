"""Exact transmon spectrum: diagonalize H = 4 E_C (n - n_g)^2 - E_J cos(phi) in the charge basis.

In the charge basis |n>, cos(phi) = (|n><n+1| + |n+1><n|) / 2. Truncating at |n| <= n_cut is
exact to machine precision once n_cut is a few times (E_J / E_C)^(1/4). Uses JAX so the
offset-charge sweep is one vectorized call.

Nothing here uses a closed-form result. `qubitlab.checks` compares these numbers to Koch et al.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class TransmonParams:
    E_J: float = 15.0  # Josephson energy, GHz
    E_C: float = 0.3  # charging energy, GHz
    n_cut: int = 25

    def __post_init__(self) -> None:
        if self.E_J <= 0 or self.E_C <= 0:
            raise ValueError("E_J and E_C must be positive")


def hamiltonian(params: TransmonParams, n_g: jax.Array | float) -> jax.Array:
    n = jnp.arange(-params.n_cut, params.n_cut + 1)
    diag = 4 * params.E_C * (n - n_g) ** 2
    off = -params.E_J / 2 * jnp.ones(2 * params.n_cut)
    return jnp.diag(diag) + jnp.diag(off, 1) + jnp.diag(off, -1)


def levels(params: TransmonParams, n_g: float = 0.0, n_levels: int = 4) -> NDArray[np.float64]:
    """Energies E_0..E_{n_levels-1} at offset charge n_g, in the units of E_J and E_C."""
    return np.asarray(jnp.linalg.eigvalsh(hamiltonian(params, n_g))[:n_levels])


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
    ng = jnp.linspace(-1.0, 1.0, n_points)
    eig = jax.vmap(lambda g: jnp.linalg.eigvalsh(hamiltonian(params, g))[:n_levels])(ng)
    e = np.asarray(eig)
    return Bands(n_g=np.asarray(ng), energies=e - e[:, 0].min())
