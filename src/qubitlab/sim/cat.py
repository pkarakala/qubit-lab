"""Dynamiqs simulation of one stabilized cat qubit.

Two stabilization schemes, same noise:
- "dissipative": kappa_2 D[a^2 - alpha^2]   (two-photon dissipation), rates in units of kappa_2.
- "kerr":        H = -K a+^2 a^2 + K alpha^2 (a+^2 + a^2), rates in units of K.
- "none":        no stabilization, for comparison. Rates in the same units as above.

Noise on top: single-photon loss kappa_1 with thermal occupation n_th, and pure dephasing
kappa_phi. Conventions match `qubitlab.theory.cat`.

Nothing here uses a closed-form result. `qubitlab.checks` compares these numbers to the theory.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Literal

import dynamiqs as dq
import jax.numpy as jnp
import numpy as np
import scipy.sparse.linalg as spla
from numpy.typing import NDArray
from scipy import sparse

dq.set_precision("double")
dq.set_progress_meter(False)

Scheme = Literal["dissipative", "kerr", "none"]
Initial = Literal["vacuum", "coherent_plus", "coherent_minus", "even_cat", "odd_cat"]

# Dynamiqs' Tsit5 stops after 100,000 steps by default. The longest solve the explorer can ask
# for takes ten times that (measured: a Kerr cat at |alpha|^2 = 9 for K t = 200 without noise,
# 1.03e6 steps; noise lowers the count), so `evolve` allows twice that.
MAX_STEPS = 2_000_000


@dataclass(frozen=True)
class CatParams:
    alpha_sq: float = 4.0  # target cat size |alpha|^2, mean photon number
    kappa_1: float = 0.0  # single-photon loss, in units of kappa_2 (or K)
    n_th: float = 0.0  # thermal photons in the loss bath
    kappa_phi: float = 0.0  # pure dephasing, in units of kappa_2 (or K)
    scheme: Scheme = "dissipative"
    dim: int | None = None  # Fock space truncation; None picks one from alpha_sq

    def __post_init__(self) -> None:
        if self.alpha_sq < 0 or self.kappa_1 < 0 or self.n_th < 0 or self.kappa_phi < 0:
            raise ValueError("cat parameters must be non-negative")
        if self.alpha_sq > 12:
            raise ValueError("alpha_sq above 12 needs more Fock states than this tool allows")

    @property
    def n_fock(self) -> int:
        if self.dim is not None:
            return self.dim
        n = self.alpha_sq
        # Converged to <2% in the bit-flip rate up to alpha_sq = 7
        # (tests/test_sim.py::test_bit_flip_rate_is_converged_in_fock_space).
        return int(min(80, max(20, math.ceil(n + 8 * math.sqrt(n) + 20))))


def operators(params: CatParams) -> tuple[dq.QArray, list[dq.QArray]]:
    """Hamiltonian and jump operators for `params`."""
    n = params.n_fock
    a = dq.destroy(n)
    eye = dq.eye(n)
    a2 = a @ a
    jumps: list[dq.QArray] = []
    if params.scheme == "dissipative":
        H = dq.zeros(n)
        jumps.append(a2 - params.alpha_sq * eye)
    elif params.scheme == "kerr":
        ad2 = a2.dag()
        H = -(ad2 @ a2) + params.alpha_sq * (ad2 + a2)
    else:
        H = dq.zeros(n)
    if params.kappa_1 > 0:
        jumps.append(math.sqrt(params.kappa_1 * (1 + params.n_th)) * a)
        if params.n_th > 0:
            jumps.append(math.sqrt(params.kappa_1 * params.n_th) * a.dag())
    if params.kappa_phi > 0:
        jumps.append(math.sqrt(params.kappa_phi) * dq.number(n))
    return H, jumps


def sign_x(n: int) -> dq.QArray:
    """sign of the x quadrature: +1 near +alpha, -1 near -alpha. The cat qubit's Z."""
    a = np.asarray(dq.destroy(n).to_jax())
    xq = (a + a.conj().T) / 2
    w, v = np.linalg.eigh(xq)
    return dq.asqarray(jnp.asarray(v @ np.diag(np.sign(w)) @ v.conj().T))


def initial_state(kind: Initial, alpha_sq: float, n: int) -> dq.QArray:
    alpha = math.sqrt(alpha_sq)
    if kind == "vacuum":
        return dq.vacuum(n)
    if kind == "coherent_plus":
        return dq.coherent(n, alpha)
    if kind == "coherent_minus":
        return dq.coherent(n, -alpha)
    if kind == "even_cat":
        return dq.unit(dq.coherent(n, alpha) + dq.coherent(n, -alpha))
    return dq.unit(dq.coherent(n, alpha) - dq.coherent(n, -alpha))


@dataclass(frozen=True)
class Evolution:
    times: NDArray[np.float64]
    photon_number: NDArray[np.float64]
    parity: NDArray[np.float64]  # cat qubit <X>
    sign_x: NDArray[np.float64]  # cat qubit <Z>
    purity: NDArray[np.float64]
    xvec: NDArray[np.float64]
    wigner: NDArray[np.float64]  # [frame, i_x, i_p]
    edge_population: NDArray[np.float64]  # weight in the top two Fock states: truncation warning


def evolve(
    params: CatParams,
    initial: Initial = "vacuum",
    t_final: float = 4.0,
    n_frames: int = 41,
    grid: int = 81,
    extent: float | None = None,
) -> Evolution:
    """Solve the Lindblad equation and return Wigner frames plus observables."""
    n = params.n_fock
    H, jumps = operators(params)
    rho0 = dq.todm(initial_state(initial, params.alpha_sq, n))
    ext = extent if extent is not None else max(3.0, math.sqrt(params.alpha_sq) + 2.5)
    return evolve_operators(H, jumps, rho0, t_final, n_frames, grid, ext, max_steps=MAX_STEPS)


def evolve_operators(
    H: dq.QArray,
    jumps: list[dq.QArray],
    rho0: dq.QArray,
    t_final: float,
    n_frames: int,
    grid: int,
    extent: float,
    max_steps: int = 100_000,
) -> Evolution:
    """Solve the Lindblad equation for any H and jumps on one mode: Wigner frames, observables.

    The solver raises RuntimeError after `max_steps` steps. The default is Dynamiqs' own, which
    the playground keeps so that a runaway typed model fails in seconds.
    """
    n = rho0.shape[-1]
    tsave = jnp.linspace(0.0, t_final, n_frames)
    edge = dq.proj(dq.basis(n, n - 1)) + dq.proj(dq.basis(n, n - 2))
    exp_ops = [dq.number(n), dq.parity(n), sign_x(n), edge]
    method = dq.method.Tsit5(max_steps=max_steps)
    result = dq.mesolve(H, jumps, rho0, tsave, exp_ops=exp_ops, method=method)
    states = result.states
    purity = np.asarray(dq.purity(states)).real
    xvec, _, w = dq.wigner(states, xmax=extent, ymax=extent, npixels=grid)
    expects = np.asarray(result.expects).real
    return Evolution(
        times=np.asarray(tsave),
        photon_number=expects[0],
        parity=expects[1],
        sign_x=expects[2],
        purity=purity,
        xvec=np.asarray(xvec),
        wigner=np.asarray(w),
        edge_population=expects[3],
    )


@dataclass(frozen=True)
class Rates:
    """Slowest decay rates of the two logical observables, from the Liouvillian spectrum.

    The Lindbladian commutes with parity, rho -> P rho P. Matrix elements |m><n| with m + n odd
    form one block; they hold the coherence between the even and odd cats, so the slowest mode
    there is the decay of <Z> (bit-flips). The m + n even block holds the populations of the
    two cats, so its slowest nonzero mode is the decay of <X> (phase-flips).
    """

    z_decay: float  # decay rate of <Z>, equal to 2 Gamma_X
    x_decay: float  # decay rate of <X> (parity), equal to 2 Gamma_Z
    n_fock: int

    @property
    def z_decay_resolved(self) -> bool:
        return self.z_decay > RATE_FLOOR

    @property
    def x_decay_resolved(self) -> bool:
        return self.x_decay > RATE_FLOOR

    @property
    def bit_flip_rate(self) -> float:
        return self.z_decay / 2

    @property
    def phase_flip_rate(self) -> float:
        return self.x_decay / 2


# Below this, eigenvalues of the Liouvillian are at double-precision noise (measured: the
# bit-flip decay rate stops converging near 1e-11 kappa_2). Results under it are not reported.
RATE_FLOOR = 1e-10


# Eigenvalues come from shift-invert about this point, nearest first. Every eigenvalue of a
# Lindbladian has Re <= 0, so a positive shift is never one of them. Much closer to zero, the
# steady state's eigenvalue swamps the others (measured at 1e-9: the phase-flip rate drifts by
# up to 1.6e-5 relative); at 1e-3 kappa_2 every rate matches a dense solve.
SHIFT = 1e-3


def _liouvillian(params: CatParams) -> sparse.csc_array:
    """The Lindbladian of `params` as a SciPy sparse matrix, without a dense n^2 x n^2 copy.

    Dynamiqs builds it in its sparse diagonal layout, whose storage follows SciPy's DIA
    convention: diags[k, j] = L[j - offsets[k], j].
    """
    H, jumps = operators(params)
    L = dq.slindbladian(H, jumps).assparsedia().data
    return sparse.dia_array((np.asarray(L.diags), L.offsets), shape=L.shape).tocsc()


def _near_zero(
    A: sparse.csc_array, k: int
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """The k eigenvalues of A nearest zero and their eigenvectors (ARPACK, shift-invert)."""
    v0 = np.random.default_rng(0).standard_normal(A.shape[0])  # fixed start: same answer each run
    return spla.eigs(A, k=k, sigma=SHIFT, v0=v0)  # type: ignore[no-any-return]


def liouvillian_rates(params: CatParams) -> Rates:
    n = params.n_fock
    L = _liouvillian(params)
    idx = np.arange(n)
    odd = ((idx[:, None] + idx[None, :]) % 2 == 1).reshape(-1)
    # The slowest modes are the eigenvalues nearest zero; three per block leaves a margin.
    ev_odd = np.sort(-_near_zero(L[odd][:, odd], 3)[0].real)
    ev_even = np.sort(-_near_zero(L[~odd][:, ~odd], 3)[0].real)
    # The even block holds the steady state (eigenvalue 0); skip it.
    return Rates(z_decay=float(ev_odd[0]), x_decay=float(ev_even[1]), n_fock=n)


def sweep_rates(base: CatParams, alpha_sq_values: list[float]) -> list[Rates]:
    return [liouvillian_rates(replace(base, alpha_sq=value)) for value in alpha_sq_values]


def steady_state(params: CatParams) -> dq.QArray:
    """Unique steady state, from the null vector of the Liouvillian. Needs kappa_1 > 0."""
    if params.kappa_1 <= 0:
        raise ValueError("the steady state is unique only with kappa_1 > 0")
    n = params.n_fock
    _, v = _near_zero(_liouvillian(params), 1)
    rho = np.asarray(dq.unvectorize(jnp.asarray(v)).to_jax()).reshape(n, n)
    rho = rho / np.trace(rho)
    return dq.asqarray(jnp.asarray((rho + rho.conj().T) / 2))
