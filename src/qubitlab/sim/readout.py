"""Dispersive readout, simulated without the dispersive approximation where it matters.

- `dressed_chi`: the transmon's levels and charge matrix elements come from scqubits; the
  transmon-resonator Hamiltonian H = H_q + f_r a+a + g_n n (a + a+) is then diagonalized with
  NumPy, and chi is read from the dressed energies. (scqubits' own HilbertSpace would do this,
  but its releases that install beside our SciPy still use np.float_, removed in NumPy 2.)
- `dephasing_rate`: Dynamiqs solves a qubit coupled to a driven, leaky resonator through
  chi sigma_z a+a, and the qubit coherence decay rate is read from the steady state.
- `pointer_trajectories`: the resonator field for the qubit in |g> and in |e>, over time.

Nothing here uses a closed-form result. `qubitlab.checks` compares these numbers to the theory.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import dynamiqs as dq
import jax
import jax.numpy as jnp
import numpy as np
import scqubits
from numpy.typing import NDArray

scqubits.settings.PROGRESSBAR_DISABLED = True
dq.set_precision("double")
dq.set_progress_meter(False)


@dataclass(frozen=True)
class Dressed:
    chi: float  # half the difference of the resonator frequency with the qubit in |1> vs |0>
    f01: float  # bare qubit frequency
    alpha: float  # bare anharmonicity
    g01: float  # g_n |<0|n|1>|, the coupling the formulas use
    g12: float  # g_n |<1|n|2>|
    shift_g: float  # resonator frequency with the qubit in |0>, minus the bare one


def dressed_chi(
    E_J: float, E_C: float, f_r: float, g_n: float, levels: int = 6, photons: int = 8
) -> Dressed:
    """Exact dispersive shift from the dressed spectrum. Energies in GHz."""
    t = scqubits.Transmon(EJ=E_J, EC=E_C, ng=0.0, ncut=31, truncated_dim=levels)
    evals, evecs = t.eigensys(evals_count=levels)
    e = evals - evals[0]
    n_op = np.asarray(t.matrixelement_table("n_operator", evecs=evecs, evals_count=levels))
    a = np.diag(np.sqrt(np.arange(1, photons)), 1)
    eye_q, eye_r = np.eye(levels), np.eye(photons)
    H = np.kron(np.diag(e), eye_r) + f_r * np.kron(eye_q, a.T @ a) + g_n * np.kron(n_op, a + a.T)
    w, v = np.linalg.eigh(H)

    def energy(q: int, n: int) -> float:
        """Dressed energy of the eigenstate with the largest overlap on bare |q, n>."""
        return float(w[np.argmax(np.abs(v[q * photons + n, :]) ** 2)])

    f_g = energy(0, 1) - energy(0, 0)
    f_e = energy(1, 1) - energy(1, 0)
    return Dressed(
        chi=(f_e - f_g) / 2,
        f01=float(e[1]),
        alpha=float(e[2] - 2 * e[1]),
        g01=g_n * abs(float(n_op[0, 1])),
        g12=g_n * abs(float(n_op[1, 2])),
        shift_g=f_g - f_r,
    )


def _dispersive(chi: float, eps: float, kappa: float, photons: int) -> tuple[dq.QArray, dq.QArray]:
    """chi sigma_z a+a + eps (a + a+), in the frame of the bare resonator. Qubit first."""
    a = dq.tensor(dq.eye(2), dq.destroy(photons))
    sz = dq.tensor(dq.sigmaz(), dq.eye(photons))
    H = chi * sz @ a.dag() @ a + eps * (a + a.dag())
    return H, math.sqrt(kappa) * a


def dephasing_rate(chi: float, eps: float, kappa: float, photons: int = 30) -> float:
    """Decay rate of the qubit coherence |<sigma_->| once the resonator has settled."""
    H, L = _dispersive(chi, eps, kappa, photons)
    plus = dq.unit(dq.basis(2, 0) + dq.basis(2, 1))
    psi0 = dq.tensor(plus, dq.basis(photons, 0))
    settle = 20 / kappa  # field transients decay at kappa / 2
    t = jnp.asarray([0.0, settle, settle + 2 / kappa])
    sm = dq.tensor(dq.sigmam(), dq.eye(photons))
    res = dq.mesolve(H, [L], psi0, t, exp_ops=[sm])
    c = np.abs(np.asarray(res.expects[0]))
    return float(math.log(c[1] / c[2]) / (2 / kappa))


@dataclass(frozen=True)
class Pointers:
    times: NDArray[np.float64]
    field_g: NDArray[np.complex128]  # <a>(t), qubit in |g>
    field_e: NDArray[np.complex128]  # <a>(t), qubit in |e>
    photons_g: NDArray[np.float64]
    photons_e: NDArray[np.float64]


def pointer_trajectories(
    chi: float,
    eps: float,
    kappa: float,
    delta_r: float,
    t_max: float,
    n_points: int = 81,
    photons: int = 30,
) -> Pointers:
    """The resonator field over time for each qubit state. Drive at omega_r - delta_r."""
    a = dq.destroy(photons)
    t = jnp.linspace(0.0, t_max, n_points)
    out: dict[int, tuple[NDArray[np.complex128], NDArray[np.float64]]] = {}
    for sign in (1, -1):  # +chi: |e>, -chi: |g>
        H = (delta_r + sign * chi) * a.dag() @ a + eps * (a + a.dag())
        res = dq.mesolve(
            H, [math.sqrt(kappa) * a], dq.basis(photons, 0), t, exp_ops=[a, a.dag() @ a]
        )
        e = np.asarray(res.expects)
        out[sign] = (e[0], e[1].real)
    return Pointers(np.asarray(t), out[-1][0], out[1][0], out[-1][1], out[1][1])


@dataclass(frozen=True)
class Readout:
    """A dispersive readout, in any one frequency unit, with times in its inverse.

    Drive at omega_r - delta_r with amplitude eps; the resonator sits at +chi with the qubit in
    |e> and -chi in |g>. The output is split for heterodyne detection with efficiency eta.
    """

    chi: float
    kappa: float
    eps: float
    tau: float  # integration time
    delta_r: float = 0.0
    eta: float = 1.0
    t1: float = math.inf
    photons: int = 10
    dt: float = 0.01  # solver step, in 1 / kappa units of time; tau / bins a multiple of it

    def __post_init__(self) -> None:
        if not 0 < self.eta <= 1:
            raise ValueError("eta must be in (0, 1]")
        if self.kappa <= 0 or self.tau <= 0 or self.t1 <= 0:
            raise ValueError("kappa, tau and T1 must be positive")


def records(
    r: Readout, excited: bool, shots: int, seed: int = 0, bins: int = 1
) -> NDArray[np.complex128]:
    """Heterodyne records I + iQ, one row per shot and one column per time bin.

    Each entry is the record integrated over its bin: the signal sqrt(2 eta kappa) <a> dt plus
    unit-variance white noise in each quadrature (the dY of `dq.dssesolve`). Summing a row
    gives the box-car signal. The qubit starts in |e> or |g>, with the resonator empty.

    Dynamiqs integrates the stochastic Schrodinger equation, with the qubit's T1 as a third
    channel whose record is dropped. The resonator records do not depend on how that channel is
    unravelled, so they have the statistics of the master equation. `measurements_sse_to_sme`
    then adds the noise of an efficiency eta < 1.

    The solver is Rouchon1 (Euler-Maruyama let about 1 trajectory in 400 blow up to NaN at
    tau = 40 / kappa). It is first order in dt: the mean field comes out about 1% low at the
    default dt = 0.01 / kappa and 5% low at 0.05 / kappa (tests/test_readout_records.py).
    """
    a = dq.tensor(dq.eye(2), dq.destroy(r.photons))
    sz = dq.tensor(dq.sigmaz(), dq.eye(r.photons))  # +1 is |e> = basis 0 (dq.sigmam lowers it)
    H = r.delta_r * a.dag() @ a + r.chi * sz @ a.dag() @ a + r.eps * (a + a.dag())
    half = math.sqrt(r.kappa / 2)
    jumps = [half * a, -1j * half * a]
    if math.isfinite(r.t1):
        jumps.append(math.sqrt(1 / r.t1) * dq.tensor(dq.sigmam(), dq.eye(r.photons)))
    psi0 = dq.tensor(dq.basis(2, 0 if excited else 1), dq.basis(r.photons, 0))
    tsave = jnp.linspace(0.0, r.tau, bins + 1)
    keys = jax.random.split(jax.random.key(seed), shots + 1)
    res = dq.dssesolve(
        H, jumps, psi0, tsave, keys[:-1], method=dq.method.Rouchon1(dt=r.dt), save_states=False
    )
    m = res.measurements[:, :2]  # (shots, 2, bins): bin averages of dY / dt
    if r.eta < 1:
        m = dq.measurements_sse_to_sme(m, tsave, jnp.asarray([r.eta, r.eta]), keys[-1])
    m = np.asarray(m) * (r.tau / bins)
    if not np.isfinite(m).all():
        raise FloatingPointError("a trajectory diverged; lower dt or raise photons")
    return m[:, 0] + 1j * m[:, 1]


def assignment_fidelity(g: NDArray[np.complex128], e: NDArray[np.complex128]) -> float:
    """1 - P(e|g) - P(g|e) for box-car signals (rows summed), split halfway between the means.

    The split is a line through the midpoint, normal to the line joining the two means: the
    best linear split for two Gaussian clouds of equal, round width. Written here because
    NumPy and SciPy have no two-class discriminator; it is three lines.
    """
    s_g, s_e = g.sum(axis=-1) if g.ndim > 1 else g, e.sum(axis=-1) if e.ndim > 1 else e
    mid, axis = (s_g.mean() + s_e.mean()) / 2, s_e.mean() - s_g.mean()
    p_e = np.mean(np.real((s_e - mid) * np.conj(axis)) > 0)  # P(e|e)
    p_g = np.mean(np.real((s_g - mid) * np.conj(axis)) > 0)  # P(e|g)
    return float(p_e - p_g)
