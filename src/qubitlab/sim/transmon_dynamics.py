"""Dynamiqs simulation of a driven, noisy transmon, kept to its lowest levels.

Model, in the frame rotating at the drive frequency (time in ns, frequencies in GHz):

    H/h = delta n + (alpha/2) a+ a+ a a + (Omega_x(t)/2)(a + a+) + (Omega_y(t)/2) i(a+ - a)

- delta: qubit frequency minus drive frequency. alpha: anharmonicity (negative, about -E_C).
- Omega_x, Omega_y: drive envelopes, as Rabi frequencies. Omega_y is the DRAG quadrature.
- Noise: relaxation sqrt(1/T1) a, pure dephasing sqrt(2/T_phi) n with 1/T_phi = 1/T2 - 1/(2 T1).
- Quasi-static frequency noise: a Gaussian spread sigma of delta, averaged over a batch of solves.

Levels above 1 are kept so leakage out of the qubit is visible. Nothing here uses a closed-form
result; `qubitlab.checks` compares these numbers to the theory.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import dynamiqs as dq
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike, NDArray

dq.set_precision("double")
dq.set_progress_meter(False)

TWO_PI = 2 * math.pi
Experiment = Literal["rabi", "pulse", "t1", "ramsey", "echo"]


@dataclass(frozen=True)
class TransmonQubit:
    alpha_ghz: float = -0.3  # anharmonicity
    t1_ns: float = 50_000.0
    t2_ns: float = 50_000.0  # must satisfy T2 <= 2 T1
    sigma_ghz: float = 0.0  # quasi-static frequency noise, standard deviation
    levels: int = 4

    def __post_init__(self) -> None:
        if self.t1_ns <= 0 or self.t2_ns <= 0:
            raise ValueError("T1 and T2 must be positive")
        if self.t2_ns > 2 * self.t1_ns * (1 + 1e-12):
            raise ValueError("T2 cannot exceed 2 T1")
        if not 3 <= self.levels <= 8:
            raise ValueError("keep between 3 and 8 levels")

    @property
    def gamma_phi(self) -> float:
        """Pure dephasing rate 1/T_phi, in 1/ns."""
        return max(0.0, 1 / self.t2_ns - 1 / (2 * self.t1_ns))


def _ops(q: TransmonQubit) -> tuple[dq.QArray, dq.QArray, dq.QArray]:
    a = dq.destroy(q.levels)
    return a, a.dag(), dq.number(q.levels)


def _jumps(q: TransmonQubit) -> list[dq.QArray]:
    a, _, n = _ops(q)
    jumps = [math.sqrt(1 / q.t1_ns) * a]
    if q.gamma_phi > 0:
        jumps.append(math.sqrt(2 * q.gamma_phi) * n)
    return jumps


def _static_h(q: TransmonQubit, delta_ghz: float) -> dq.QArray:
    a, ad, n = _ops(q)
    return TWO_PI * (delta_ghz * n + q.alpha_ghz / 2 * (ad @ ad @ a @ a))


def _detunings(
    q: TransmonQubit, delta_ghz: float, n_samples: int = 41
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Gauss-Hermite nodes and weights for averaging over quasi-static noise."""
    if q.sigma_ghz == 0:
        return np.array([delta_ghz]), np.array([1.0])
    x, w = np.polynomial.hermite_e.hermegauss(n_samples)
    return delta_ghz + q.sigma_ghz * x, w / w.sum()


@dataclass(frozen=True)
class Trajectory:
    times_ns: NDArray[np.float64]
    bloch: NDArray[np.float64]  # [time, 3]: <X>, <Y>, <Z> in the 0-1 subspace, Z = P0 - P1
    populations: NDArray[np.float64]  # [time, level]
    omega_x: NDArray[np.float64]  # drive envelopes, GHz (Rabi frequency / 2 pi)
    omega_y: NDArray[np.float64]

    @property
    def leakage(self) -> NDArray[np.float64]:
        return self.populations[:, 2:].sum(axis=1)


def _observables(levels: int) -> list[dq.QArray]:
    p = [dq.proj(dq.basis(levels, k)) for k in range(levels)]
    s01 = dq.basis(levels, 0) @ dq.basis(levels, 1).dag()  # |0><1|
    return [s01 + s01.dag(), -1j * s01 + 1j * s01.dag(), p[0] - p[1], *p]


def driven_h(
    q: TransmonQubit, delta_ghz: float, edges: ArrayLike, omega_x: ArrayLike, omega_y: ArrayLike
) -> dq.TimeQArray:
    """H with piecewise-constant drives: omega_x[k], omega_y[k] (GHz) on [edges[k], edges[k+1])."""
    a, ad, _ = _ops(q)
    edges = jnp.asarray(edges)
    hx = dq.pwc(edges, TWO_PI * jnp.asarray(omega_x), (a + ad) / 2)
    hy = dq.pwc(edges, TWO_PI * jnp.asarray(omega_y), 1j * (ad - a) / 2)
    return _static_h(q, delta_ghz) + hx + hy


def _solve(
    q: TransmonQubit,
    delta_ghz: float,
    rho0: dq.QArray,
    tsave: NDArray[np.float64],
    drive: tuple[NDArray[np.float64], NDArray[np.float64]] | None,
) -> NDArray[np.float64]:
    """Expectations [obs, time], averaged over quasi-static detuning."""
    deltas, weights = _detunings(q, delta_ghz)
    obs = _observables(q.levels)
    total = np.zeros((len(obs), len(tsave)))
    for d, w in zip(deltas, weights, strict=True):
        H: dq.QArray | dq.TimeQArray = _static_h(q, float(d))
        if drive is not None:
            H = driven_h(q, float(d), tsave, drive[0][:-1], drive[1][:-1])
        res = dq.mesolve(H, _jumps(q), rho0, jnp.asarray(tsave), exp_ops=obs, save_states=False)
        total += w * np.asarray(res.expects).real
    return total


def _trajectory(
    q: TransmonQubit,
    delta_ghz: float,
    rho0: dq.QArray,
    tsave: NDArray[np.float64],
    drive: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None,
) -> Trajectory:
    e = _solve(q, delta_ghz, rho0, tsave, drive)
    zeros = np.zeros_like(tsave)
    return Trajectory(
        times_ns=tsave,
        bloch=e[:3].T,
        populations=e[3:].T,
        omega_x=drive[0] if drive else zeros,
        omega_y=drive[1] if drive else zeros,
    )


def gaussian_pulse(
    t: NDArray[np.float64], duration_ns: float, angle: float, drag: float, alpha_ghz: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Truncated Gaussian, sigma = duration / 4, scaled so its area is `angle` radians.

    DRAG adds a quadrature Omega_y = -drag * dOmega_x/dt / (2 pi alpha). drag = 1 is the
    first-order leakage correction; the explorer lets you sweep it.
    """
    sigma = duration_ns / 4
    mid = duration_ns / 2
    g = np.exp(-((t - mid) ** 2) / (2 * sigma**2)) - math.exp(-(mid**2) / (2 * sigma**2))
    g = np.where((t >= 0) & (t <= duration_ns), g, 0.0)
    dt = t[1] - t[0]
    area = TWO_PI * g.sum() * dt  # integral of 2 pi Omega_x dt for unit amplitude
    ox = g * angle / area
    oy = -drag * np.gradient(ox, t) / (TWO_PI * alpha_ghz)
    return ox, oy


def run(
    experiment: Experiment,
    q: TransmonQubit,
    *,
    delta_ghz: float = 0.0,
    omega_ghz: float = 0.02,
    duration_ns: float = 20.0,
    drag: float = 0.0,
    t_max_ns: float = 200.0,
    n_points: int = 201,
) -> Trajectory:
    """One of five standard lab experiments.

    rabi:   constant drive Omega from |0>.
    pulse:  one Gaussian pi pulse (with optional DRAG) from |0>, sampled finely.
    t1:     start in |1>, wait.
    ramsey: start in |+>, wait with detuning delta (ideal pi/2 pulses assumed).
    echo:   start in |+>, ideal pi pulse about X at t/2. The returned point at time t is the
            state at the end of an echo sequence of total length t.
    """
    g0 = dq.todm(dq.basis(q.levels, 0))
    g1 = dq.todm(dq.basis(q.levels, 1))
    plus = dq.todm(dq.unit(dq.basis(q.levels, 0) + dq.basis(q.levels, 1)))
    if experiment == "rabi":
        t = np.linspace(0, t_max_ns, n_points)
        drive = (np.full_like(t, omega_ghz), np.zeros_like(t))
        return _trajectory(q, delta_ghz, g0, t, drive)
    if experiment == "pulse":
        t = np.linspace(0, duration_ns, n_points)
        drive = gaussian_pulse(t, duration_ns, math.pi, drag, q.alpha_ghz)
        return _trajectory(q, delta_ghz, g0, t, drive)
    if experiment == "t1":
        return _trajectory(q, delta_ghz, g1, np.linspace(0, t_max_ns, n_points))
    if experiment == "ramsey":
        return _trajectory(q, delta_ghz, plus, np.linspace(0, t_max_ns, n_points))
    return _echo(q, delta_ghz, plus, np.linspace(0, t_max_ns, n_points))


def run_pulse(
    q: TransmonQubit,
    edges: NDArray[np.float64],
    omega_x: NDArray[np.float64],
    omega_y: NDArray[np.float64],
    per_slice: int = 4,
) -> Trajectory:
    """A piecewise-constant pulse from |0>, saved `per_slice` times within each slice."""
    t = np.linspace(edges[0], edges[-1], per_slice * (len(edges) - 1) + 1)
    k = np.minimum(np.searchsorted(edges, t, side="right") - 1, len(omega_x) - 1)
    g0 = dq.todm(dq.basis(q.levels, 0))
    return _trajectory(q, 0.0, g0, t, (omega_x[k], omega_y[k]))


def _x_pi(levels: int) -> dq.QArray:
    """Ideal pi rotation about X in the 0-1 subspace, identity elsewhere."""
    p01 = dq.proj(dq.basis(levels, 0)) + dq.proj(dq.basis(levels, 1))
    s01 = dq.basis(levels, 0) @ dq.basis(levels, 1).dag()
    return dq.eye(levels) - p01 - 1j * (s01 + s01.dag())


def _echo(
    q: TransmonQubit, delta_ghz: float, rho0: dq.QArray, t: NDArray[np.float64]
) -> Trajectory:
    """Hahn echo: for each total time t, evolve t/2, apply X_pi, evolve t/2."""
    deltas, weights = _detunings(q, delta_ghz)
    obs = _observables(q.levels)
    xpi = _x_pi(q.levels)
    half = jnp.asarray(t / 2)
    total = np.zeros((len(obs), len(t)))
    for d, w in zip(deltas, weights, strict=True):
        H = _static_h(q, float(d))
        first = dq.mesolve(H, _jumps(q), rho0, half).states  # [time, n, n]
        flipped = xpi @ first @ xpi.dag()
        out = np.zeros((len(obs), len(t)))
        for k in range(len(t)):
            tk = float(half[k])
            if tk == 0:
                rho = flipped[k]
            else:
                rho = dq.mesolve(H, _jumps(q), flipped[k], jnp.asarray([0.0, tk])).states[-1]
            out[:, k] = [float(dq.expect(o, rho).real) for o in obs]
        total += w * out
    zeros = np.zeros_like(t)
    return Trajectory(t, total[:3].T, total[3:].T, zeros, zeros)


def average_gate_fidelity(
    q: TransmonQubit,
    H: dq.QArray | dq.TimeQArray,
    t_ns: float,
    target: NDArray[np.complex128] | None = None,
) -> float:
    """Mean of <U psi|rho(t)|U psi> over the six Bloch-axis states, with T1 and T2 on.

    U is the 2x2 `target` on levels 0 and 1 (identity if None). The six axis states form a
    2-design, so this mean equals the average gate fidelity over all pure qubit states.
    Population that leaks above level 1 counts as an error.
    """
    b0, b1 = dq.basis(q.levels, 0), dq.basis(q.levels, 1)
    states = [b0, b1]
    for phase in (1, -1, 1j, -1j):
        states.append(dq.unit(b0 + phase * b1))
    u = np.eye(q.levels, dtype=complex)
    if target is not None:
        u[:2, :2] = target
    U = dq.asqarray(jnp.asarray(u))
    total = 0.0
    for psi in states:
        rho = dq.mesolve(H, _jumps(q), dq.todm(psi), jnp.asarray([0.0, t_ns])).states[-1]
        total += float(dq.fidelity(U @ psi, rho))
    return total / len(states)


def average_idle_fidelity(q: TransmonQubit, t_ns: float) -> float:
    """Average gate fidelity of doing nothing for t_ns at zero detuning."""
    return average_gate_fidelity(q, _static_h(q, 0.0), t_ns)


def flip_probabilities(q: TransmonQubit, t_ns: float) -> tuple[float, float]:
    """Bit-flip and phase-flip probabilities after idling for t, at zero mean detuning.

    From the Pauli transfer matrix: R_Z = (<Z>|0> - <Z>|1>)/2 and R_X = (<X>|+> - <X>|->)/2,
    then p_bit = (1 - R_Z)/2 and p_phase = (1 - R_X)/2. Averaged over quasi-static noise.
    """
    b0, b1 = dq.basis(q.levels, 0), dq.basis(q.levels, 1)
    tsave = np.array([0.0, t_ns])

    def final(psi: dq.QArray, row: int) -> float:
        return float(_solve(q, 0.0, dq.todm(psi), tsave, None)[row, -1])

    r_z = (final(b0, 2) - final(b1, 2)) / 2
    r_x = (final(dq.unit(b0 + b1), 0) - final(dq.unit(b0 - b1), 0)) / 2
    return (1 - r_z) / 2, (1 - r_x) / 2
