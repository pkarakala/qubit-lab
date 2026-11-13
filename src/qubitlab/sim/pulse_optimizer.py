"""Gradient pulse optimization (GRAPE) for a transmon pi pulse, through Dynamiqs.

The pulse is piecewise constant: Omega_x and Omega_y on N equal slices, in the same drive frame,
Hamiltonian and units as `transmon_dynamics`. Each amplitude is Omega_max tanh(u), so the
optimizer can never ask for more drive than Omega_max.

Cost: 1 - F, with F the subspace-averaged gate fidelity of Pedersen et al. (pedersen2007,
Eq. 3) for an X gate on levels 0 and 1. Only the 0-1 block of the propagator enters, so
population left above level 1 counts as error. JAX differentiates the cost through
`dq.sepropagator`; this is the GRAPE idea of Khaneja et al. (khaneja2005). optax.adam takes the
steps.

A pulse made for one model can be scored on another: `gate_error` takes the true anharmonicity
(through `q`) and the true detuning `delta_ghz`. To make a pulse insensitive to a known
uncertainty, `optimize` can instead average 1 - F over a few models (alpha, delta), as
Kobzar et al. average their quality factor over offsets and rf amplitudes (kobzar2004, Eq. 1).

The optimization is closed-system (no T1, T2). `transmon_dynamics.average_gate_fidelity` then
scores the pulse with noise on, like any other pulse.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Sequence
from dataclasses import dataclass

import dynamiqs as dq
import jax
import jax.numpy as jnp
import numpy as np
import optax
from numpy.typing import NDArray

from qubitlab.sim.transmon_dynamics import TransmonQubit, driven_h, gaussian_pulse

X_GATE = np.array([[0, 1], [1, 0]], dtype=complex)


def block(
    q: TransmonQubit, edges: jax.Array, ox: jax.Array, oy: jax.Array, delta_ghz: float = 0.0
) -> jax.Array:
    """The 0-1 block of the propagator over the whole pulse, at drive detuning delta."""
    H = driven_h(q, delta_ghz, edges, ox, oy)
    U = dq.sepropagator(H, jnp.asarray([0.0, edges[-1]])).final_propagator
    return jnp.asarray(U.to_jax())[:2, :2]


def subspace_fidelity(m: jax.Array) -> jax.Array:
    """pedersen2007 Eq. (3), with M_rel = P U0^dagger U P and n_rel = 2."""
    n = m.shape[0]
    return (jnp.trace(m @ m.conj().T).real + jnp.abs(jnp.trace(m)) ** 2) / (n * (n + 1))


def gate_error(
    q: TransmonQubit,
    edges: NDArray[np.float64],
    ox: NDArray[np.float64],
    oy: NDArray[np.float64],
    delta_ghz: float = 0.0,
) -> tuple[float, float]:
    """Closed-system X-gate error 1 - F and the mean leakage out of levels 0 and 1."""
    b = block(q, jnp.asarray(edges), jnp.asarray(ox), jnp.asarray(oy), delta_ghz)
    m = jnp.asarray(X_GATE.conj().T) @ b
    leakage = 1 - float(jnp.trace(b @ b.conj().T).real) / 2
    return 1 - float(subspace_fidelity(m)), leakage


def drag_slices(
    q: TransmonQubit, duration_ns: float, slices: int, drag: float = 1.0
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """The Gaussian DRAG pi pulse of `transmon_dynamics`, sampled at slice midpoints."""
    edges = np.linspace(0.0, duration_ns, slices + 1)
    fine = np.linspace(0.0, duration_ns, 20 * slices + 1)
    ox, oy = gaussian_pulse(fine, duration_ns, math.pi, drag, q.alpha_ghz)
    mid = (edges[:-1] + edges[1:]) / 2
    return edges, np.interp(mid, fine, ox), np.interp(mid, fine, oy)


@dataclass(frozen=True)
class Optimized:
    edges_ns: NDArray[np.float64]
    omega_x: NDArray[np.float64]  # GHz, one value per slice
    omega_y: NDArray[np.float64]
    history: NDArray[np.float64]  # cost 1 - F at each step
    error: float
    leakage: float


def _cost(
    u: jax.Array,
    levels: int,
    alphas: jax.Array,
    deltas: jax.Array,
    edges: jax.Array,
    omega_max: jax.Array,
) -> jax.Array:
    """Mean of 1 - F over the models (alphas[k], deltas[k]), amplitudes omega_max tanh(u)."""
    ox, oy = omega_max * jnp.tanh(u)

    def one(alpha: jax.Array, delta: jax.Array) -> jax.Array:
        q = TransmonQubit(alpha_ghz=alpha, levels=levels)  # type: ignore[arg-type]
        m = jnp.asarray(X_GATE.conj().T) @ block(q, edges, ox, oy, delta)  # type: ignore[arg-type]
        return 1 - subspace_fidelity(m)

    return jnp.mean(jax.vmap(one)(alphas, deltas))


@functools.partial(jax.jit, static_argnums=0)
def _step(
    levels: int,
    u: jax.Array,
    state: optax.OptState,
    alphas: jax.Array,
    deltas: jax.Array,
    edges: jax.Array,
    omega_max: jax.Array,
    learning_rate: jax.Array,
) -> tuple[jax.Array, optax.OptState, jax.Array]:
    """One Adam step on the mean of 1 - F over the models.

    Everything but the level count is an argument, not a constant, so JAX compiles this once
    per level count, slice count and number of models, and its disk cache keeps that across
    restarts.
    """
    value, grad = jax.value_and_grad(_cost)(u, levels, alphas, deltas, edges, omega_max)
    updates, state = optax.adam(learning_rate).update(grad, state)
    return optax.apply_updates(u, updates), state, value


def optimize(
    q: TransmonQubit,
    duration_ns: float,
    *,
    slices: int = 24,
    omega_max_ghz: float = 0.3,
    steps: int = 300,
    learning_rate: float = 0.05,
    delta_ghz: float = 0.0,
    ensemble: Sequence[tuple[float, float]] | None = None,
) -> Optimized:
    """Start from the DRAG pulse and follow the gradient of 1 - F.

    `ensemble` is a list of (alpha_ghz, delta_ghz) models; the cost is then the mean of 1 - F
    over them (kobzar2004, Eq. 1). By default it is the single model (q.alpha_ghz, delta_ghz).
    `error` and `leakage` are always for that single model.
    """
    edges, ox0, oy0 = drag_slices(q, duration_ns, slices)
    start = np.clip(np.stack([ox0, oy0]) / omega_max_ghz, -0.99, 0.99)
    u = jnp.arctanh(jnp.asarray(start))
    state = optax.adam(learning_rate).init(u)
    models = np.asarray(ensemble if ensemble is not None else [(q.alpha_ghz, delta_ghz)])
    args = (
        jnp.asarray(models[:, 0]),
        jnp.asarray(models[:, 1]),
        jnp.asarray(edges),
        jnp.asarray(omega_max_ghz),
    )
    lr = jnp.asarray(learning_rate)
    history = np.empty(steps)
    for k in range(steps):
        u, state, value = _step(q.levels, u, state, *args, lr)
        history[k] = float(value)
    ox, oy = (np.asarray(v) for v in omega_max_ghz * jnp.tanh(u))
    error, leakage = gate_error(q, edges, ox, oy, delta_ghz)
    return Optimized(edges, ox, oy, history, error, leakage)
