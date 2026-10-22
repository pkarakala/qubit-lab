"""Gradient pulse optimization (GRAPE) for a transmon pi pulse, through Dynamiqs.

The pulse is piecewise constant: Omega_x and Omega_y on N equal slices, in the same drive frame,
Hamiltonian and units as `transmon_dynamics`. Each amplitude is Omega_max tanh(u), so the
optimizer can never ask for more drive than Omega_max.

Cost: 1 - F, with F the subspace-averaged gate fidelity of Pedersen et al. (pedersen2007,
Eq. 3) for an X gate on levels 0 and 1. Only the 0-1 block of the propagator enters, so
population left above level 1 counts as error. JAX differentiates the cost through
`dq.sepropagator`; this is the GRAPE idea of Khaneja et al. (khaneja2005). optax.adam takes the
steps.

The optimization is closed-system (no T1, T2). `transmon_dynamics.average_gate_fidelity` then
scores the pulse with noise on, like any other pulse.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import dynamiqs as dq
import jax
import jax.numpy as jnp
import numpy as np
import optax
from numpy.typing import NDArray

from qubitlab.sim.transmon_dynamics import TransmonQubit, driven_h, gaussian_pulse

X_GATE = np.array([[0, 1], [1, 0]], dtype=complex)


def block(q: TransmonQubit, edges: jax.Array, ox: jax.Array, oy: jax.Array) -> jax.Array:
    """The 0-1 block of the propagator over the whole pulse."""
    H = driven_h(q, 0.0, edges, ox, oy)
    U = dq.sepropagator(H, jnp.asarray([0.0, edges[-1]])).final_propagator
    return jnp.asarray(U.to_jax())[:2, :2]


def subspace_fidelity(m: jax.Array) -> jax.Array:
    """pedersen2007 Eq. (3), with M_rel = P U0^dagger U P and n_rel = 2."""
    n = m.shape[0]
    return (jnp.trace(m @ m.conj().T).real + jnp.abs(jnp.trace(m)) ** 2) / (n * (n + 1))


def gate_error(
    q: TransmonQubit, edges: NDArray[np.float64], ox: NDArray[np.float64], oy: NDArray[np.float64]
) -> tuple[float, float]:
    """Closed-system X-gate error 1 - F and the mean leakage out of levels 0 and 1."""
    b = block(q, jnp.asarray(edges), jnp.asarray(ox), jnp.asarray(oy))
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


def optimize(
    q: TransmonQubit,
    duration_ns: float,
    *,
    slices: int = 24,
    omega_max_ghz: float = 0.3,
    steps: int = 300,
    learning_rate: float = 0.05,
) -> Optimized:
    """Start from the DRAG pulse and follow the gradient of 1 - F."""
    edges, ox0, oy0 = drag_slices(q, duration_ns, slices)
    edges_j = jnp.asarray(edges)

    def amplitudes(u: jax.Array) -> jax.Array:
        return omega_max_ghz * jnp.tanh(u)

    def cost(u: jax.Array) -> jax.Array:
        ox, oy = amplitudes(u)
        m = jnp.asarray(X_GATE.conj().T) @ block(q, edges_j, ox, oy)
        return 1 - subspace_fidelity(m)

    start = np.clip(np.stack([ox0, oy0]) / omega_max_ghz, -0.99, 0.99)
    u = jnp.arctanh(jnp.asarray(start))
    opt = optax.adam(learning_rate)
    state = opt.init(u)

    @jax.jit
    def step(u: jax.Array, state: optax.OptState) -> tuple[jax.Array, optax.OptState, jax.Array]:
        value, grad = jax.value_and_grad(cost)(u)
        updates, state = opt.update(grad, state)
        return optax.apply_updates(u, updates), state, value

    history = np.empty(steps)
    for k in range(steps):
        u, state, value = step(u, state)
        history[k] = float(value)
    ox, oy = (np.asarray(v) for v in amplitudes(u))
    error, leakage = gate_error(q, edges, ox, oy)
    return Optimized(edges, ox, oy, history, error, leakage)
