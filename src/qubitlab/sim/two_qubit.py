"""Two-qubit gates, simulated: two coupled transmons and two dissipative cats.

- Transmons: two Duffing oscillators with 3 levels each, H = sum_i (w_i n_i + a_i/2 b_i+^2 b_i^2)
  + g (b_1+ b_2 + b_1 b_2+), built and diagonalized with Dynamiqs operators. `static_zz` reads
  zeta from the dressed spectrum; `cz` evolves a sudden jump of qubit 1 to the 11-20 resonance.
- Cats: the CNOT of chamberland2022 Eqs. (22)-(25), solved as a Lindblad equation with a
  time-dependent jump operator. The control and target parities after the gate give the
  phase-flip chances.

Nothing here uses a closed-form result. `qubitlab.checks` compares these numbers to the theory.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import dynamiqs as dq
import jax.numpy as jnp
import numpy as np

dq.set_precision("double")
dq.set_progress_meter(False)

LEVELS = 3
TWO_PI = 2 * math.pi


def _pair_h(w1: float, w2: float, a1: float, a2: float, g: float) -> dq.QArray:
    """Two Duffing transmons with exchange coupling, frequencies in angular units."""
    b1 = dq.tensor(dq.destroy(LEVELS), dq.eye(LEVELS))
    b2 = dq.tensor(dq.eye(LEVELS), dq.destroy(LEVELS))
    H = w1 * b1.dag() @ b1 + w2 * b2.dag() @ b2
    H += a1 / 2 * b1.dag() @ b1.dag() @ b1 @ b1 + a2 / 2 * b2.dag() @ b2.dag() @ b2 @ b2
    return H + g * (b1.dag() @ b2 + b1 @ b2.dag())


def _index(n1: int, n2: int) -> int:
    return n1 * LEVELS + n2


def static_zz(w1: float, w2: float, a1: float, a2: float, g: float) -> float:
    """zeta = E_11 - E_01 - E_10 + E_00 from the dressed spectrum (same units as the inputs)."""
    H = np.asarray(_pair_h(w1, w2, a1, a2, g).to_jax())
    w, v = np.linalg.eigh(H)

    def e(n1: int, n2: int) -> float:
        return float(w[np.argmax(np.abs(v[_index(n1, n2), :]) ** 2)])

    return e(1, 1) - e(0, 1) - e(1, 0) + e(0, 0)


@dataclass(frozen=True)
class CZ:
    conditional_phase: float  # in [0, 2 pi): arg U11 - arg U01 - arg U10 + arg U00
    leakage: float  # population of |11> that ends outside the computational states


def cz(w2: float, a1: float, a2: float, g: float, t_hold: float) -> CZ:
    """Qubit 1 parked on the 11-20 resonance (w1 = w2 - a1) for t_hold. Angular units."""
    H = _pair_h(w2 - a1, w2, a1, a2, g)
    U = np.asarray(dq.sepropagator(H, jnp.asarray([0.0, t_hold])).final_propagator.to_jax())
    comp = [_index(0, 0), _index(0, 1), _index(1, 0), _index(1, 1)]
    d = [U[k, k] for k in comp]
    phase = np.angle(d[3]) - np.angle(d[1]) - np.angle(d[2]) + np.angle(d[0])
    leak = 1 - float(sum(abs(U[k, comp[3]]) ** 2 for k in comp))
    return CZ(float(phase % TWO_PI), leak)


def cz_sweep(
    w2: float, a1: float, a2: float, g: float, times: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Conditional phase in [0, 2 pi) and leakage of |11> at each hold time, in one solve."""
    H = _pair_h(w2 - a1, w2, a1, a2, g)
    Us = np.asarray(dq.sepropagator(H, jnp.asarray(times)).propagators.to_jax())
    comp = [_index(0, 0), _index(0, 1), _index(1, 0), _index(1, 1)]
    d = Us[:, comp, comp]  # [time, 4] diagonal elements
    phase = np.angle(d[:, 3]) - np.angle(d[:, 1]) - np.angle(d[:, 2]) + np.angle(d[:, 0])
    leak = 1 - np.sum(np.abs(Us[:, comp, comp[3]]) ** 2, axis=1)
    return phase % TWO_PI, leak


@dataclass(frozen=True)
class CatCNOT:
    control_flip: float  # (1 - <P_control>)/2: p_Z1 + p_Z1Z2
    target_flip: float  # (1 - <P_target>)/2: p_Z2 + p_Z1Z2


def cat_cnot(alpha_sq: float, kappa_1: float, T: float, dim: int = 18) -> CatCNOT:
    """chamberland2022 Eqs. (22)-(25), kappa_2 = 1, both cats starting in the even cat |C+>.

    alpha_sq should be an even integer, so the geometric phase pi |alpha|^2 on the control is
    a multiple of 2 pi (chamberland2022 Eq. (26)). 18 Fock states per mode converge |alpha|^2 = 4
    to 1e-3; 14 do not (the control flip came out 70% high).
    """
    alpha = math.sqrt(alpha_sq)
    a1 = dq.tensor(dq.destroy(dim), dq.eye(dim))
    a2 = dq.tensor(dq.eye(dim), dq.destroy(dim))
    eye = dq.eye(dim, dim)
    L1 = a1 @ a1 - alpha_sq * eye
    rot = alpha / 2 * (a1 - alpha * eye)  # multiplies (e^{2 i pi t/T} - 1)
    L2 = (a2 @ a2 - alpha_sq * eye - rot) + dq.modulated(
        lambda t: jnp.exp(2j * jnp.pi * t / T), rot
    )
    H = (
        math.pi
        / (4 * alpha * T)
        * (a1 + a1.dag() - 2 * alpha * eye)
        @ (a2.dag() @ a2 - alpha_sq * eye)
    )
    jumps = [L1, L2, math.sqrt(kappa_1) * a1, math.sqrt(kappa_1) * a2]
    cat = dq.unit(dq.coherent(dim, alpha) + dq.coherent(dim, -alpha))
    rho0 = dq.todm(dq.tensor(cat, cat))
    p1 = dq.tensor(dq.parity(dim), dq.eye(dim))
    p2 = dq.tensor(dq.eye(dim), dq.parity(dim))
    res = dq.mesolve(H, jumps, rho0, jnp.asarray([0.0, T]), exp_ops=[p1, p2])
    e = np.asarray(res.expects).real
    return CatCNOT((1 - e[0, -1]) / 2, (1 - e[1, -1]) / 2)
