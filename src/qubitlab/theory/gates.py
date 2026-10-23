"""Closed forms for two-qubit gates: a transmon CZ and a bias-preserving cat CNOT.

Transmons (krantz2019 Sec. IV.F): energies E_nm = E_n(q1) + E_m(q2), exchange coupling g,
sqrt(2) g between |11> and |20>. Angular units, hbar = 1. Delta = omega_1 - omega_2,
alpha_i = anharmonicities (negative for transmons).

Cats (guillaud2019 Sec. IV and VI): two-photon dissipation kappa_2, single-photon loss kappa_1,
n = |alpha|^2 photons per cat, gate time T. Z1 is a phase-flip on the control, Z2 on the target.
"""

from __future__ import annotations

import sympy as sp

from qubitlab.provenance import Kind, Source
from qubitlab.theory.registry import Formula

g, Delta, alpha_1, alpha_2 = sp.symbols("g Delta alpha_1 alpha_2", real=True)
n, kappa_1, kappa_2, T = sp.symbols("n kappa_1 kappa_2 T", positive=True)


def hand(step: str) -> Source:
    return Source("qubitlab_hand_calcs", Kind.DERIVED, "docs/hand_calcs/two_qubit_gates.py", step)


static_zz = Formula(
    key="gates.static_zz",
    title="Always-on ZZ shift of two coupled transmons",
    lhs=r"\zeta",
    expr=2 * g**2 * (1 / (Delta - alpha_2) - 1 / (Delta + alpha_1)),
    args=(g, Delta, alpha_1, alpha_2),
    sources=(
        Source("krantz2019", Kind.PEER_REVIEWED, "Eqs. (116), (119)", "zeta = w11 - w01 - w10."),
        hand("Step 1: second-order shifts of |01>, |10>, |11>."),
    ),
    valid_when="g much smaller than |Delta|, |Delta - alpha_2| and |Delta + alpha_1|.",
)

cz_time = Formula(
    key="gates.cz_time",
    title="Diabatic CZ: hold time at the 11-20 resonance",
    lhs=r"t_{CZ}",
    expr=sp.pi / (sp.sqrt(2) * g),
    args=(g,),
    sources=(
        Source("krantz2019", Kind.PEER_REVIEWED, "Sec. IV.F, text after Eq. (123)"),
        hand("Step 2: a full cycle in {|11>, |20>} returns |11> with a phase of -1."),
    ),
    valid_when="Sudden jump to the resonance, g << |alpha|, so |01>, |10> barely move.",
)

cnot_z1 = Formula(
    key="gates.cat_cnot_z1",
    title="Cat CNOT: phase-flip chance on the control",
    lhs=r"p_{Z_1}",
    expr=sp.Add(n * kappa_1 * T, 1 / (2 * sp.pi * n * kappa_2 * T), evaluate=False),
    args=(n, kappa_1, kappa_2, T),
    sources=(
        Source(
            "guillaud2019",
            Kind.PEER_REVIEWED,
            "Sec. VI, Gate fidelity and optimal gate time",
            "The 1/(2 pi) is a numerical fit in the paper.",
        ),
        Source(
            "chamberland2022",
            Kind.PEER_REVIEWED,
            "Eq. (27)",
            "Perturbation theory gives pi^2/64 = 0.154 in place of 1/(2 pi) = 0.159.",
        ),
    ),
    valid_when="Photon loss only, kappa_1 << kappa_2, gate not much faster than 1/(kappa_2 n).",
)

cnot_z2 = Formula(
    key="gates.cat_cnot_z2",
    title="Cat CNOT: phase-flip chance on the target (and on both)",
    lhs=r"p_{Z_2} = p_{Z_1 Z_2}",
    expr=n * kappa_1 * T / 2,
    args=(n, kappa_1, T),
    sources=(
        Source("guillaud2019", Kind.PEER_REVIEWED, "Sec. VI, Eq. (3)"),
        Source("chamberland2022", Kind.PEER_REVIEWED, "Eq. (27)"),
    ),
    valid_when="Photon loss only. Loss on the target during the gate spreads to the control.",
)

ALL = (static_zz, cz_time, cnot_z1, cnot_z2)
