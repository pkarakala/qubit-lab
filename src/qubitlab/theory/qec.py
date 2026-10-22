"""Closed forms for error correction: how logical errors and qubit counts scale with distance.

Conventions (shared with `qubitlab.sim.qec`):
- p: physical error probability per operation, d: code distance (odd).
- P_L: logical error probability per round of stabilizer measurement.
"""

from __future__ import annotations

import sympy as sp

from qubitlab.provenance import Kind, Source
from qubitlab.theory.registry import Formula

p, p_th, d, p_bit = sp.symbols("p p_th d p_bit", positive=True)


def fowler(locator: str, note: str = "") -> Source:
    return Source("fowler2012", Kind.PEER_REVIEWED, locator, note)


def hand(step: str) -> Source:
    return Source("qubitlab_hand_calcs", Kind.DERIVED, "docs/hand_calcs/repetition_code.py", step)


error_exponent = Formula(
    key="qec.error_exponent",
    title="Errors needed to fool a distance-d code",
    lhs=r"d_e",
    expr=(d + 1) / 2,
    args=(d,),
    sources=(
        fowler("Eq. (10)", "Odd d. For the surface code."),
        hand("Step 1: the same count for the repetition code."),
    ),
    valid_when="Odd d, p well below threshold. Sets the slope of log P_L against log p.",
)

surface_code_logical_error = Formula(
    key="qec.surface_code_logical_error",
    title="Surface code logical error rate",
    lhs=r"P_L",
    expr=sp.Mul(sp.Float(0.03, 2), sp.Pow(p / p_th, (d + 1) / 2, evaluate=False), evaluate=False),
    args=(p, p_th, d),
    sources=(
        fowler(
            "Eq. (11); p_th = 0.57% in Sec. VII",
            "Empirical fit to Fowler's own noise model. Whether P_L is per round is unverified.",
        ),
    ),
    valid_when="p below threshold, Fowler's noise model. The prefactor is model dependent.",
)

surface_code_qubits = Formula(
    key="qec.surface_code_qubits",
    title="Physical qubits for one surface code logical qubit",
    lhs=r"n_q",
    expr=(2 * d - 1) ** 2,
    args=(d,),
    sources=(fowler("Sec. VII.A, text after Eq. (12)", "Data plus measurement qubits."),),
    valid_when="Rotated-boundary-free planar layout as drawn by Fowler.",
)

repetition_code_qubits = Formula(
    key="qec.repetition_code_qubits",
    title="Physical qubits for one repetition code logical qubit",
    lhs=r"n_q",
    expr=2 * d - 1,
    args=(d,),
    sources=(hand("Header: d data qubits and d - 1 ancillas."),),
    valid_when="Line of qubits, one ancilla between each pair of neighbors.",
)

repetition_code_bit_flip_floor = Formula(
    key="qec.repetition_bit_flip_floor",
    title="Logical errors a repetition code cannot correct",
    lhs=r"P_{L,\mathrm{bit}}",
    expr=d * p_bit,
    args=(d, p_bit),
    sources=(hand("Step 2: a flip on any data qubit is a logical flip."),),
    valid_when="d * p_bit << 1. Grows with d, so a bigger code can make things worse.",
)

ALL = (
    error_exponent,
    surface_code_logical_error,
    surface_code_qubits,
    repetition_code_qubits,
    repetition_code_bit_flip_floor,
)
