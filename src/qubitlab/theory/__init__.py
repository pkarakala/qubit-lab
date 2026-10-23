"""Closed-form formulas. SymPy expressions with sources, compiled to JAX.

No simulation code lives here. See `qubitlab.sim` for Dynamiqs solvers and `qubitlab.checks`
for where the two are compared.
"""

from qubitlab.theory import cat, gates, qec, qubit, readout, transmon
from qubitlab.theory.registry import FORMULAS, Formula

__all__ = ["FORMULAS", "Formula", "cat", "gates", "qec", "qubit", "readout", "transmon"]
