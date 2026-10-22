"""Closed-form formulas. SymPy expressions with sources, compiled to JAX.

No simulation code lives here. See `qubitlab.sim` for Dynamiqs solvers and `qubitlab.checks`
for where the two are compared.
"""

from qubitlab.theory import cat, qubit, transmon
from qubitlab.theory.registry import FORMULAS, Formula

__all__ = ["FORMULAS", "Formula", "cat", "qubit", "transmon"]
