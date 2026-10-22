"""Closed-form formulas. SymPy expressions with sources, compiled to JAX.

No simulation code lives here. See `catlab.sim` for Dynamiqs solvers and `catlab.checks`
for where the two are compared.
"""

from catlab.theory import cat, transmon
from catlab.theory.registry import FORMULAS, Formula

__all__ = ["FORMULAS", "Formula", "cat", "transmon"]
