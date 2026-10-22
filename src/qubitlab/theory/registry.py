"""A closed-form formula: one SymPy expression, its sources, and a JAX function made from it.

The SymPy expression is the single source of truth. The LaTeX shown in the docs and the
explorer, and the JAX function used in checks and sweeps, are both generated from it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.scipy.special as jsp
import sympy as sp

from qubitlab.provenance import Source

FORMULAS: dict[str, Formula] = {}

# SymPy maps a few functions to the math module, which JAX cannot trace.
_JAX_EXTRAS = {"gamma": jsp.gamma, "factorial": lambda n: jsp.gamma(n + 1)}


@dataclass(eq=False)
class Formula:
    key: str
    title: str
    lhs: str  # LaTeX for the left-hand side
    expr: sp.Expr
    args: tuple[sp.Symbol, ...]
    sources: tuple[Source, ...]
    valid_when: str  # the regime where the formula holds, one line
    _fn: Callable[..., Any] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.sources:
            raise ValueError(f"{self.key}: a formula needs at least one source")
        missing = self.expr.free_symbols - set(self.args)
        if missing:
            raise ValueError(f"{self.key}: symbols {missing} are not in args")
        if self.key in FORMULAS:
            raise ValueError(f"duplicate formula key {self.key}")
        FORMULAS[self.key] = self

    @property
    def latex(self) -> str:
        return f"{self.lhs} = {sp.latex(self.expr, order='none')}"

    @property
    def arg_names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.args)

    def fn(self) -> Callable[..., Any]:
        """JIT-compiled JAX function, positional args in `self.args` order."""
        if self._fn is None:
            self._fn = jax.jit(sp.lambdify(self.args, self.expr, modules=[_JAX_EXTRAS, "jax"]))
        return self._fn

    def __call__(self, **kwargs: Any) -> Any:
        unknown = set(kwargs) - set(self.arg_names)
        if unknown:
            raise TypeError(f"{self.key}: unknown arguments {sorted(unknown)}")
        return self.fn()(*(kwargs[name] for name in self.arg_names))

    def describe(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "title": self.title,
            "latex": self.latex,
            "args": list(self.arg_names),
            "valid_when": self.valid_when,
            "sources": [
                {"key": s.key, "kind": s.kind.value, "locator": s.locator, "note": s.note}
                for s in self.sources
            ],
        }
