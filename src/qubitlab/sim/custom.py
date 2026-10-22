"""Lindblad dynamics of one oscillator mode, from a Hamiltonian and jumps the user types.

The expressions are parsed with Python's `ast` against a whitelist, never `eval`. Allowed:
numbers (1j for i), the operators a, ad (a dagger), n (number) and I (identity), the user's
parameter names, + - * /, ** with a small whole-number exponent, and sqrt() of a scalar.
Operator times operator is a matrix product. A scalar added to an operator means scalar * I.

Each parse also writes the same expression as Dynamiqs code, so the explorer can hand the user
a script that reproduces the run.
"""

from __future__ import annotations

import ast
import cmath
import math
from dataclasses import dataclass

import dynamiqs as dq
import numpy as np

from qubitlab.sim.cat import Evolution, Initial, evolve_operators, initial_state

MAX_CHARS = 300
MAX_NODES = 200
MAX_POWER = 8
OPERATORS = {"a": "a", "ad": "ad", "n": "n", "I": "eye"}
RESERVED = set(OPERATORS) | {"sqrt", "dq", "jnp", "N"}


class ModelError(ValueError):
    """The typed model is not allowed or does not make sense."""


@dataclass(frozen=True)
class Term:
    value: complex | dq.QArray
    code: str

    @property
    def is_op(self) -> bool:
        return not isinstance(self.value, complex)


def _scalar(t: Term) -> complex:
    if not isinstance(t.value, complex):
        raise ModelError(f"{t.code} must be a number here, not an operator")
    return t.value


def _finite_op(op: dq.QArray) -> bool:
    return bool(np.isfinite(np.asarray(op.to_jax())).all())


class _Parser:
    def __init__(self, dim: int, params: dict[str, float]) -> None:
        a = dq.destroy(dim)
        self.ops = {"a": a, "ad": a.dag(), "n": dq.number(dim), "I": dq.eye(dim)}
        self.params = params

    def parse(self, text: str) -> Term:
        if len(text) > MAX_CHARS:
            raise ModelError(f"keep expressions under {MAX_CHARS} characters")
        try:
            tree = ast.parse(text, mode="eval")
        except SyntaxError as e:
            raise ModelError(f"cannot read {text!r}: {e.msg}") from e
        if sum(1 for _ in ast.walk(tree)) > MAX_NODES:
            raise ModelError("expression is too long")
        try:
            return self.node(tree.body)
        except (OverflowError, ZeroDivisionError) as e:
            raise ModelError(f"{text!r}: {e}") from e

    def node(self, x: ast.expr) -> Term:
        t = self._node(x)
        finite = cmath.isfinite(t.value) if isinstance(t.value, complex) else _finite_op(t.value)
        if not finite:
            raise ModelError(f"{ast.unparse(x)!r} overflows")
        return t

    def _node(self, x: ast.expr) -> Term:
        if isinstance(x, ast.Constant) and isinstance(x.value, int | float | complex):
            if isinstance(x.value, bool):
                raise ModelError("booleans are not allowed")
            return Term(complex(x.value), repr(x.value))
        if isinstance(x, ast.Name):
            if x.id in self.ops:
                return Term(self.ops[x.id], OPERATORS[x.id])
            if x.id in self.params:
                return Term(complex(self.params[x.id]), x.id)
            raise ModelError(f"unknown name {x.id!r}; add it as a parameter")
        if isinstance(x, ast.UnaryOp) and isinstance(x.op, ast.USub | ast.UAdd):
            t = self.node(x.operand)
            sign = -1 if isinstance(x.op, ast.USub) else 1
            return Term(sign * t.value, f"{'-' if sign < 0 else '+'}({t.code})")
        if isinstance(x, ast.Call) and isinstance(x.func, ast.Name) and x.func.id == "sqrt":
            if len(x.args) != 1 or x.keywords:
                raise ModelError("sqrt takes one argument")
            arg = self.node(x.args[0])
            v = _scalar(arg)
            if v.imag != 0 or v.real < 0:
                raise ModelError("sqrt needs a non-negative real number")
            return Term(complex(math.sqrt(v.real)), f"jnp.sqrt({arg.code})")
        if isinstance(x, ast.BinOp):
            return self.binop(x)
        raise ModelError(f"{ast.unparse(x)!r} is not allowed")

    def binop(self, x: ast.BinOp) -> Term:
        left = self.node(x.left)
        if isinstance(x.op, ast.Pow):
            if not (isinstance(x.right, ast.Constant) and type(x.right.value) is int):
                raise ModelError("powers must be whole numbers, like a**2")
            k = x.right.value
            if not 0 <= k <= MAX_POWER:
                raise ModelError(f"powers must be between 0 and {MAX_POWER}")
            if left.is_op:
                return Term(dq.powm(left.value, k), f"dq.powm({left.code}, {k})")
            return Term(_scalar(left) ** k, f"({left.code}) ** {k}")
        right = self.node(x.right)
        if isinstance(x.op, ast.Add | ast.Sub):
            sign = 1 if isinstance(x.op, ast.Add) else -1
            sym = "+" if sign > 0 else "-"
            lv, lc = self._promote(left, right)
            rv, rc = self._promote(right, left)
            return Term(lv + sign * rv, f"({lc} {sym} {rc})")
        if isinstance(x.op, ast.Mult):
            if not isinstance(left.value, complex) and not isinstance(right.value, complex):
                return Term(left.value @ right.value, f"({left.code} @ {right.code})")
            return Term(left.value * right.value, f"({left.code} * {right.code})")
        if isinstance(x.op, ast.Div):
            d = _scalar(right)
            if d == 0:
                raise ModelError("division by zero")
            return Term(left.value / d, f"({left.code} / {right.code})")
        raise ModelError(f"operator {type(x.op).__name__} is not allowed")

    def _promote(self, t: Term, other: Term) -> tuple[complex | dq.QArray, str]:
        """A scalar next to an operator in a sum stands for scalar * I."""
        if other.is_op and not t.is_op:
            return t.value * self.ops["I"], f"{t.code} * eye"
        return t.value, t.code


@dataclass(frozen=True)
class Jump:
    op: str  # operator expression
    rate: str  # non-negative scalar expression; the jump operator is sqrt(rate) * op


@dataclass(frozen=True)
class Model:
    hamiltonian: str
    jumps: tuple[Jump, ...]
    params: dict[str, float]
    dim: int = 30

    def __post_init__(self) -> None:
        if not 4 <= self.dim <= 60:
            raise ModelError("keep the Fock space between 4 and 60 states")
        bad = [p for p in self.params if not p.isidentifier() or p in RESERVED or p[0] == "_"]
        if bad:
            raise ModelError(f"parameter names {bad} are not allowed")
        if len(self.params) > 12 or not all(math.isfinite(v) for v in self.params.values()):
            raise ModelError("at most 12 parameters, each a finite number")
        if len(self.jumps) > 6:
            raise ModelError("at most 6 jump operators")


@dataclass(frozen=True)
class Built:
    H: dq.QArray
    jumps: list[dq.QArray]
    h_code: str
    jump_codes: list[str]


def build(model: Model) -> Built:
    p = _Parser(model.dim, model.params)
    h = p.parse(model.hamiltonian or "0")
    H = h.value if h.is_op else h.value * p.ops["I"]
    Hm = np.asarray(H.to_jax())  # type: ignore[union-attr]
    if not np.allclose(Hm, Hm.conj().T, atol=1e-9 * max(1.0, float(np.abs(Hm).max()))):
        raise ModelError("the Hamiltonian is not Hermitian; add the conjugate terms")
    jumps, codes = [], []
    for j in model.jumps:
        op = p.parse(j.op)
        if not op.is_op:
            raise ModelError(f"jump {j.op!r} must contain an operator")
        r = p.parse(j.rate)
        rate = _scalar(r)
        if rate.imag != 0 or rate.real < 0:
            raise ModelError(f"rate {j.rate!r} must be a non-negative real number")
        if rate.real == 0:
            continue
        jumps.append(math.sqrt(rate.real) * op.value)
        codes.append(f"jnp.sqrt({r.code}) * {op.code}")
    return Built(H, jumps, h.code if h.is_op else f"{h.code} * eye", codes)


def evolve(
    model: Model,
    initial: Initial,
    alpha_sq: float,
    t_final: float,
    n_frames: int,
    grid: int,
    extent: float,
) -> Evolution:
    b = build(model)
    rho0 = dq.todm(initial_state(initial, alpha_sq, model.dim))
    return evolve_operators(b.H, b.jumps, rho0, t_final, n_frames, grid, extent)


_INITIAL_CODE = {
    "vacuum": "dq.vacuum(N)",
    "coherent_plus": "dq.coherent(N, alpha)",
    "coherent_minus": "dq.coherent(N, -alpha)",
    "even_cat": "dq.unit(dq.coherent(N, alpha) + dq.coherent(N, -alpha))",
    "odd_cat": "dq.unit(dq.coherent(N, alpha) - dq.coherent(N, -alpha))",
}


def python_script(
    model: Model, initial: Initial, alpha_sq: float, t_final: float, n_frames: int
) -> str:
    """A standalone Dynamiqs script that runs the same model."""
    b = build(model)
    params = "\n".join(f"{k} = {v!r}" for k, v in model.params.items())
    jumps = ",\n    ".join(b.jump_codes)
    return f"""# Generated by Qubit Lab, github.com/pkarakala/qubit-lab. Needs: pip install dynamiqs
import dynamiqs as dq
import jax.numpy as jnp

N = {model.dim}  # Fock states kept
{params}
alpha = {math.sqrt(alpha_sq)!r}  # initial state amplitude

a = dq.destroy(N)
ad = a.dag()
n = dq.number(N)
eye = dq.eye(N)

H = {b.h_code}
jumps = [
    {jumps}
]
rho0 = dq.todm({_INITIAL_CODE[initial]})
tsave = jnp.linspace(0.0, {t_final!r}, {n_frames})
result = dq.mesolve(H, jumps, rho0, tsave, exp_ops=[n, dq.parity(N)])
print("<n> at the end:", float(result.expects[0, -1].real))
print("parity at the end:", float(result.expects[1, -1].real))
"""
