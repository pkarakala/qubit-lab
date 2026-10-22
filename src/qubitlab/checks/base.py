from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum


class Verdict(StrEnum):
    AGREE = "AGREE"  # within tolerance
    DIVERGE = "DIVERGE"  # outside tolerance: the formula's assumptions do not hold here


@dataclass(frozen=True)
class CheckResult:
    key: str
    formula: str  # key into qubitlab.theory.FORMULAS
    regime: str  # the parameters, in words
    theory: float
    sim: float
    rel_tol: float
    expected: Verdict  # what the model card predicts for this regime

    @property
    def rel_error(self) -> float:
        scale = max(abs(self.theory), 1e-300)
        return abs(self.sim - self.theory) / scale

    @property
    def verdict(self) -> Verdict:
        return Verdict.AGREE if self.rel_error <= self.rel_tol else Verdict.DIVERGE

    @property
    def as_expected(self) -> bool:
        return self.verdict == self.expected

    def describe(self) -> dict[str, object]:
        return {
            "key": self.key,
            "formula": self.formula,
            "regime": self.regime,
            "theory": self.theory,
            "sim": self.sim,
            "rel_error": self.rel_error,
            "rel_tol": self.rel_tol,
            "verdict": self.verdict.value,
            "expected": self.expected.value,
        }


@dataclass(frozen=True)
class Check:
    key: str
    run: Callable[[], list[CheckResult]]
    slow: bool = False


REGISTRY: list[Check] = []


def check(key: str, slow: bool = False) -> Callable[[Callable[[], list[CheckResult]]], Check]:
    def wrap(fn: Callable[[], list[CheckResult]]) -> Check:
        c = Check(key=key, run=fn, slow=slow)
        REGISTRY.append(c)
        return c

    return wrap


def run_all(include_slow: bool = True) -> list[CheckResult]:
    # Import for side effects: each module registers its checks.
    from qubitlab.checks import cat_checks, transmon_checks  # noqa: F401

    out: list[CheckResult] = []
    for c in REGISTRY:
        if c.slow and not include_slow:
            continue
        out.extend(c.run())
    return out
