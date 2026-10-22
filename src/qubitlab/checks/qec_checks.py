"""Error correction: the d_e = (d + 1)/2 scaling law vs Stim + PyMatching memory experiments.

Below threshold, log P_L against log p should have slope d_e (AGREE). Near threshold the
power law no longer holds and the slope collapses (DIVERGE).
"""

from __future__ import annotations

from qubitlab.checks.base import CheckResult, Verdict, check
from qubitlab.sim import qec as sim
from qubitlab.theory import qec as th

A, D = Verdict.AGREE, Verdict.DIVERGE
SHOTS = 400_000


@check("qec.error_exponent", slow=True)
def error_exponent() -> list[CheckResult]:
    out = []
    # (code, d, p_lo, p_hi, expected). Low-p windows are chosen so each point sees failures.
    regimes = (
        ("surface", 3, 0.0008, 0.002, A),
        ("surface", 5, 0.0015, 0.003, A),
        ("repetition", 3, 0.002, 0.006, A),
        ("repetition", 5, 0.003, 0.008, A),
        ("surface", 5, 0.008, 0.012, D),
    )
    for code, d, lo, hi, expected in regimes:
        out.append(
            CheckResult(
                key=f"qec.error_exponent.{code}.{d}.{lo:g}",
                formula=th.error_exponent.key,
                regime=f"{code} code, d = {d}, p = {lo:g} to {hi:g}",
                theory=float(th.error_exponent(d=float(d))),
                sim=sim.slope(code, d, lo, hi, SHOTS),  # type: ignore[arg-type]
                rel_tol=0.12,
                expected=expected,
            )
        )
    return out


@check("qec.qubit_count")
def qubit_count() -> list[CheckResult]:
    out = []
    # Stim builds the rotated surface code. Fowler's count is for the planar layout he draws,
    # so it should not match: the rotated layout is the smaller one. The repetition code should.
    regimes = (
        ("repetition", th.repetition_code_qubits, 3, A),
        ("repetition", th.repetition_code_qubits, 7, A),
        ("surface", th.surface_code_qubits, 5, D),
    )
    for code, formula, d, expected in regimes:
        out.append(
            CheckResult(
                key=f"qec.qubit_count.{code}.{d}",
                formula=formula.key,
                regime=f"{code} code, d = {d}",
                theory=float(formula(d=float(d))),
                sim=float(sim.physical_qubits(code, d)),  # type: ignore[arg-type]
                rel_tol=0.01,
                expected=expected,
            )
        )
    return out
