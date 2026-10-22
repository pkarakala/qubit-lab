"""Transmon: Koch et al. closed forms vs exact charge-basis diagonalization.

The closed forms are expansions in (E_C / E_J)^(1/2). They should AGREE deep in the transmon
regime and DIVERGE near the Cooper pair box regime, E_J / E_C ~ 1.
"""

from __future__ import annotations

from qubitlab.checks.base import CheckResult, Verdict, check
from qubitlab.sim import transmon as sim
from qubitlab.theory import transmon as th

A, D = Verdict.AGREE, Verdict.DIVERGE
E_C = 1.0  # energies in units of E_C


def _levels(ratio: float, n_g: float) -> list[float]:
    return list(sim.levels(sim.TransmonParams(E_J=ratio * E_C, E_C=E_C), n_g=n_g, n_levels=3))


@check("transmon.qubit_frequency")
def qubit_frequency() -> list[CheckResult]:
    out = []
    for ratio, expected in ((100.0, A), (50.0, A), (20.0, A), (1.0, D)):
        # Average over n_g so charge dispersion does not bias the comparison.
        e0 = [_levels(ratio, g) for g in (0.0, 0.5)]
        e01 = sum(lv[1] - lv[0] for lv in e0) / 2
        out.append(
            CheckResult(
                key=f"transmon.qubit_frequency.{ratio:g}",
                formula=th.qubit_frequency.key,
                regime=f"E_J/E_C = {ratio:g}",
                theory=float(th.qubit_frequency(E_J=ratio * E_C, E_C=E_C)),
                sim=e01,
                rel_tol=0.01,
                expected=expected,
            )
        )
    return out


@check("transmon.anharmonicity")
def anharmonicity() -> list[CheckResult]:
    out = []
    for ratio, expected in ((100.0, A), (50.0, A), (10.0, D)):
        lv = _levels(ratio, 0.25)
        out.append(
            CheckResult(
                key=f"transmon.anharmonicity.{ratio:g}",
                formula=th.anharmonicity.key,
                regime=f"E_J/E_C = {ratio:g}",
                theory=float(th.anharmonicity(E_C=E_C)),
                sim=(lv[2] - lv[1]) - (lv[1] - lv[0]),
                rel_tol=0.20,
                expected=expected,
            )
        )
    return out


@check("transmon.charge_dispersion")
def charge_dispersion() -> list[CheckResult]:
    out = []
    regimes = ((0, 50.0, 0.10, A), (0, 20.0, 0.10, A), (1, 50.0, 0.25, A), (1, 1.0, 0.25, D))
    for level, ratio, tol, expected in regimes:
        half, zero = _levels(ratio, 0.5), _levels(ratio, 0.0)
        out.append(
            CheckResult(
                key=f"transmon.charge_dispersion.m{level}.{ratio:g}",
                formula=th.charge_dispersion.key,
                regime=f"level m = {level}, E_J/E_C = {ratio:g}",
                theory=float(th.charge_dispersion(m=float(level), E_J=ratio * E_C, E_C=E_C)),
                sim=half[level] - zero[level],
                rel_tol=tol,
                expected=expected,
            )
        )
    return out
