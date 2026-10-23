"""Dispersive readout: closed forms vs the dressed spectrum (scqubits + NumPy) and Dynamiqs.

Findings from the first runs, kept as regimes:
- The first prediction was AGREE within 5% for chi with g_12 = sqrt(2) g_01. It missed by 21%
  at E_J/E_C = 50, f_r = 7 GHz. scqubits gives g_12/g_01 = 1.38 there, not 1.41, and the two
  terms of chi nearly cancel. With the true g_12 (Koch Eqs. 27-28) it agrees to 2%.
- At large |Delta| the remaining gap is the counter-rotating terms: dropping them from the
  simulation brings it to 0.5%. So that regime DIVERGEs, by the formula's own assumptions.
- The weak-chi dephasing rate was predicted to be 2x high at chi = kappa/2. It is 4x: the
  drive also fills half as many photons as 4 eps^2 / kappa^2.
"""

from __future__ import annotations

from qubitlab.checks.base import CheckResult, Verdict, check
from qubitlab.sim import readout as sim
from qubitlab.theory import readout as th

A, D = Verdict.AGREE, Verdict.DIVERGE
E_C, G_N = 0.3, 0.05  # GHz; g_n multiplies n (a + a+)


@check("readout.chi")
def chi() -> list[CheckResult]:
    out = []
    regimes = (  # (E_J/E_C, f_r in GHz, formula, expected)
        (50, 7.0, th.chi_transmon, A),
        (100, 6.2, th.chi_transmon, A),
        (200, 5.0, th.chi_transmon, D),
        (50, 7.0, th.chi_transmon_harmonic, D),
        (100, 6.2, th.chi_transmon_harmonic, D),
        (50, 7.0, th.chi_two_level, D),
    )
    for ratio, f_r, formula, expected in regimes:
        d = sim.dressed_chi(ratio * E_C, E_C, f_r, G_N)
        delta = d.f01 - f_r
        if formula is th.chi_transmon:
            theory = th.chi_transmon(g=d.g01, g_12=d.g12, Delta=delta, alpha=d.alpha)
        elif formula is th.chi_transmon_harmonic:
            theory = th.chi_transmon_harmonic(g=d.g01, Delta=delta, alpha=d.alpha)
        else:
            theory = th.chi_two_level(g=d.g01, Delta=delta)
        out.append(
            CheckResult(
                key=f"{formula.key}.{ratio:g}.{f_r:g}",
                formula=formula.key,
                regime=(
                    f"E_J/E_C = {ratio:g}, f_q = {d.f01:.2f} GHz, f_r = {f_r:g} GHz, "
                    f"g = {d.g01 * 1e3:.0f} MHz"
                ),
                theory=float(theory),
                sim=d.chi,
                rel_tol=0.05,
                expected=expected,
            )
        )
    return out


@check("readout.dephasing")
def dephasing() -> list[CheckResult]:
    out = []
    kappa, eps = 1.0, 0.2  # units of kappa
    for chi_over_kappa, formula, expected in (
        (0.05, th.dephasing_rate, A),
        (0.5, th.dephasing_rate, A),
        (0.05, th.dephasing_rate_weak, A),
        (0.5, th.dephasing_rate_weak, D),
    ):
        c = chi_over_kappa * kappa
        out.append(
            CheckResult(
                key=f"{formula.key}.{chi_over_kappa:g}",
                formula=formula.key,
                regime=f"chi/kappa = {chi_over_kappa:g}, eps/kappa = {eps:g}",
                theory=float(formula(kappa=kappa, chi=c, epsilon=eps)),
                sim=sim.dephasing_rate(c, eps, kappa),
                rel_tol=0.05,
                expected=expected,
            )
        )
    return out


@check("readout.pointer_photons")
def pointer_photons() -> list[CheckResult]:
    out = []
    kappa, eps, c = 1.0, 0.3, 0.5
    for delta_r in (0.0, c):  # drive at the bare resonator, then on the |g> resonance
        p = sim.pointer_trajectories(c, eps, kappa, delta_r, t_max=20 / kappa, n_points=2)
        for name, sign, n in (("e", 1, p.photons_e[-1]), ("g", -1, p.photons_g[-1])):
            out.append(
                CheckResult(
                    key=f"readout.pointer_photons.{name}.{delta_r:g}",
                    formula=th.pointer_photons.key,
                    regime=f"qubit in |{name}>, chi/kappa = {c:g}, delta_r/kappa = {delta_r:g}",
                    theory=float(
                        th.pointer_photons(epsilon=eps, kappa=kappa, delta_r=delta_r, chi=sign * c)
                    ),
                    sim=float(n),
                    rel_tol=0.001,
                    expected=A,
                )
            )
    return out
