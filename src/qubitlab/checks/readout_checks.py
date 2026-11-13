"""Dispersive readout: closed forms vs the dressed spectrum (scqubits + NumPy) and Dynamiqs.

Findings from the first runs, kept as regimes:
- The first prediction was AGREE within 5% for chi with g_12 = sqrt(2) g_01. It missed by 21%
  at E_J/E_C = 50, f_r = 7 GHz. scqubits gives g_12/g_01 = 1.38 there, not 1.41, and the two
  terms of chi nearly cancel. With the true g_12 (Koch Eqs. 3.9-3.10) it agrees to 2%.
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


@check("readout.fidelity", slow=True)
def fidelity() -> list[CheckResult]:
    """Gambetta 2007 box-car fidelity vs heterodyne records from the stochastic Schrodinger eq.

    Predictions: without decay, a long window AGREEs with Eq. (2.6) at eta = 1 and 1/2; a
    window of 2 / kappa DIVERGEs, because the fields never reach steady state. With T1 = 5 tau
    and well-separated clouds, Eq. (3.8) AGREEs (the resonator lags a decay by ~1/kappa, under
    1% here) and Eq. (2.6), which has no decay, DIVERGEs.

    Finding from the first run: at a window of 40 / kappa the simulated fidelity was 3.9% low,
    close to the 5% tolerance. The ring-up from an empty resonator costs a fixed ~1.4 / kappa of
    signal, more than the 1 / kappa guessed. Rouchon1 also shrinks the field by about 1% at
    dt = 0.01 / kappa (5% at 0.05 / kappa), so the long window is 60 / kappa at dt = 0.01 / kappa.
    """
    out = []
    long = (2.14 / 60) ** 0.5 / 2  # eps for tau_f SNR = 1.07 (F = 0.70) at 60 / kappa
    decay = sim.Readout(0.5, 1.0, 0.5, 40.0, t1=200.0, photons=8)  # tau_f SNR = 20
    regimes = (  # (name, readout, formula, rel_tol, expected); chi = kappa / 2
        ("long", sim.Readout(0.5, 1.0, long, 60.0, photons=5), th.fidelity, 0.05, A),
        (
            "eta",
            sim.Readout(0.5, 1.0, long * 2**0.5, 60.0, eta=0.5, photons=5),
            th.fidelity,
            0.05,
            A,
        ),
        ("short", sim.Readout(0.5, 1.0, 0.517, 2.0, photons=8), th.fidelity, 0.05, D),
        ("t1", decay, th.fidelity_t1, 0.03, A),
        ("t1", decay, th.fidelity, 0.03, D),
    )
    for name, r, formula, tol, expected in regimes:
        sep = th.pointer_separation(epsilon=r.eps, chi=r.chi, kappa=r.kappa, delta_r=r.delta_r)
        snr = th.heterodyne_snr(eta=r.eta, kappa=r.kappa, tau=r.tau, D=sep)
        if formula is th.fidelity_t1:
            theory = th.fidelity_t1(tau_f=r.tau / r.t1, SNR_tau=snr)
        else:
            theory = th.fidelity(SNR_tau=snr)
        g, e = sim.records(r, False, 6000, seed=1), sim.records(r, True, 6000, seed=2)
        out.append(
            CheckResult(
                key=f"{formula.key}.{name}",
                formula=formula.key,
                regime=(
                    f"chi/kappa = {r.chi:g}, kappa tau = {r.tau:g}, eta = {r.eta:g}, "
                    f"T1/tau = {r.t1 / r.tau:g}, tau_f SNR = {float(snr):.3g}"
                ),
                theory=float(theory),
                sim=sim.assignment_fidelity(g, e),
                rel_tol=tol,
                expected=expected,
            )
        )
    return out
