"""Driven, decaying transmon: two-level closed forms vs a Dynamiqs simulation with 4 levels.

The two-level formulas should AGREE whenever level 2 stays empty, and DIVERGE for a drive
that is not small compared with the anharmonicity.
"""

from __future__ import annotations

import math

import numpy as np

from qubitlab.checks.base import CheckResult, Verdict, check
from qubitlab.sim import pulse_optimizer as po
from qubitlab.sim import transmon_dynamics as sim
from qubitlab.theory import qubit as th

A, D = Verdict.AGREE, Verdict.DIVERGE
LONG = 1e9  # ns; effectively no decoherence


@check("qubit.rabi_population")
def rabi_population() -> list[CheckResult]:
    out = []
    # (Omega, Delta, alpha) in GHz. Read P1 at its first maximum, t = 1 / (2 Omega_R).
    # Finding (first run predicted AGREE for the detuned case at alpha = -300 MHz): level 2
    # shifts the 0-1 resonance by about -Omega^2 / (2 alpha), 0.17 MHz here, which is 1.7% of
    # Delta. Adding that shift to Delta matches the simulation to 0.1%, and the error drops
    # 10x at alpha = -3 GHz. The two-level formula has no level 2, so it misses this.
    regimes = (
        (0.01, 0.0, -0.3, A),
        (0.01, 0.01, -0.3, D),
        (0.01, 0.01, -3.0, A),
        (0.1, 0.0, -0.2, D),
    )
    for omega, delta, alpha, expected in regimes:
        q = sim.TransmonQubit(alpha_ghz=alpha, t1_ns=LONG, t2_ns=LONG)
        t_peak = 1 / (2 * math.hypot(omega, delta))
        traj = sim.run("rabi", q, delta_ghz=delta, omega_ghz=omega, t_max_ns=t_peak, n_points=101)
        out.append(
            CheckResult(
                key=f"qubit.rabi_population.{omega:g}.{delta:g}.{alpha:g}",
                formula=th.rabi_population.key,
                regime=f"Ω = {omega * 1e3:g} MHz, Δ = {delta * 1e3:g} MHz, α = {alpha * 1e3:g} MHz",
                theory=float(th.rabi_population(Omega=omega, Delta=delta, t=t_peak)),
                sim=float(traj.populations[-1, 1]),
                rel_tol=0.01,
                expected=expected,
            )
        )
    return out


@check("qubit.t1_decay")
def t1_decay() -> list[CheckResult]:
    out = []
    for t1, t2 in ((100.0, 200.0), (100.0, 50.0)):
        q = sim.TransmonQubit(t1_ns=t1, t2_ns=t2)
        traj = sim.run("t1", q, t_max_ns=t1, n_points=11)
        out.append(
            CheckResult(
                key=f"qubit.t1_decay.{t2:g}",
                formula=th.t1_decay.key,
                regime=f"T1 = {t1:g} ns, T2 = {t2:g} ns, read at t = T1",
                theory=float(th.t1_decay(t=t1, T_1=t1)),
                sim=float(traj.populations[-1, 1]),
                rel_tol=0.01,
                expected=A,
            )
        )
    return out


@check("qubit.t2")
def t2() -> list[CheckResult]:
    out = []
    # The sim takes T2 as input and builds T_phi from it, so here we read T2 back from a
    # Ramsey trace and compare with the formula applied to the T_phi the sim used.
    for t1, t2_in in ((100.0, 200.0), (100.0, 100.0), (100.0, 30.0)):
        q = sim.TransmonQubit(t1_ns=t1, t2_ns=t2_in)
        traj = sim.run("ramsey", q, t_max_ns=50.0, n_points=11)
        t2_sim = -50.0 / math.log(traj.bloch[-1, 0])
        t_phi = 1 / q.gamma_phi if q.gamma_phi > 0 else math.inf
        out.append(
            CheckResult(
                key=f"qubit.t2.{t2_in:g}",
                formula=th.t2_from_t1_tphi.key,
                regime=f"T1 = {t1:g} ns, T_φ = {t_phi:g} ns",
                theory=1 / float(th.t2_from_t1_tphi(T_1=t1, T_phi=t_phi)),
                sim=t2_sim,
                rel_tol=0.01,
                expected=A,
            )
        )
    return out


@check("qubit.ramsey", slow=True)
def ramsey() -> list[CheckResult]:
    out = []
    for sigma, delta, t in ((0.0, 0.01, 130.0), (0.002, 0.0, 150.0), (0.002, 0.01, 120.0)):
        q = sim.TransmonQubit(t1_ns=20_000.0, t2_ns=10_000.0, sigma_ghz=sigma)
        traj = sim.run("ramsey", q, delta_ghz=delta, t_max_ns=t, n_points=11)
        out.append(
            CheckResult(
                key=f"qubit.ramsey.{sigma:g}.{delta:g}",
                formula=th.ramsey.key,
                regime=f"σ = {sigma * 1e3:g} MHz, Δ = {delta * 1e3:g} MHz, t = {t:g} ns",
                theory=float(th.ramsey(t=t, T_2=q.t2_ns, sigma=sigma, Delta=delta)),
                sim=float(traj.bloch[-1, 0]),
                rel_tol=0.02,
                expected=A,
            )
        )
    return out


@check("qubit.echo", slow=True)
def echo() -> list[CheckResult]:
    q = sim.TransmonQubit(t1_ns=400.0, t2_ns=300.0, sigma_ghz=0.005)
    t = 200.0
    traj = sim.run("echo", q, t_max_ns=t, n_points=2)
    return [
        CheckResult(
            key="qubit.echo.0.005",
            formula=th.echo.key,
            regime=f"σ = 5 MHz (Ramsey would be gone by {1 / (2 * math.pi * 0.005):.0f} ns)",
            theory=float(th.echo(t=t, T_2=q.t2_ns)),
            sim=float(traj.bloch[-1, 0]),
            rel_tol=0.01,
            expected=A,
        )
    ]


@check("qubit.idle_fidelity")
def idle_fidelity() -> list[CheckResult]:
    out = []
    for t1, t2, t in ((100.0, 150.0, 20.0), (100.0, 50.0, 100.0)):
        q = sim.TransmonQubit(t1_ns=t1, t2_ns=t2)
        out.append(
            CheckResult(
                key=f"qubit.idle_fidelity.{t2:g}.{t:g}",
                formula=th.idle_fidelity.key,
                regime=f"T1 = {t1:g} ns, T2 = {t2:g} ns, t = {t:g} ns",
                theory=float(th.idle_fidelity(t=t, T_1=t1, T_2=t2)),
                sim=sim.average_idle_fidelity(q, t),
                rel_tol=0.001,
                expected=A,
            )
        )
    return out


@check("qubit.flip_probability")
def flip_probability() -> list[CheckResult]:
    out = []
    # (T1, T2, sigma, t). Predicted: AGREE with white noise only. With 2 MHz of quasi-static
    # noise the X decay picks up a Gaussian factor exp(-(2 pi sigma t)^2 / 2) = 0.45 at
    # t = 100 ns, so the phase-flip probability DIVERGEs while the bit-flip one still AGREEs.
    regimes = ((100.0, 150.0, 0.0, 20.0, A, A), (20_000.0, 20_000.0, 0.002, 100.0, A, D))
    for t1, t2, sigma, t, exp_bit, exp_phase in regimes:
        q = sim.TransmonQubit(t1_ns=t1, t2_ns=t2, sigma_ghz=sigma)
        p_bit, p_phase = sim.flip_probabilities(q, t)
        regime = f"T1 = {t1:g} ns, T2 = {t2:g} ns, σ = {sigma * 1e3:g} MHz, t = {t:g} ns"
        for name, T, p, expected in (
            ("bit", t1, p_bit, exp_bit),
            ("phase", t2, p_phase, exp_phase),
        ):
            out.append(
                CheckResult(
                    key=f"qubit.flip_probability.{name}.{sigma:g}",
                    formula=th.flip_probability.key,
                    regime=f"{name}-flip, T = T{1 if name == 'bit' else 2}, {regime}",
                    theory=float(th.flip_probability(t=t, T=T)),
                    sim=p,
                    rel_tol=0.001,
                    expected=expected,
                )
            )
    return out


@check("qubit.detuned_pi_error")
def detuned_pi_error() -> list[CheckResult]:
    out = []
    # (pulse length T, alpha, Delta). Gaussian DRAG pi pulse made for Delta = 0, run at Delta.
    # Hand estimate: sigma = T/4 gives tau ~ 2 sigma = T/2, and peak Omega ~ 21 MHz at T = 40 ns.
    # Predicted before running:
    # - 40 ns, alpha = -3 GHz, 3 MHz: pi Delta tau ~ 0.19, next order ~ 1%; the level-2 Stark
    #   shift moves Delta by ~1.5%, so ~3% in the error. AGREE at 5%.
    # - 40 ns, alpha = -3 GHz, 15 MHz: pi Delta tau ~ 0.94; sin^2 vs square differ by ~25%. DIVERGE.
    # - 6 ns, alpha = -200 MHz, 3 MHz: the DRAG pulse alone has 8.7% error (docs/learn/32),
    #   far above the ~1e-3 from detuning. DIVERGE.
    # Finding (first run, at +Delta only): the first regime came out 5.1% low, a DIVERGE. At
    # -Delta it is 3.1% high. The pulse's own error at Delta = 0 (7.5e-6, from level 2) adds a
    # cross term odd in Delta, of relative size ~2 sqrt(7.5e-6 / 0.017) = 4% (hand calc step 8).
    # The formula is the even part, so the sim number is now the mean over +Delta and -Delta.
    # That mean was 1.0% low on the same run, so the first regime AGREEs after the change.
    regimes = ((40.0, -3.0, 0.003, A), (40.0, -3.0, 0.015, D), (6.0, -0.2, 0.003, D))
    for t_ns, alpha, delta, expected in regimes:
        q = sim.TransmonQubit(alpha_ghz=alpha)
        edges, ox, oy = po.drag_slices(q, t_ns, 4 * int(t_ns))
        phi = np.concatenate([[0.0], np.cumsum(2 * math.pi * ox * np.diff(edges))])
        tau = float(np.trapezoid(np.sin(phi), edges))
        error = sum(po.gate_error(q, edges, ox, oy, d)[0] for d in (delta, -delta)) / 2
        out.append(
            CheckResult(
                key=f"qubit.detuned_pi_error.{t_ns:g}.{alpha:g}.{delta:g}",
                formula=th.detuned_pi_error.key,
                regime=(
                    f"Gaussian DRAG π pulse, T = {t_ns:g} ns, α = {alpha * 1e3:g} MHz, "
                    f"Δ = ±{delta * 1e3:g} MHz, mean"
                ),
                theory=float(th.detuned_pi_error(Delta=delta, tau=tau)),
                sim=error,
                rel_tol=0.05,
                expected=expected,
            )
        )
    return out
