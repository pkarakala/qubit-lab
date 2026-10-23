"""Two-qubit gates: closed forms vs Dynamiqs.

Predictions were written before the runs. Findings:
- The cat CNOT at T = 0.5 / kappa_2 (about 2 / (kappa_2 |alpha|^2)) was predicted to DIVERGE,
  because the papers assume a slow gate. It agrees to 4%: the non-adiabatic term holds there.
  At that time Chamberland's pi^2/64 fits the simulation slightly better than Guillaud's 1/(2 pi).
- With 14 Fock states per mode the control flip came out 70% high; 18 and 22 agree to 1e-3.
"""

from __future__ import annotations

import math

from qubitlab.checks.base import CheckResult, Verdict, check
from qubitlab.sim import two_qubit as sim
from qubitlab.theory import gates as th

A, D = Verdict.AGREE, Verdict.DIVERGE
TWO_PI = 2 * math.pi


@check("gates.static_zz")
def static_zz() -> list[CheckResult]:
    out = []
    w2, alpha, g = 5.0, -0.3, 0.01  # GHz
    for delta, expected in ((0.7, A), (0.32, D)):  # 0.32: |Delta + alpha_1| = 2 g
        out.append(
            CheckResult(
                key=f"gates.static_zz.{delta:g}",
                formula=th.static_zz.key,
                regime=(
                    f"Delta = {delta * 1e3:g} MHz, alpha = {alpha * 1e3:g} MHz, g = {g * 1e3:g} MHz"
                ),
                theory=float(th.static_zz(g=g, Delta=delta, alpha_1=alpha, alpha_2=alpha)),
                sim=sim.static_zz(w2 + delta, w2, alpha, alpha, g),
                rel_tol=0.05,
                expected=expected,
            )
        )
    return out


@check("gates.cz_time")
def cz_time() -> list[CheckResult]:
    out = []
    alpha = -0.3  # GHz
    for g, expected in ((0.005, A), (0.05, D)):
        t = float(th.cz_time(g=TWO_PI * g))
        r = sim.cz(TWO_PI * 5.0, TWO_PI * alpha, TWO_PI * alpha, TWO_PI * g, t)
        out.append(
            CheckResult(
                key=f"gates.cz_time.{g:g}",
                formula=th.cz_time.key,
                regime=f"g = {g * 1e3:g} MHz, alpha = {alpha * 1e3:g} MHz, phase at t_CZ",
                theory=math.pi,
                sim=r.conditional_phase,
                rel_tol=0.05,
                expected=expected,
            )
        )
    return out


@check("gates.cat_cnot", slow=True)
def cat_cnot() -> list[CheckResult]:
    out = []
    n, k1 = 4.0, 1e-3
    t_star = 1 / (2 * n * math.sqrt(math.pi * k1))  # docs/hand_calcs/two_qubit_gates.py step 3
    for T in (0.5, t_star, 10.0):
        r = sim.cat_cnot(n, k1, T)
        z1 = float(th.cnot_z1(n=n, kappa_1=k1, kappa_2=1.0, T=T))
        z2 = float(th.cnot_z2(n=n, kappa_1=k1, T=T))
        regime = f"|alpha|^2 = {n:g}, kappa_1/kappa_2 = {k1:g}, kappa_2 T = {T:.3g}"
        out.append(
            CheckResult(
                key=f"gates.cat_cnot.control.{T:.3g}",
                formula=th.cnot_z1.key,
                regime=f"control parity flip, p_Z1 + p_Z1Z2; {regime}",
                theory=z1 + z2,
                sim=r.control_flip,
                rel_tol=0.1,
                expected=A,
            )
        )
        out.append(
            CheckResult(
                key=f"gates.cat_cnot.target.{T:.3g}",
                formula=th.cnot_z2.key,
                regime=f"target parity flip, p_Z2 + p_Z1Z2; {regime}",
                theory=2 * z2,
                sim=r.target_flip,
                rel_tol=0.1,
                expected=A,
            )
        )
    return out
