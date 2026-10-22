"""Cat qubit: closed-form formulas vs Dynamiqs.

Each regime states, before running, whether the formula should hold there. Inside its
validity range a formula must AGREE; outside it, we expect it to DIVERGE.
"""

from __future__ import annotations

import math

import dynamiqs as dq
import numpy as np

from qubitlab.checks.base import CheckResult, Verdict, check
from qubitlab.sim import cat as sim
from qubitlab.theory import cat as th

A, D = Verdict.AGREE, Verdict.DIVERGE


@check("cat.wigner")
def wigner() -> list[CheckResult]:
    out = []
    for alpha_sq in (1.0, 4.0):
        alpha = math.sqrt(alpha_sq)
        n = sim.CatParams(alpha_sq=alpha_sq).n_fock
        for kind, formula in (("even_cat", th.wigner_even_cat), ("odd_cat", th.wigner_odd_cat)):
            state = sim.initial_state(kind, alpha_sq, n)  # type: ignore[arg-type]
            for label, (xp, pp) in {"origin": (0.0, 0.0), "blob": (alpha, 0.0)}.items():
                _, _, w = dq.wigner(state, xvec=np.array([xp]), yvec=np.array([pp]))
                out.append(
                    CheckResult(
                        key=f"cat.wigner.{kind}.{label}",
                        formula=formula.key,
                        regime=f"|α|² = {alpha_sq:g}, W at ({xp:.2f}, {pp:.2f})",
                        theory=float(formula(x=xp, p=pp, alpha=alpha)),
                        sim=float(np.asarray(w).ravel()[0]),
                        rel_tol=1e-6,
                        expected=A,
                    )
                )
    return out


@check("cat.photon_number")
def photon_number() -> list[CheckResult]:
    out = []
    for alpha_sq in (0.5, 1.0, 4.0):
        n = sim.CatParams(alpha_sq=alpha_sq).n_fock
        for kind, formula in (
            ("even_cat", th.even_cat_photon_number),
            ("odd_cat", th.odd_cat_photon_number),
        ):
            state = sim.initial_state(kind, alpha_sq, n)  # type: ignore[arg-type]
            out.append(
                CheckResult(
                    key=f"cat.photon_number.{kind}.{alpha_sq:g}",
                    formula=formula.key,
                    regime=f"|α|² = {alpha_sq:g}",
                    theory=float(formula(alpha=math.sqrt(alpha_sq))),
                    sim=float(dq.expect(dq.number(n), state).real),
                    rel_tol=1e-6,
                    expected=A,
                )
            )
    return out


@check("cat.stabilized_photon_number", slow=True)
def stabilized_photon_number() -> list[CheckResult]:
    # (|α|², κ₁/κ₂, expected). Valid while kappa_1 << kappa_2 |α|².
    regimes = ((4.0, 0.01, A), (4.0, 0.1, A), (2.0, 0.2, A), (1.0, 0.5, D))
    out = []
    for alpha_sq, k1, expected in regimes:
        p = sim.CatParams(alpha_sq=alpha_sq, kappa_1=k1)
        rho = sim.steady_state(p)
        out.append(
            CheckResult(
                key=f"cat.stabilized_photon_number.{alpha_sq:g}.{k1:g}",
                formula=th.stabilized_photon_number.key,
                regime=f"|α|² = {alpha_sq:g}, κ₁/κ₂ = {k1:g}",
                theory=float(
                    th.stabilized_photon_number(alpha=math.sqrt(alpha_sq), kappa_1=k1, kappa_2=1.0)
                ),
                sim=float(dq.expect(dq.number(p.n_fock), rho).real),
                rel_tol=0.02,
                expected=expected,
            )
        )
    return out


@check("cat.parity_decay", slow=True)
def parity_decay() -> list[CheckResult]:
    # Needs a well-formed cat: valid for |α|² >~ 1, fails for small cats.
    regimes = ((2.0, A), (4.0, A), (8.0, A), (0.3, D))
    k1 = 0.01
    out = []
    for alpha_sq, expected in regimes:
        rates = sim.liouvillian_rates(sim.CatParams(alpha_sq=alpha_sq, kappa_1=k1))
        out.append(
            CheckResult(
                key=f"cat.parity_decay.{alpha_sq:g}",
                formula=th.parity_decay_rate.key,
                regime=f"|α|² = {alpha_sq:g}, κ₁/κ₂ = {k1:g}",
                theory=float(th.parity_decay_rate(kappa_1=k1, alpha=math.sqrt(alpha_sq))),
                sim=rates.x_decay,
                rel_tol=0.01,
                expected=expected,
            )
        )
    return out


@check("cat.parity_decay_exact", slow=True)
def parity_decay_exact() -> list[CheckResult]:
    # The coth form keeps the small-cat physics the large-cat law drops. It still assumes the
    # state never leaves the two-cat space, so strong loss (kappa_1 ~ kappa_2) should break it.
    regimes = ((0.3, 0.01, A), (1.0, 0.01, A), (4.0, 0.01, A), (1.0, 0.5, D))
    out = []
    for alpha_sq, k1, expected in regimes:
        rates = sim.liouvillian_rates(sim.CatParams(alpha_sq=alpha_sq, kappa_1=k1))
        out.append(
            CheckResult(
                key=f"cat.parity_decay_exact.{alpha_sq:g}.{k1:g}",
                formula=th.parity_decay_rate_exact.key,
                regime=f"|α|² = {alpha_sq:g}, κ₁/κ₂ = {k1:g}",
                theory=float(th.parity_decay_rate_exact(kappa_1=k1, alpha=math.sqrt(alpha_sq))),
                sim=rates.x_decay,
                rel_tol=0.01,
                expected=expected,
            )
        )
    return out


@check("cat.bit_flip_suppression", slow=True)
def bit_flip_suppression() -> list[CheckResult]:
    # The e^{-2} per photon law is asymptotic: holds for big cats, not small ones.
    regimes = ((5.0, A), (6.0, A), (1.0, D))
    k1 = 0.1
    out = []
    for alpha_sq, expected in regimes:
        lo = sim.liouvillian_rates(sim.CatParams(alpha_sq=alpha_sq, kappa_1=k1))
        hi = sim.liouvillian_rates(sim.CatParams(alpha_sq=alpha_sq + 1, kappa_1=k1))
        out.append(
            CheckResult(
                key=f"cat.bit_flip_suppression.{alpha_sq:g}",
                formula=th.bit_flip_suppression.key,
                regime=f"|α|² = {alpha_sq:g} -> {alpha_sq + 1:g}, κ₁/κ₂ = {k1:g}",
                theory=float(th.bit_flip_suppression(Delta_n=1.0)),
                sim=hi.z_decay / lo.z_decay,
                rel_tol=0.10,
                expected=expected,
            )
        )
    return out


@check("cat.bit_flip_dephasing", slow=True)
def bit_flip_dephasing() -> list[CheckResult]:
    # Perturbative in kappa_phi / kappa_2: agreement loosens as the cat grows.
    k_phi = 0.01
    out = []
    for alpha_sq in (1.0, 2.0, 4.0):
        rates = sim.liouvillian_rates(sim.CatParams(alpha_sq=alpha_sq, kappa_phi=k_phi))
        out.append(
            CheckResult(
                key=f"cat.bit_flip_dephasing.{alpha_sq:g}",
                formula=th.bit_flip_rate_dephasing.key,
                regime=f"|α|² = {alpha_sq:g}, κ_φ/κ₂ = {k_phi:g}, κ₁ = 0",
                theory=float(
                    th.bit_flip_rate_dephasing(kappa_phi=k_phi, alpha=math.sqrt(alpha_sq))
                ),
                sim=rates.bit_flip_rate,
                rel_tol=0.10,
                expected=A,
            )
        )
    return out
