"""Closed forms for dispersive readout: a qubit pulling the frequency of a resonator.

Conventions (koch2007, gambetta2006, blais2021): the resonator shifts by +-chi with the qubit
state, Delta = qubit minus resonator frequency, alpha = anharmonicity (negative for a
transmon), kappa = the resonator's photon decay rate, eps = drive amplitude, H_d = eps (a + a+).
Frequencies are in any one unit (GHz or rad/us), as long as all of them match.
"""

from __future__ import annotations

import sympy as sp

from qubitlab.provenance import Kind, Source
from qubitlab.theory.registry import Formula

g, g_12, Delta, alpha, kappa, chi, eps, delta_r = sp.symbols(
    "g g_12 Delta alpha kappa chi epsilon delta_r", real=True
)
eta, tau, snr_t, tau_f = sp.symbols("eta tau SNR_tau tau_f", positive=True)
sep = sp.Symbol("D", positive=True)  # |alpha_e - alpha_g|


def hand(step: str) -> Source:
    return Source("qubitlab_hand_calcs", Kind.DERIVED, "docs/hand_calcs/readout.py", step)


chi_two_level = Formula(
    key="readout.chi_two_level",
    title="Dispersive shift of a two-level qubit",
    lhs=r"\chi_{2\mathrm{lvl}}",
    expr=g**2 / Delta,
    args=(g, Delta),
    sources=(
        Source(
            "blais2004",
            Kind.PEER_REVIEWED,
            "Eq. (12)",
            "Written as the cavity shift sigma_z g^2/Delta; the symbol chi is not used there.",
        ),
    ),
    valid_when="Two levels only, g << |Delta|. A transmon's third level changes it a lot.",
)

chi_transmon = Formula(
    key="readout.chi_transmon",
    title="Dispersive shift of a transmon, from its 0-1 and 1-2 couplings",
    lhs=r"\chi",
    expr=sp.Add(g**2 / Delta, -(g_12**2) / (2 * (Delta + alpha)), evaluate=False),
    args=(g, g_12, Delta, alpha),
    sources=(
        Source(
            "koch2007",
            Kind.PEER_REVIEWED,
            "Eqs. (3.9), (3.10)",
            "chi = chi_01 - chi_12/2, chi_ij = g_ij^2/(omega_ij - omega_r).",
        ),
    ),
    valid_when=(
        "Rotating-wave coupling, levels 0 to 2, g << |Delta|, |Delta + alpha|. The neglected "
        "counter-rotating terms matter more as |Delta| grows, because chi falls as 1/Delta^2."
    ),
)

chi_transmon_harmonic = Formula(
    key="readout.chi_transmon_harmonic",
    title="Dispersive shift of a transmon, with g_12 = sqrt(2) g_01",
    lhs=r"\chi_{\sqrt 2}",
    expr=g**2 * alpha / (Delta * (Delta + alpha)),
    args=(g, Delta, alpha),
    sources=(
        Source("koch2007", Kind.PEER_REVIEWED, "Eqs. (3.4), (3.9), (3.10)"),
        Source("blais2021", Kind.PEER_REVIEWED, "Eq. (43), arXiv v1", "alpha = -E_C there."),
        hand("Step 1: chi_01 - chi_12/2 with g_12 = sqrt(2) g."),
    ),
    valid_when=(
        "E_J/E_C -> infinity, where g_12/g_01 -> sqrt(2). At E_J/E_C = 50 to 200 the ratio is "
        "1.38 to 1.39, and the two terms of chi nearly cancel, so the shortcut can miss by tens "
        "of percent."
    ),
)

pointer_photons = Formula(
    key="readout.pointer_photons",
    title="Resonator photons in steady state, for the qubit in |e> (+) or |g> (-)",
    lhs=r"\bar n_\pm",
    expr=eps**2 / (kappa**2 / 4 + (delta_r + chi) ** 2),
    args=(eps, kappa, delta_r, chi),
    sources=(
        Source(
            "gambetta2006",
            Kind.PEER_REVIEWED,
            "Eq. (5.21)",
            "Pass chi with its sign: +chi for the excited state, -chi for the ground state.",
        ),
    ),
    valid_when="Linear dispersive model: n well below the critical number Delta^2 / 4g^2.",
)

dephasing_rate = Formula(
    key="readout.dephasing_rate",
    title="Measurement-induced dephasing, drive at the bare resonator",
    lhs=r"\Gamma_m",
    expr=2 * kappa * chi**2 * eps**2 / (kappa**2 / 4 + chi**2) ** 2,
    args=(kappa, chi, eps),
    sources=(
        Source("gambetta2006", Kind.PEER_REVIEWED, "Eqs. (5.20), (5.21)"),
        hand("Step 3: Eq. (5.20) at Delta_r = 0."),
    ),
    valid_when="Steady state of the linear dispersive model, any chi / kappa.",
)

dephasing_rate_weak = Formula(
    key="readout.dephasing_rate_weak",
    title="Measurement-induced dephasing for a small dispersive shift",
    lhs=r"\Gamma_m^{(\chi \ll \kappa)}",
    expr=32 * chi**2 * eps**2 / kappa**3,
    args=(kappa, chi, eps),
    sources=(
        Source(
            "gambetta2006",
            Kind.PEER_REVIEWED,
            "Sec. V.A, text after Eq. (5.9)",
            "2 kappa n theta_0^2 with theta_0 = 2 chi / kappa and n = 4 eps^2 / kappa^2.",
        ),
        Source("blais2004", Kind.PEER_REVIEWED, "Eq. (26)"),
        hand("Step 3: leading order in chi."),
    ),
    valid_when="chi << kappa. At chi = kappa/2 it is 4x too high: 2x in the rate, 2x in n.",
)

pointer_separation = Formula(
    key="readout.pointer_separation",
    title="Distance between the two pointer states in steady state",
    lhs=r"|\alpha_e - \alpha_g|",
    expr=2
    * eps
    * chi
    / sp.sqrt((kappa**2 / 4 + (delta_r + chi) ** 2) * (kappa**2 / 4 + (delta_r - chi) ** 2)),
    args=(eps, chi, kappa, delta_r),
    sources=(
        Source(
            "gambetta2006",
            Kind.PEER_REVIEWED,
            "Eq. (5.21)",
            "Gives |alpha|^2 for each state; the separation needs the phases too.",
        ),
        hand(
            "Step 4: steady state of d alpha/dt = -i(delta_r +- chi) alpha - kappa alpha/2 - i eps."
        ),
    ),
    valid_when="Steady state (tau >> 1/kappa), linear dispersive model, chi > 0.",
)

heterodyne_snr = Formula(
    key="readout.heterodyne_snr",
    title="Signal-to-noise of a box-car heterodyne readout, in Gambetta's convention",
    lhs=r"\mathrm{SNR}\,\tau_f",
    expr=eta * kappa * tau * sep**2 / 2,
    args=(eta, kappa, tau, sep),
    sources=(
        Source(
            "gambetta2007",
            Kind.PEER_REVIEWED,
            "Eqs. (2.1)-(2.3)",
            "Signal +-1 per unit time, noise variance tau_f / SNR. Here D = |alpha_e - alpha_g|.",
        ),
        hand("Step 5: each heterodyne arm records sqrt(2 eta kappa) <a> dt plus unit white noise."),
    ),
    valid_when="Steady-state fields over the whole window: tau >> 1/kappa.",
)

fidelity = Formula(
    key="readout.fidelity",
    title="Readout fidelity 1 - P(e|g) - P(g|e) with no relaxation",
    lhs=r"F",
    expr=sp.erf(sp.sqrt(snr_t / 2)),
    args=(snr_t,),
    sources=(
        Source(
            "gambetta2007",
            Kind.PEER_REVIEWED,
            "Eqs. (2.5), (2.6)",
            "SNR_tau is the product tau_f SNR there: Gaussian clouds, threshold halfway.",
        ),
    ),
    valid_when="No qubit relaxation during the window (tau << T1).",
)

fidelity_t1 = Formula(
    key="readout.fidelity_t1",
    title="Readout fidelity when the qubit can relax during the window",
    lhs=r"F_0",
    expr=sp.exp(-tau_f / 2) * sp.erf(sp.sqrt(snr_t / 2)),
    args=(tau_f, snr_t),
    sources=(
        Source(
            "gambetta2007",
            Kind.PEER_REVIEWED,
            "Eq. (3.8)",
            "Box-car filter, threshold at zero, tau_f = tau / T1. A decay in the first half of the "
            "window lands on the wrong side.",
        ),
    ),
    valid_when="Large SNR, so the clouds do not overlap and only decays cause errors.",
)

ALL = (
    chi_two_level,
    chi_transmon,
    chi_transmon_harmonic,
    pointer_photons,
    dephasing_rate,
    dephasing_rate_weak,
    pointer_separation,
    heterodyne_snr,
    fidelity,
    fidelity_t1,
)
