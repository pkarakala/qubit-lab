"""Closed forms that turn chip geometry into circuit parameters.

Conventions: SI units throughout (metres, farads, henries, hertz). Energies E_C and E_J are
returned as frequencies, E / h, in hertz. Physical constants are arguments, so the formulas
print as in the papers; `CONSTANTS` holds their CODATA values from SciPy.
The geometry these feed comes from Palace solves (`qubitlab.sim.chip`), compared in
`qubitlab.checks.chip_checks`.
"""

from __future__ import annotations

import scipy.constants as sc
import sympy as sp

from qubitlab.provenance import Kind, Source
from qubitlab.theory.registry import Formula

eps0, c0, e, h = sp.symbols("epsilon_0 c e h", positive=True)
eps_r, eps_eff, area, d, w, s, length = sp.symbols("epsilon_r epsilon_eff A d w s l", positive=True)
C_l, C_S, C_g, C_r, L_J, f_q, f_r = sp.symbols("C_l C_Sigma C_g C_r L_J f_q f_r", positive=True)

CONSTANTS = {"epsilon_0": sc.epsilon_0, "c": sc.c, "e": sc.e, "h": sc.h}


def hand(step: str) -> Source:
    return Source("qubitlab_hand_calcs", Kind.DERIVED, "docs/hand_calcs/chip.py", step)


def goeppl(locator: str) -> Source:
    return Source(
        "goeppl2008",
        Kind.PEER_REVIEWED,
        locator,
        "Equation numbers from arXiv:0807.4094v1.",
    )


def koch(locator: str) -> Source:
    return Source("koch2007", Kind.PEER_REVIEWED, locator)


parallel_plate = Formula(
    key="chip.parallel_plate",
    title="Parallel-plate capacitance",
    lhs=r"C",
    expr=eps0 * eps_r * area / d,
    args=(eps0, eps_r, area, d),
    sources=(hand("step 1"),),
    valid_when="Gap d much smaller than the plate side; fringing fields add capacitance otherwise.",
)

cpw_effective_permittivity = Formula(
    key="chip.cpw_effective_permittivity",
    title="Effective permittivity of a coplanar waveguide on a thick substrate",
    lhs=r"\epsilon_{\mathrm{eff}}",
    expr=(1 + eps_r) / 2,
    args=(eps_r,),
    sources=(hand("step 2"), goeppl("Eq. (3), definition of epsilon_eff")),
    valid_when="Thin conductors on a substrate much thicker than w + 2s, vacuum above.",
)

# k0 = w/(w+2s) (Eq. 4) and k0' = sqrt(1 - k0^2) (Eq. 5); SymPy's elliptic_k takes m = k^2.
_k0 = w / (w + 2 * s)
cpw_capacitance_per_length = Formula(
    key="chip.cpw_capacitance_per_length",
    title="Capacitance per length of a coplanar waveguide",
    lhs=r"C_\ell",
    expr=4 * eps0 * eps_eff * sp.elliptic_k(_k0**2) / sp.elliptic_k(1 - _k0**2),
    args=(eps0, eps_eff, w, s),
    sources=(goeppl("Eqs. (3)-(5)"),),
    valid_when="Quasi-static field; ground planes much wider than the gap s.",
)

half_wave_frequency = Formula(
    key="chip.half_wave_frequency",
    title="Fundamental frequency of a half-wave CPW resonator",
    lhs=r"f_0",
    expr=c0 / sp.sqrt(eps_eff) / (2 * length),
    args=(c0, eps_eff, length),
    sources=(goeppl("Eq. (1)"),),
    valid_when="Open ends with no load; the end fringing fields lengthen the line slightly.",
)

half_wave_capacitance = Formula(
    key="chip.half_wave_capacitance",
    title="Lumped capacitance of a half-wave resonator near its fundamental",
    lhs=r"C_r",
    expr=C_l * length / 2,
    args=(C_l, length),
    sources=(goeppl("Eq. (12)"),),
    valid_when="Near the fundamental mode, n = 1.",
)

charging_energy = Formula(
    key="chip.charging_energy",
    title="Charging energy from the total capacitance",
    lhs=r"E_C/h",
    expr=e**2 / (2 * C_S) / h,
    args=(e, C_S, h),
    sources=(koch("text after Eq. (2.1)"),),
    valid_when="C_Sigma is everything the island sees: junction, shunt and couplers.",
)

josephson_energy = Formula(
    key="chip.josephson_energy",
    title="Josephson energy from the junction inductance",
    lhs=r"E_J/h",
    expr=(h / (2 * sp.pi * 2 * e)) ** 2 / L_J / h,
    args=(h, e, L_J),
    sources=(hand("step 3"),),
    valid_when="Small phase across the junction, as in the transmon ground state.",
)

harmonic_frequency = Formula(
    key="chip.harmonic_frequency",
    title="Harmonic frequency of the junction and its capacitance",
    lhs=r"f_p",
    expr=1 / (2 * sp.pi * sp.sqrt(L_J * C_S)),
    args=(L_J, C_S),
    sources=(hand("step 4"), koch("Eq. (2.11), leading term")),
    valid_when="Linear junction; the qubit's own transition sits about E_C lower.",
)

coupling_g = Formula(
    key="chip.coupling_g",
    title="Qubit-resonator coupling through a capacitance",
    lhs=r"g/2\pi",
    expr=C_g / 2 * sp.sqrt(f_q * f_r / (C_S * C_r)),
    args=(C_g, f_q, f_r, C_S, C_r),
    sources=(hand("step 5"), koch("Eqs. (3.1), (3.3), (3.4)")),
    valid_when="C_g much smaller than C_Sigma and C_r; E_J/E_C large; f_q near f_r.",
)

coupling_g_n = Formula(
    key="chip.coupling_g_n",
    title="Coupling to the charge operator, from the capacitances",
    lhs=r"g_n/2\pi",
    expr=2 * (C_g / C_S) * e * sp.sqrt(h * f_r / (2 * C_r)) / h,
    args=(C_g, C_S, e, h, f_r, C_r),
    sources=(koch("Eq. (3.1), the term 2 beta e V_rms n (a + a+)"),),
    valid_when="C_g much smaller than C_Sigma and C_r. No E_J/E_C limit: the matrix elements of n "
    "are left to the transmon spectrum.",
)

ALL = (
    parallel_plate,
    cpw_effective_permittivity,
    cpw_capacitance_per_length,
    half_wave_frequency,
    half_wave_capacitance,
    charging_energy,
    josephson_energy,
    harmonic_frequency,
    coupling_g,
    coupling_g_n,
)
