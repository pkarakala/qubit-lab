"""Chip layer: closed forms for capacitors, CPW resonators and couplings vs Palace solves.

The Palace numbers come from data/chip.json (scripts/palace/run.py). Where a formula needs a
capacitance of the actual layout, it takes the one Palace computed, so each check tests one
step of the path from geometry to Hamiltonian.

Findings from the first runs:
- The first device mesh sampled its size field too sparsely along the 8 mm line, and order 1
  elements put the line capacitance 24% above order 2. Zero-thickness metal makes the field
  singular at the edges, so the error shrinks slowly; the straight CPW at orders 2 and 3
  (2.5% then 1.1% above Goeppl) shows the convergence.
- The qubit mode at 12 nH and the coupling g from the avoided crossing agreed to 1.1% and
  0.1%, inside the predicted 2% and 10%.
"""

from __future__ import annotations

from qubitlab.checks.base import CheckResult, Verdict, check
from qubitlab.sim import chip as sim
from qubitlab.theory import chip as th

A, D = Verdict.AGREE, Verdict.DIVERGE
K = th.CONSTANTS
UM = 1e-6


def _eps_eff() -> float:
    return float(th.cpw_effective_permittivity(epsilon_r=sim.eps_r()))


def _c_per_length() -> float:
    geo = sim.geometry()
    return float(
        th.cpw_capacitance_per_length(
            epsilon_0=K["epsilon_0"], epsilon_eff=_eps_eff(), w=geo["w"] * UM, s=geo["s"] * UM
        )
    )


@check("chip.parallel_plate")
def parallel_plate() -> list[CheckResult]:
    # Fringing grows as the gap approaches the plate side: about 5% at side/gap = 100,
    # comparable to C itself at side/gap = 4.
    out = []
    for case in sim.plates():
        side, gap = case["side_um"], case["gap_um"]
        ratio = side / gap
        out.append(
            CheckResult(
                key=f"chip.parallel_plate.{ratio:g}",
                formula=th.parallel_plate.key,
                regime=f"square plates, side/gap = {ratio:g} ({side:g} um, {gap:g} um), vacuum",
                theory=float(
                    th.parallel_plate(
                        epsilon_0=K["epsilon_0"], epsilon_r=1.0, A=(side * UM) ** 2, d=gap * UM
                    )
                ),
                sim=sim.plate_capacitance(side),
                rel_tol=0.08,
                expected=A if ratio >= 50 else D,
            )
        )
    return out


@check("chip.cpw_capacitance_per_length")
def cpw_capacitance_per_length() -> list[CheckResult]:
    # A straight line at two lengths, so the end fields cancel. Zero-thickness metal makes the
    # field singular at the strip edges, so the finite-element value comes down with the order.
    geo = sim.geometry()
    return [
        CheckResult(
            key=f"chip.cpw_capacitance_per_length.order{order}",
            formula=th.cpw_capacitance_per_length.key,
            regime=f"w = {geo['w']:g} um, s = {geo['s']:g} um on silicon, "
            f"Palace elements of order {order}",
            theory=_c_per_length(),
            sim=sim.cpw_capacitance_per_length(order),
            rel_tol=0.05,
            expected=A,
        )
        for order in sim.cpw_orders()
    ]


@check("chip.half_wave_frequency")
def half_wave_frequency() -> list[CheckResult]:
    # The resonator-like mode with the qubit far detuned. The coupler bar and the pad load
    # its open end and pull it down by a few percent.
    geo = sim.geometry()
    _, _, f_res = sim.gallery_modes()
    return [
        CheckResult(
            key="chip.half_wave_frequency.device",
            formula=th.half_wave_frequency.key,
            regime=f"l = {geo['length'] / 1000:g} mm, qubit far detuned",
            theory=float(
                th.half_wave_frequency(c=K["c"], epsilon_eff=_eps_eff(), l=geo["length"] * UM)
            ),
            sim=f_res,
            rel_tol=0.05,
            expected=A,
        )
    ]


@check("chip.harmonic_frequency")
def harmonic_frequency() -> list[CheckResult]:
    # The junction as a linear inductor with Palace's C_Sigma, against the qubit-like mode.
    L_J, f_q, _ = sim.gallery_modes()
    return [
        CheckResult(
            key="chip.harmonic_frequency.device",
            formula=th.harmonic_frequency.key,
            regime=f"L_J = {L_J * 1e9:g} nH, C_Sigma from the electrostatic solve",
            theory=float(th.harmonic_frequency(L_J=L_J, C_Sigma=sim.capacitances().C_sigma)),
            sim=f_q,
            rel_tol=0.02,
            expected=A,
        )
    ]


@check("chip.coupling_g")
def coupling_g() -> list[CheckResult]:
    # Half the smallest splitting of the L_J sweep, against the capacitance formula with
    # C_r = C_line / 2 (goeppl2008 Eq. (12) with C_l l from Palace) at f_q = f_r.
    caps, cross = sim.capacitances(), sim.crossing()
    C_r = float(
        th.half_wave_capacitance(
            C_l=caps.C_line / (sim.geometry()["length"] * UM), l=sim.geometry()["length"] * UM
        )
    )
    return [
        CheckResult(
            key="chip.coupling_g.device",
            formula=th.coupling_g.key,
            regime=f"at resonance, f_r = {cross.f_r / 1e9:.2f} GHz, C_g = {caps.C_g * 1e15:.1f} fF",
            theory=float(
                th.coupling_g(
                    C_g=caps.C_g, f_q=cross.f_r, f_r=cross.f_r, C_Sigma=caps.C_sigma, C_r=C_r
                )
            ),
            sim=cross.g,
            rel_tol=0.10,
            expected=A,
        )
    ]
