"""Closed-form results for a driven, decaying qubit (levels 0 and 1 of a transmon).

Conventions (shared with `qubitlab.sim.transmon_dynamics`):
- Frequencies in cycles per unit time (GHz with time in ns). Rabi frequency Omega, detuning Delta.
- T1: energy relaxation. T_phi: pure dephasing. T2: coherence time.
- The drive frame rotates at the drive frequency; Delta = qubit minus drive frequency.

These are two-level results. The simulation keeps more levels, so the checks show where the
extra levels matter (fast pulses leak to level 2).
"""

from __future__ import annotations

import sympy as sp

from qubitlab.provenance import Kind, Source
from qubitlab.theory.registry import Formula

t, Omega, Delta, T_1, T_2, T_phi, sigma, alpha = sp.symbols(
    "t Omega Delta T_1 T_2 T_phi sigma alpha", real=True
)
Omega_dot = sp.Symbol("Omegadot_x", real=True)  # prints as \dot{\Omega}_x


def hand(step: str) -> Source:
    return Source("qubitlab_hand_calcs", Kind.DERIVED, "docs/hand_calcs/qubit_decoherence.py", step)


rabi_population = Formula(
    key="qubit.rabi_population",
    title="Rabi oscillation: population of |1> under a constant drive",
    lhs=r"P_1(t)",
    expr=sp.Mul(
        Omega**2 / (Omega**2 + Delta**2),
        sp.sin(sp.pi * sp.sqrt(Omega**2 + Delta**2) * t) ** 2,
        evaluate=False,
    ),
    args=(Omega, Delta, t),
    sources=(hand("Step 1: two-level Schrodinger equation in the drive frame."),),
    valid_when="Drive much weaker than |alpha|, so level 2 stays empty.",
)

t1_decay = Formula(
    key="qubit.t1_decay",
    title="Energy relaxation from |1>",
    lhs=r"P_1(t)",
    expr=sp.exp(-t / T_1),
    args=(t, T_1),
    sources=(hand("Step 2: Lindblad equation with jump sqrt(1/T1) |0><1|."),),
    valid_when="Zero temperature, no drive.",
)

t2_from_t1_tphi = Formula(
    key="qubit.t2",
    title="Coherence time from relaxation and pure dephasing",
    lhs=r"\frac{1}{T_2}",
    expr=sp.Add(1 / (2 * T_1), 1 / T_phi, evaluate=False),
    args=(T_1, T_phi),
    sources=(hand("Step 3: decay of rho_01 under both jumps."),),
    valid_when="Markovian (white) noise. So T2 can never exceed 2 T1.",
)

ramsey = Formula(
    key="qubit.ramsey",
    title="Ramsey fringe with quasi-static frequency noise",
    lhs=r"\langle X\rangle(t)",
    expr=sp.Mul(
        sp.exp(-t / T_2),
        sp.exp(-((2 * sp.pi * sigma * t) ** 2) / 2),
        sp.cos(2 * sp.pi * Delta * t),
        evaluate=False,
    ),
    args=(t, T_2, sigma, Delta),
    sources=(hand("Step 4: Gaussian average of the fringe over a static detuning."),),
    valid_when="Noise slower than one shot; sigma is its spread in frequency.",
)

echo = Formula(
    key="qubit.echo",
    title="Hahn echo: a pi pulse at t/2 cancels static detuning",
    lhs=r"\langle X\rangle(t)",
    expr=sp.exp(-t / T_2),
    args=(t, T_2),
    sources=(hand("Step 5: the pi pulse reverses the phase from any static Delta."),),
    valid_when="Noise static over one sequence; ideal, instantaneous pi pulse.",
)

drag_quadrature = Formula(
    key="qubit.drag_quadrature",
    title="DRAG: quadrature drive that cancels leakage to level 2",
    lhs=r"\Omega_y(t)",
    expr=-Omega_dot / (2 * sp.pi * alpha),
    args=(Omega_dot, alpha),
    sources=(
        Source(
            "motzoi2009",
            Kind.PEER_REVIEWED,
            "Eq. (9)",
            "Paper writes E^y = -dE^x/dt / Delta with Delta = 2 pi alpha in angular units.",
        ),
    ),
    valid_when="First order in 1/(alpha * pulse length). Ignores the AC Stark shift.",
)

idle_fidelity = Formula(
    key="qubit.idle_fidelity",
    title="Average fidelity of doing nothing for time t",
    lhs=r"\bar F(t)",
    expr=sp.Add(
        sp.Rational(1, 2),
        sp.exp(-t / T_2) / 3,
        sp.exp(-t / T_1) / 6,
        evaluate=False,
    ),
    args=(t, T_1, T_2),
    sources=(
        Source("nielsen2002", Kind.PEER_REVIEWED, "Eq. (3)", "F_avg = (d F_e + 1)/(d + 1)."),
        hand("Step 6: entanglement fidelity of the T1/T2 channel, d = 2."),
    ),
    valid_when="Two levels, Markovian noise. Sets the error per idle gate.",
)

ALL = (
    rabi_population,
    t1_decay,
    t2_from_t1_tphi,
    ramsey,
    echo,
    drag_quadrature,
    idle_fidelity,
)
