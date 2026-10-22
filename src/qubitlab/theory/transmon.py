"""Closed-form transmon formulas (Koch et al. 2007).

Conventions (shared with `qubitlab.sim.transmon`):
- H = 4 E_C (n - n_g)^2 - E_J cos(phi), energies in any one unit (GHz in the explorer).
- Levels m = 0, 1, 2, ... ; E_mn = E_n - E_m.
- The formulas hold in the transmon regime E_J / E_C >~ 20.
"""

from __future__ import annotations

import sympy as sp

from qubitlab.provenance import Kind, Source
from qubitlab.theory.registry import Formula

E_J, E_C = sp.symbols("E_J E_C", positive=True)
m = sp.Symbol("m", nonnegative=True, integer=True)

# sqrt(8 E_J E_C), kept unevaluated so it prints as in the paper rather than 2 sqrt(2) ...
SQRT_8EJEC = sp.Pow(sp.Mul(8, E_J, E_C, evaluate=False), sp.Rational(1, 2), evaluate=False)


def koch(locator: str) -> Source:
    return Source("koch2007", Kind.PEER_REVIEWED, locator)


plasma_frequency = Formula(
    key="transmon.plasma_frequency",
    title="Josephson plasma energy",
    lhs=r"\hbar\omega_p",
    expr=SQRT_8EJEC,
    args=(E_J, E_C),
    sources=(koch("Eq. (2.11), leading term"),),
    valid_when="Small-phase expansion of the cosine.",
)

level_energy = Formula(
    key="transmon.level_energy",
    title="Energy of level m (Duffing oscillator approximation)",
    lhs=r"E_m",
    expr=sp.Add(
        -E_J,
        sp.Mul(SQRT_8EJEC, m + sp.Rational(1, 2), evaluate=False),
        -sp.Mul(E_C / 12, 6 * m**2 + 6 * m + 3, evaluate=False),
        evaluate=False,
    ),
    args=(m, E_J, E_C),
    sources=(koch("Eq. (2.11)"),),
    valid_when="E_J / E_C >> 1, low levels m.",
)

qubit_frequency = Formula(
    key="transmon.qubit_frequency",
    title="Qubit transition energy E_01",
    lhs=r"E_{01}",
    expr=SQRT_8EJEC - E_C,
    args=(E_J, E_C),
    sources=(koch("Eq. (2.11), m=1 minus m=0"),),
    valid_when="E_J / E_C >> 1.",
)

anharmonicity = Formula(
    key="transmon.anharmonicity",
    title="Anharmonicity E_12 - E_01",
    lhs=r"\alpha_{\mathrm{tr}}",
    expr=-E_C,
    args=(E_C,),
    sources=(koch("Eq. (2.12); definition Eq. (2.10)"),),
    valid_when="E_J / E_C >> 1. Sets how fast a gate can be without leaking to level 2.",
)

charge_dispersion = Formula(
    key="transmon.charge_dispersion",
    title="Charge dispersion of level m, E_m(n_g=1/2) - E_m(n_g=0)",
    lhs=r"\epsilon_m",
    expr=sp.Mul(
        (-1) ** m,
        E_C,
        sp.Mul(2 ** (4 * m + 5), sp.Pow(sp.factorial(m), -1, evaluate=False), evaluate=False),
        sp.sqrt(2 / sp.pi),
        sp.Pow(E_J / (2 * E_C), m / 2 + sp.Rational(3, 4), evaluate=False),
        sp.exp(
            -sp.Pow(
                sp.Mul(8, E_J, sp.Pow(E_C, -1, evaluate=False), evaluate=False),
                sp.Rational(1, 2),
                evaluate=False,
            ),
            evaluate=False,
        ),
        evaluate=False,
    ),
    args=(m, E_J, E_C),
    sources=(koch("Eq. (2.5); definition Eq. (2.4)"),),
    valid_when="Asymptotic in E_J / E_C. Why transmons shrug off charge noise.",
)

ALL = (plasma_frequency, level_energy, qubit_frequency, anharmonicity, charge_dispersion)
