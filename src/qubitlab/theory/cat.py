"""Closed-form cat qubit formulas.

Conventions (shared with `qubitlab.sim`):
- One bosonic mode with annihilation operator a. The cat amplitude alpha is real and positive.
- Phase space point beta = x + i p, so x = Re(beta), p = Im(beta). This matches Dynamiqs'
  `dq.wigner` default (hbar = 1/2): a coherent state |alpha> peaks at x = alpha with height 2/pi.
- Rates are in units of the two-photon dissipation rate kappa_2 unless a name says otherwise.
- Cat qubit basis (convention of reglade2024): |0>, |1> ~ |+alpha>, |-alpha>. A bit-flip swaps them.
  |+>, |-> are the even and odd cats. A phase-flip swaps those, which is a parity flip.
"""

from __future__ import annotations

import sympy as sp

from qubitlab.provenance import Kind, Source
from qubitlab.theory.registry import Formula

alpha = sp.Symbol("alpha", positive=True)
x, p = sp.symbols("x p", real=True)
kappa_1, kappa_2, kappa_phi = sp.symbols("kappa_1 kappa_2 kappa_phi", positive=True)
delta_n = sp.Symbol("Delta_n", positive=True)
K, epsilon_2 = sp.symbols("K epsilon_2", positive=True)

GERRY = Source("gerry2005", Kind.TEXTBOOK, "Ch. 3 (coherent states), Ch. 7 (cat states)")


def _blob(x0: sp.Expr) -> sp.Expr:
    """exp(-2|beta - x0|^2), kept unevaluated so the LaTeX reads like the textbook."""
    return sp.exp(-2 * ((x - x0) ** 2 + p**2), evaluate=False)


def _cat_wigner(sign: int) -> sp.Expr:
    """Cat Wigner function: two blobs plus +-2 e^{-2|beta|^2} cos(4 alpha p) fringes, normalized."""
    norm = sp.Pow(sp.pi * (1 + sign * sp.exp(-2 * alpha**2)), -1, evaluate=False)
    fringes = sp.Mul(2 * sign, sp.exp(-2 * (x**2 + p**2)), sp.cos(4 * alpha * p), evaluate=False)
    return sp.Mul(
        norm, sp.Add(_blob(alpha), _blob(-alpha), fringes, evaluate=False), evaluate=False
    )


coherent_overlap = Formula(
    key="cat.coherent_overlap",
    title="Overlap of the two cat basis states",
    lhs=r"|\langle \alpha | -\alpha \rangle|^2",
    expr=sp.exp(-4 * alpha**2),
    args=(alpha,),
    sources=(GERRY,),
    valid_when="Exact for coherent states.",
)

wigner_coherent = Formula(
    key="cat.wigner_coherent",
    title="Wigner function of a coherent state",
    lhs=r"W_{|\alpha\rangle}(x, p)",
    expr=sp.Mul(2 / sp.pi, _blob(alpha), evaluate=False),
    args=(x, p, alpha),
    sources=(GERRY,),
    valid_when="Exact. Convention beta = x + ip, hbar = 1/2.",
)

wigner_even_cat = Formula(
    key="cat.wigner_even_cat",
    title="Wigner function of the even cat |C+>",
    lhs=r"W_{|\mathcal{C}^+_\alpha\rangle}(x, p)",
    expr=_cat_wigner(+1),
    args=(x, p, alpha),
    sources=(GERRY,),
    valid_when="Exact for real alpha. The cosine fringes near the origin carry the parity.",
)

wigner_odd_cat = Formula(
    key="cat.wigner_odd_cat",
    title="Wigner function of the odd cat |C->",
    lhs=r"W_{|\mathcal{C}^-_\alpha\rangle}(x, p)",
    expr=_cat_wigner(-1),
    args=(x, p, alpha),
    sources=(GERRY,),
    valid_when="Exact for real alpha.",
)

even_cat_photon_number = Formula(
    key="cat.even_cat_photon_number",
    title="Mean photon number of the even cat",
    lhs=r"\langle n \rangle_{\mathcal{C}^+}",
    expr=alpha**2 * sp.tanh(alpha**2),
    args=(alpha,),
    sources=(GERRY,),
    valid_when="Exact. Tends to |alpha|^2 once |alpha|^2 > 2.",
)

odd_cat_photon_number = Formula(
    key="cat.odd_cat_photon_number",
    title="Mean photon number of the odd cat",
    lhs=r"\langle n \rangle_{\mathcal{C}^-}",
    expr=alpha**2 / sp.tanh(alpha**2),
    args=(alpha,),
    sources=(GERRY,),
    valid_when="Exact.",
)

stabilized_photon_number = Formula(
    key="cat.stabilized_photon_number",
    title="Cat size when single-photon loss competes with two-photon dissipation",
    lhs=r"|\alpha_\infty|^2",
    expr=alpha**2 - kappa_1 / (2 * kappa_2),
    args=(alpha, kappa_1, kappa_2),
    sources=(
        Source(
            "qubitlab_hand_calcs",
            Kind.DERIVED,
            "docs/hand_calcs/cat_steady_state.py",
            "Mean-field steady state of d<a>/dt, starting from lescanne2020 Eq. (1).",
        ),
    ),
    valid_when="Semiclassical. kappa_1 << kappa_2 |alpha|^2.",
)

parity_decay_rate = Formula(
    key="cat.parity_decay_rate",
    title="Decay rate of the cat parity <P> under single-photon loss",
    lhs=r"\Gamma_{P}",
    expr=2 * kappa_1 * alpha**2,
    args=(kappa_1, alpha),
    sources=(
        Source(
            "mirrahimi2014",
            Kind.PEER_REVIEWED,
            "Sec. 2, unnumbered text after Eq. (1)",
            "Gives the flip rate kappa_1 |alpha|^2; <P> decays at twice that.",
        ),
    ),
    valid_when="Two-photon dissipation dominates; |alpha|^2 > ~1.",
)

phase_flip_rate = Formula(
    key="cat.phase_flip_rate",
    title="Phase-flip probability rate",
    lhs=r"\Gamma_Z",
    expr=kappa_1 * alpha**2,
    args=(kappa_1, alpha),
    sources=(
        Source(
            "mirrahimi2014",
            Kind.PEER_REVIEWED,
            "Sec. 2, unnumbered text after Eq. (1)",
            "That paper swaps the basis names and calls this error a bit-flip.",
        ),
        Source("guillaud2019", Kind.PEER_REVIEWED, "Sec. II, unnumbered", "Linear in |alpha|^2."),
    ),
    valid_when="Same as parity decay. <X> = 1 - 2 p_Z decays at 2 Gamma_Z.",
)

bit_flip_suppression = Formula(
    key="cat.bit_flip_suppression",
    title="Bit-flip rate ratio after adding Delta_n photons",
    lhs=r"\frac{\Gamma_X(|\alpha|^2 + \Delta n)}{\Gamma_X(|\alpha|^2)}",
    expr=sp.exp(-2 * delta_n),
    args=(delta_n,),
    sources=(
        Source(
            "guillaud2019",
            Kind.PEER_REVIEWED,
            "Sec. II, unnumbered",
            "Bit-flips are exponentially suppressed with 2|alpha|^2.",
        ),
        Source(
            "lescanne2020",
            Kind.PEER_REVIEWED,
            "Fig. 3",
            "Measured: x4.2 per added photon (not e^2 = 7.4), saturating at 1 ms.",
        ),
        Source(
            "reglade2024",
            Kind.PEER_REVIEWED,
            "Fig. 4b caption",
            "A later device measured the same x4.2 per added photon.",
        ),
    ),
    valid_when="Asymptotic, |alpha|^2 >~ 4. Prefactor depends on the noise model.",
)

bit_flip_rate_dephasing = Formula(
    key="cat.bit_flip_rate_dephasing",
    title="Bit-flip rate caused by pure dephasing",
    lhs=r"\Gamma_X^{(\phi)}",
    expr=kappa_phi * alpha**2 / (2 * sp.sinh(2 * alpha**2)),
    args=(kappa_phi, alpha),
    sources=(
        Source(
            "mirrahimi2014",
            Kind.PEER_REVIEWED,
            "Sec. 2, unnumbered display",
            "Called a phase-flip there, because that paper swaps the basis names.",
        ),
    ),
    valid_when="kappa_phi << kappa_2. Falls as e^{-2|alpha|^2}, like the loss-induced rate.",
)

kerr_cat_amplitude = Formula(
    key="cat.kerr_cat_amplitude",
    title="Kerr cat amplitude",
    lhs=r"\alpha",
    expr=sp.sqrt(epsilon_2 / K),
    args=(epsilon_2, K),
    sources=(
        Source("grimm2020", Kind.PEER_REVIEWED, "Eq. (1)", "The Kerr-cat Hamiltonian."),
        Source(
            "qubitlab_hand_calcs",
            Kind.DERIVED,
            "docs/hand_calcs/kerr_cat_amplitude.py",
            "Classical fixed points of Eq. (1).",
        ),
    ),
    valid_when="H = -K a+^2 a^2 + epsilon_2 (a+^2 + a^2); |+-alpha> are degenerate eigenstates.",
)

parity_decay_rate_exact = Formula(
    key="cat.parity_decay_rate_exact",
    title="Parity decay, any cat size",
    lhs=r"\Gamma_{P}",
    expr=sp.Mul(
        2,
        kappa_1,
        alpha**2,
        sp.coth(sp.Mul(2, alpha**2, evaluate=False), evaluate=False),
        evaluate=False,
    ),
    args=(kappa_1, alpha),
    sources=(
        Source(
            "qubitlab_hand_calcs",
            Kind.DERIVED,
            "docs/hand_calcs/cat_parity_decay.py",
            "Two-state jump picture: |C+> -> |C-> at kappa_1 n_+, back at kappa_1 n_-.",
        ),
    ),
    valid_when="Two-photon dissipation dominates, so the state stays in the two-cat space.",
)

ALL = (
    coherent_overlap,
    wigner_coherent,
    wigner_even_cat,
    wigner_odd_cat,
    even_cat_photon_number,
    odd_cat_photon_number,
    stabilized_photon_number,
    parity_decay_rate,
    parity_decay_rate_exact,
    phase_flip_rate,
    bit_flip_suppression,
    bit_flip_rate_dephasing,
    kerr_cat_amplitude,
)
