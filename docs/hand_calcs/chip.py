"""Hand calc: from chip geometry to circuit parameters, the steps between the papers and ours.

Units are SI throughout: farads, henries, metres, hertz, joules.

1. Parallel-plate capacitor. Gauss's law for two plates of area A a distance d apart, charge
   +-Q, no fringing: the field between them is E = Q / (eps0 eps_r A), the voltage V = E d, so
       C = Q / V = eps0 eps_r A / d.
   Fringing fields at the plate edges add capacitance, so the formula needs d << sqrt(A).
2. Effective permittivity of a coplanar waveguide. goeppl2008 Eq. (3) writes the capacitance
   per length as C_l = 4 eps0 eps_eff K(k0)/K(k0'), the vacuum result times eps_eff. With
   conductors of zero thickness on the flat surface of a substrate much thicker than the CPW
   (w + 2s), the conformal map of the half space above the conductors and the one below them
   are mirror images, so each half carries half of the vacuum capacitance. Filling the lower
   half with eps_r gives
       C_l = (1 + eps_r)/2 * C_l(vacuum),  so  eps_eff = (1 + eps_r) / 2.
   goeppl2008 instead takes eps_eff from a simulation of their two-layer substrate (Si with
   550 nm of SiO2), so this closed form is ours, for a single thick substrate.
3. Josephson inductance. Expanding koch2007 Eq. (2.1) for small phase,
   -E_J cos(phi) = -E_J + E_J phi^2 / 2. With the flux Phi = (hbar / 2e) phi, an inductor
   stores Phi^2 / (2 L), so
       L_J = (hbar / 2e)^2 / E_J.
4. Harmonic (plasma) frequency. koch2007 Eq. (2.11), leading term, is hbar omega_p =
   sqrt(8 E_J E_C). With E_C = e^2 / (2 C_Sigma) (koch2007, text after Eq. (2.1)) and step 3,
       omega_p = 1 / sqrt(L_J C_Sigma),
   the LC resonance of the junction inductance with the total capacitance.
5. Qubit-resonator coupling from capacitances. koch2007 Eq. (3.3) with j = 0 and the
   asymptotic matrix element Eq. (3.4), beta = C_g / C_Sigma and V_rms = sqrt(hbar omega_r /
   (2 C_r)) (text after Eq. (3.1)):
       hbar g = 2 beta e V_rms (E_J / 8 E_C)^(1/4) / sqrt(2).
   Step 4 gives (E_J / 8 E_C)^(1/4) = sqrt(hbar omega_q C_Sigma) / (2 e), with omega_q the
   harmonic frequency, so
       g = (C_g / 2) sqrt(omega_q omega_r / (C_Sigma C_r)),
   the coupling of two LC oscillators through C_g. It holds as written for g / 2 pi in hertz
   with f_q, f_r in hertz. For a half-wave resonator, C_r = C_l l / 2 is goeppl2008 Eq. (12).
   At resonance (f_q = f_r) two coupled oscillators split by 2 g, which is what the eigenmode
   sweep measures.

Run this file to redo 4 and 5 with SymPy.
"""

import sympy as sp

hbar, e, L_J, C_S, C_g, C_r, w_q, w_r = sp.symbols(
    "hbar e L_J C_Sigma C_g C_r omega_q omega_r", positive=True
)

# 3 and 4.
E_J = (hbar / (2 * e)) ** 2 / L_J
E_C = e**2 / (2 * C_S)
print(
    "hbar omega_p - hbar/sqrt(L_J C_Sigma) =",
    sp.simplify(sp.sqrt(8 * E_J * E_C) - hbar / sp.sqrt(L_J * C_S)),
)  # 0

# 5. Koch Eqs. (3.3)-(3.4) with E_J/E_C eliminated through hbar omega_q = sqrt(8 E_J E_C).
E_J_q = (hbar * w_q) ** 2 / (8 * E_C)
g_koch = (
    2
    * (C_g / C_S)
    * e
    * sp.sqrt(hbar * w_r / (2 * C_r))
    * (E_J_q / (8 * E_C)) ** sp.Rational(1, 4)
    / sp.sqrt(2)
    / hbar
)
print(
    "g_koch - (C_g/2) sqrt(w_q w_r/(C_Sigma C_r)) =",
    sp.simplify(g_koch - C_g / 2 * sp.sqrt(w_q * w_r / (C_S * C_r))),
)  # 0
