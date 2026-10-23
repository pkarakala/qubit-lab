"""Hand calc: dispersive readout, the algebra between the papers' equations and our formulas.

Conventions (Koch 2007, Gambetta 2006, Blais 2021): resonator shift +-chi, Delta = qubit minus
resonator frequency, alpha = anharmonicity (negative for a transmon), kappa = photon decay rate.

1. Transmon chi with the harmonic shortcut. koch2007 Eqs. (27) and (28):
   chi = chi_01 - chi_12 / 2 with chi_ij = g_ij^2 / (omega_ij - omega_r). Eq. (22) gives
   |<j+1|n|j>| ~ sqrt(j+1), so g_12 = sqrt(2) g_01. With g = g_01,
   omega_01 - omega_r = Delta and omega_12 - omega_r = Delta + alpha:
       chi = g^2 alpha / (Delta (Delta + alpha)).
   blais2021 Eq. (43) is the same with alpha = -E_C. As alpha -> infinity it becomes g^2/Delta,
   the two-level result of blais2004 Eq. (12). The shortcut g_12 = sqrt(2) g_01 is the
   E_J/E_C -> infinity limit of Eq. (22); scqubits gives g_12/g_01 = 1.38 to 1.39 at
   E_J/E_C = 50 to 200 (see the readout checks).
2. Resonator frequency with the qubit in |0>. Only chi_01 acts on the ground state, so the
   resonator sits at omega_r - g^2/Delta (koch2007, text after Eq. (26): omega_r' - chi).
3. Measurement-induced dephasing. gambetta2006 Eq. (27) with Eq. (28), driving at the bare
   resonator (Delta_r = 0), where n_+ = n_- = eps^2 / (kappa^2/4 + chi^2):
       Gamma_m = 2 kappa chi^2 eps^2 / (kappa^2/4 + chi^2)^2.
   For chi << kappa this is 32 chi^2 eps^2 / kappa^3 = 8 chi^2 n / kappa with n = 4 eps^2/kappa^2,
   gambetta2006 Sec. V.1 (Gamma = 2 kappa n theta_0^2, theta_0 = 2 chi / kappa).

Run this file to redo 1 and 3 with SymPy.
"""

import sympy as sp

g, D, a, k, chi, eps = sp.symbols("g Delta alpha kappa chi epsilon", real=True)

# 1. Koch Eqs. (27)-(28) with g_12 = sqrt(2) g.
chi_koch = g**2 / D - (sp.sqrt(2) * g) ** 2 / (D + a) / 2
print(
    "chi_01 - chi_12/2 - g^2 alpha/(Delta(Delta+alpha)) =",
    sp.simplify(chi_koch - g**2 * a / (D * (D + a))),
)  # 0
print("alpha -> oo:", sp.limit(g**2 * a / (D * (D + a)), a, sp.oo))  # g**2/Delta

# 3. Gambetta Eq. (27) at Delta_r = 0, then the chi << kappa limit.
n_pm = eps**2 / (k**2 / 4 + chi**2)
gamma = 2 * n_pm * k * chi**2 / (k**2 / 4 + chi**2)
print("Gamma_m =", sp.simplify(gamma))
print("leading order in chi:", sp.series(gamma, chi, 0, 3).removeO())  # 32 chi^2 eps^2 / kappa^3
