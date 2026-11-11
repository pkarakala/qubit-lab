"""Hand calc: dispersive readout, the algebra between the papers' equations and our formulas.

Conventions (Koch 2007, Gambetta 2006, Blais 2021): resonator shift +-chi, Delta = qubit minus
resonator frequency, alpha = anharmonicity (negative for a transmon), kappa = photon decay rate.

1. Transmon chi with the harmonic shortcut. koch2007 Eqs. (3.9) and (3.10):
   chi = chi_01 - chi_12 / 2 with chi_ij = g_ij^2 / (omega_ij - omega_r). Eq. (3.4) gives
   |<j+1|n|j>| ~ sqrt(j+1), so g_12 = sqrt(2) g_01. With g = g_01,
   omega_01 - omega_r = Delta and omega_12 - omega_r = Delta + alpha:
       chi = g^2 alpha / (Delta (Delta + alpha)).
   blais2021 Eq. (43) is the same with alpha = -E_C. As alpha -> infinity it becomes g^2/Delta,
   the two-level result of blais2004 Eq. (12). The shortcut g_12 = sqrt(2) g_01 is the
   E_J/E_C -> infinity limit of Eq. (3.4); scqubits gives g_12/g_01 = 1.38 to 1.39 at
   E_J/E_C = 50 to 200 (see the readout checks).
2. Resonator frequency with the qubit in |0>. Only chi_01 acts on the ground state, so the
   resonator sits at omega_r - g^2/Delta (koch2007, text after Eq. (3.8): omega_r' - chi).
3. Measurement-induced dephasing. gambetta2006 Eq. (5.20) with Eq. (5.21), driving at the bare
   resonator (Delta_r = 0), where n_+ = n_- = eps^2 / (kappa^2/4 + chi^2):
       Gamma_m = 2 kappa chi^2 eps^2 / (kappa^2/4 + chi^2)^2.
   For chi << kappa this is 32 chi^2 eps^2 / kappa^3 = 8 chi^2 n / kappa with n = 4 eps^2/kappa^2,
   gambetta2006 Sec. V.A (Gamma = 2 kappa n theta_0^2, theta_0 = 2 chi / kappa).
4. Pointer separation. The resonator field obeys
   d alpha/dt = -i (delta_r +- chi) alpha - (kappa/2) alpha - i eps (H of sim.readout, damping
   sqrt(kappa) a), so in steady state alpha_+- = -i eps / (kappa/2 + i (delta_r +- chi)), whose
   |alpha|^2 is gambetta2006 Eq. (5.21). Their difference is -2 eps chi over the product of the
   denominators:
       |alpha_e - alpha_g| = 2 eps chi / sqrt(n_+ n_-),
   n_+- = kappa^2/4 + (delta_r +- chi)^2.
5. Heterodyne SNR in gambetta2007's convention. Splitting the output into two arms measured
   through sqrt(kappa/2) a and -i sqrt(kappa/2) a, each record is dY = sqrt(eta) <L + L+> dt + dW,
   so I + iQ gains sqrt(2 eta kappa) alpha dt with unit-variance noise per quadrature. Along the
   line joining the two clouds the signal is +-r/2 per unit time, r = sqrt(2 eta kappa) D. Scaling
   time so the signal is +-1 (gambetta2007 Eqs. 2.1-2.3) gives noise variance tau_f / SNR with
       tau_f SNR = (r/2)^2 tau = eta kappa tau D^2 / 2.
   Check: gambetta2007 Eq. (2.6), F = erf(sqrt(tau_f SNR / 2)), is then
   erf(r sqrt(tau) / (2 sqrt 2)): two Gaussians a distance r tau apart, each of width
   sqrt(tau), split halfway.

Run this file to redo 1, 3 and 4 with SymPy.
"""

import sympy as sp

g, D, a, k, chi, eps = sp.symbols("g Delta alpha kappa chi epsilon", real=True)

# 1. Koch Eqs. (3.9)-(3.10) with g_12 = sqrt(2) g.
chi_koch = g**2 / D - (sp.sqrt(2) * g) ** 2 / (D + a) / 2
print(
    "chi_01 - chi_12/2 - g^2 alpha/(Delta(Delta+alpha)) =",
    sp.simplify(chi_koch - g**2 * a / (D * (D + a))),
)  # 0
print("alpha -> oo:", sp.limit(g**2 * a / (D * (D + a)), a, sp.oo))  # g**2/Delta

# 3. Gambetta Eq. (5.20) at Delta_r = 0, then the chi << kappa limit.
n_pm = eps**2 / (k**2 / 4 + chi**2)
gamma = 2 * n_pm * k * chi**2 / (k**2 / 4 + chi**2)
print("Gamma_m =", sp.simplify(gamma))
print("leading order in chi:", sp.series(gamma, chi, 0, 3).removeO())  # 32 chi^2 eps^2 / kappa^3

# 4. Pointer separation.
dr = sp.symbols("delta_r", real=True)
a_p = -sp.I * eps / (k / 2 + sp.I * (dr + chi))
a_m = -sp.I * eps / (k / 2 + sp.I * (dr - chi))
sep = 2 * eps * chi / sp.sqrt((k**2 / 4 + (dr + chi) ** 2) * (k**2 / 4 + (dr - chi) ** 2))
check = {eps: sp.Rational(3, 10), chi: sp.Rational(1, 2), k: 1, dr: sp.Rational(1, 5)}
print("|a_e - a_g| - formula =", sp.N(sp.Abs(a_p - a_m).subs(check) - sep.subs(check)))  # 0
