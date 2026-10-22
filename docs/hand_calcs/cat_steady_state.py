"""Hand calc: cat size when single-photon loss competes with two-photon dissipation.

Start: d(rho)/dt = kappa_2 D[a^2 - alpha^2] rho + kappa_1 D[a] rho   (lescanne2020, Eq. (1)).

For any jump operator L, d<a>/dt = sum_L < L^dag [a, L] + [L^dag, a] L > / 2.
- L = sqrt(kappa_1) a:              contributes -kappa_1 <a> / 2.
- L = sqrt(kappa_2) (a^2 - alpha^2): [a, a^2] = 0 and [a^dag^2, a] = -2 a^dag, so it contributes
  -kappa_2 < a^dag (a^2 - alpha^2) >.

Mean field: replace a by a real number a_inf and set d<a>/dt = 0:
    0 = -kappa_2 a_inf (a_inf^2 - alpha^2) - kappa_1 a_inf / 2
    =>  a_inf^2 = alpha^2 - kappa_1 / (2 kappa_2)

Run this file to redo the algebra with SymPy and print the numbers used in the checks.
"""

import sympy as sp

a_inf, alpha, k1, k2 = sp.symbols("a_inf alpha kappa_1 kappa_2", positive=True)
rate = -k2 * a_inf * (a_inf**2 - alpha**2) - k1 * a_inf / 2
(solution,) = sp.solve(sp.Eq(rate, 0), a_inf)
print("a_inf^2 =", sp.simplify(solution**2))

for alpha_sq, kappa_1 in [(4, 0.01), (4, 0.1), (2, 0.2), (1, 0.5)]:
    value = (solution**2).subs({alpha: sp.sqrt(alpha_sq), k1: kappa_1, k2: 1})
    print(f"|alpha|^2 = {alpha_sq}, kappa_1/kappa_2 = {kappa_1}: n_inf = {float(value):.4f}")
