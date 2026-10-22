"""Hand calc: parity decay of a stabilized cat, for any size.

Two-photon dissipation keeps the state in the space spanned by the even and odd cats |C+>, |C->.
A single-photon loss jump a turns |C+> into |C-> and back (a|C+> is proportional to |C->).
The jump rate out of each state is kappa_1 <n> in that state:
    n_+ = |alpha|^2 tanh|alpha|^2,   n_- = |alpha|^2 coth|alpha|^2   (gerry2005).
For a two-state process with rates r1 (+ -> -) and r2 (- -> +), the difference of populations
<P> = P_+ - P_- relaxes at r1 + r2. So
    Gamma_P = kappa_1 |alpha|^2 (tanh|alpha|^2 + coth|alpha|^2)
            = 2 kappa_1 |alpha|^2 coth(2|alpha|^2),     using tanh x + coth x = 2 coth 2x.
For large cats coth -> 1 and this is 2 kappa_1 |alpha|^2.
"""

import sympy as sp

x, k1 = sp.symbols("x kappa_1", positive=True)
rate = k1 * x * (sp.tanh(x) + sp.coth(x))
print("identity holds:", sp.simplify((rate - 2 * k1 * x * sp.coth(2 * x)).rewrite(sp.exp)) == 0)
for n in (0.3, 1, 2, 4):
    exact = float((2 * x * sp.coth(2 * x)).subs(x, n))
    print(f"|alpha|^2 = {n}: Gamma_P / kappa_1 = {exact:.5f}  (large-cat law: {2 * n})")
