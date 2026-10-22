"""Hand calc: amplitude of the Kerr cat.

Start: H = -K a^dag^2 a^2 + epsilon_2 (a^dag^2 + a^2)   (grimm2020, Eq. (1)).

Classical limit: a -> a complex number b. The energy is
    E(b) = -K |b|^4 + epsilon_2 (b*^2 + b^2).
For real b = x: E(x) = -K x^4 + 2 epsilon_2 x^2, and dE/dx = 0 gives x = 0 or x^2 = epsilon_2 / K.
The two extrema x = +- sqrt(epsilon_2 / K) are the two cat states (degenerate, by symmetry).
They are maxima of E, so with this sign of H the cats are the highest-energy states; they are
the ground states of -H, the sign many papers use.
"""

import sympy as sp

x, K, eps2 = sp.symbols("x K epsilon_2", positive=True)
energy = -K * x**4 + 2 * eps2 * x**2
print("extrema:", sp.solve(sp.diff(energy, x), x))
