"""Hand calc: two-qubit gates, the steps between the papers and our formulas.

Model for two transmons (krantz2019 Eq. (116)): levels |nm> with energies
E_nm = E_n(q1) + E_m(q2), E_1 = omega_i, E_2 = 2 omega_i + alpha_i, and an exchange coupling g
between |01> and |10> that becomes sqrt(2) g between |11> and |20>, |02>. Angular units, hbar = 1.
Delta = omega_1 - omega_2.

1. Static ZZ to second order in g. Shifts: |01> by g^2/(omega_2 - omega_1), |10> by
   g^2/(omega_1 - omega_2), |11> by 2g^2/(E_11 - E_20) + 2g^2/(E_11 - E_02). With
   zeta = omega_11 - (omega_01 + omega_10) (krantz2019 Eq. (119)):
       zeta = 2 g^2 (1/(Delta - alpha_2) - 1/(Delta + alpha_1)).
2. Diabatic CZ. At the |11>-|20> resonance the two states form a two-level system coupled by
   sqrt(2) g. A full cycle |11> -> |20> -> |11> takes t = 2 pi / (2 sqrt(2) g) = pi / (sqrt(2) g)
   and returns |11> with a phase of -1 (krantz2019, text after Eq. (123)).
3. Cat CNOT. guillaud2019 Sec. VI: p_Z1 = n kappa_1 T + 1/(2 pi n kappa_2 T) and
   p_Z2 = p_Z1Z2 = n kappa_1 T / 2. The paper's optimal time minimizes the total,
   2 n kappa_1 T + 1/(2 pi n kappa_2 T), giving T* = 1/(2 n sqrt(pi kappa_1 kappa_2)) and a total
   of sqrt(4 kappa_1 / (pi kappa_2)): F = sqrt(1 - 0.0357) = 98.2% at kappa_1/kappa_2 = 1e-3,
   as the paper states. (Minimizing p_Z1 alone would give a sqrt(2) longer gate.)

Run this file to redo 1, 2 and 3 with SymPy.
"""

import sympy as sp

g, D, a1, a2, w2 = sp.symbols("g Delta alpha_1 alpha_2 omega_2", real=True)
w1 = w2 + D

# 1. Second-order shifts.
E = {"01": w2, "10": w1, "11": w1 + w2, "20": 2 * w1 + a1, "02": 2 * w2 + a2}
s01 = g**2 / (E["01"] - E["10"])
s10 = g**2 / (E["10"] - E["01"])
s11 = 2 * g**2 / (E["11"] - E["20"]) + 2 * g**2 / (E["11"] - E["02"])
zeta = sp.simplify(s11 - s01 - s10)
print(
    "zeta - 2g^2(1/(D-a2) - 1/(D+a1)) =",
    sp.simplify(zeta - 2 * g**2 * (1 / (D - a2) - 1 / (D + a1))),
)  # 0

# 2. Two-level Rabi in {|11>, |20>} on resonance, coupling sqrt(2) g.
t = sp.symbols("t", positive=True)
H = sp.Matrix([[0, sp.sqrt(2) * g], [sp.sqrt(2) * g, 0]])
U = sp.simplify((-sp.I * H * t).exp())
print("U_11,11 at t = pi/(sqrt2 g):", sp.simplify(U[0, 0].subs(t, sp.pi / (sp.sqrt(2) * g))))  # -1

# 3. Optimal CNOT time.
n, k1, k2, T = sp.symbols("n kappa_1 kappa_2 T", positive=True)
p = 2 * n * k1 * T + 1 / (2 * sp.pi * n * k2 * T)  # p_Z1 + p_Z2 + p_Z1Z2
Tstar = sp.solve(sp.diff(p, T), T)[0]
print("T* - 1/(2 n sqrt(pi k1 k2)) =", sp.simplify(Tstar - 1 / (2 * n * sp.sqrt(sp.pi * k1 * k2))))
print("total at T*:", sp.simplify(p.subs(T, Tstar)))  # 2 sqrt(k1/(pi k2))
