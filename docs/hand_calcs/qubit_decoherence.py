"""Hand calc: a driven, decaying two-level qubit (levels 0 and 1 of the transmon).

Model, the same one `qubitlab.sim.transmon_dynamics` solves with more levels kept:
    H = 2 pi (Delta |1><1| + (Omega / 2) sigma_x)        frequencies in cycles per unit time
    jumps: sqrt(1/T1) |0><1|,  sqrt(2 Gamma_phi) |1><1|,  1/T_phi = Gamma_phi

1. Rabi (no noise). Generalized Rabi frequency Omega_R = sqrt(Omega^2 + Delta^2):
       P1(t) = (Omega / Omega_R)^2 sin^2(pi Omega_R t)
2. T1. d rho_11/dt = -rho_11 / T1, so P1(t) = exp(-t / T1).
3. T2. d rho_01/dt = -(1/(2 T1) + Gamma_phi) rho_01, so 1/T2 = 1/(2 T1) + 1/T_phi.
4. Ramsey from |+>. <X>(t) = exp(-t/T2) cos(2 pi Delta t). If Delta is Gaussian with mean
   Delta_0 and spread sigma (quasi-static noise), averaging cos gives an extra
   exp(-(2 pi sigma t)^2 / 2).
5. Hahn echo. A pi pulse at t/2 reverses the phase from any static Delta, so
   <X>(t) = exp(-t/T2) whatever sigma is.
6. Idle gate fidelity. The channel keeps coherences by c = exp(-t/T2) and populations of |1>
   by g = exp(-t/T1). Entanglement fidelity F_e = (1 + 2c + g)/4, and Nielsen's formula
   (nielsen2002, Eq. (3)) F_avg = (d F_e + 1)/(d + 1) with d = 2 gives
       F_avg = 1/2 + c/3 + g/6.

Run this file to redo steps 1, 3, 4 and 6 with SymPy.
"""

import sympy as sp

t, om, de, T1, gphi, sig, x = sp.symbols("t Omega Delta T_1 Gamma_phi sigma x", positive=True)

# 1. Rabi: evolve |0> under H and read |<1|psi>|^2.
H = 2 * sp.pi * sp.Matrix([[0, om / 2], [om / 2, de]])
U = sp.exp(-sp.I * H * t)  # sympy matrix exponential
p1 = sp.simplify(sp.Abs(U[1, 0]) ** 2)
target = om**2 / (om**2 + de**2) * sp.sin(sp.pi * sp.sqrt(om**2 + de**2) * t) ** 2
check = (p1 - target).subs({om: 0.3, de: 0.4, t: 1.7})
print("Rabi residual:", sp.N(check))

# 3. Coherence decay from the Lindblad equation.
r00, r01, r10, r11 = sp.symbols("r00 r01 r10 r11")
rho = sp.Matrix([[r00, r01], [r10, r11]])
lower = sp.Matrix([[0, 1], [0, 0]])
n1 = sp.Matrix([[0, 0], [0, 1]])


def dissipator(L: sp.Matrix, r: sp.Matrix) -> sp.Matrix:
    return L * r * L.H - (L.H * L * r + r * L.H * L) / 2


drho = dissipator(sp.sqrt(1 / T1) * lower, rho) + dissipator(sp.sqrt(2 * gphi) * n1, rho)
print("d rho_01 / dt =", sp.simplify(drho[0, 1]))  # -(1/(2 T1) + Gamma_phi) r01
print("d rho_11 / dt =", sp.simplify(drho[1, 1]))  # -r11 / T1

# 4. Gaussian average of cos(2 pi x t), x ~ N(0, sigma^2).
pdf = sp.exp(-(x**2) / (2 * sig**2)) / sp.sqrt(2 * sp.pi * sig**2)
avg = sp.integrate(sp.cos(2 * sp.pi * x * t) * pdf, (x, -sp.oo, sp.oo))
print("<cos> =", sp.simplify(avg))  # exp(-2 pi^2 sigma^2 t^2)

# 6. Average fidelity from entanglement fidelity.
c, g = sp.symbols("c g", positive=True)
print("F_avg =", sp.expand((2 * (1 + 2 * c + g) / 4 + 1) / 3))
