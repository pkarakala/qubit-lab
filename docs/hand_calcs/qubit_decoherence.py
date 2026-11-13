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
7. Flip probabilities. Twirling any qubit channel over the Paulis leaves a Pauli channel with
   probabilities p_I, p_X, p_Y, p_Z and keeps the diagonal of its Pauli transfer matrix,
   R_P = Tr(P Lambda(P)) / 2. A Pauli Q multiplies <P> by -1 if it anticommutes with P, so
       R_Z = 1 - 2 (p_X + p_Y),   R_X = 1 - 2 (p_Z + p_Y).
   A bit-flip is any error that flips Z (X or Y), a phase-flip any error that flips X (Z or Y):
       p_bit = (1 - R_Z) / 2,   p_phase = (1 - R_X) / 2.
   If <P> decays as exp(-t/T), then R_P = exp(-t/T) and p = (1 - exp(-t/T)) / 2.
   Transmon: R_Z = g (T1), R_X = c (T2). Cat: R_Z = exp(-t/T_X), R_X = exp(-t/T_Z), the
   definitions of T_X and T_Z in reglade2024 (Fig. 3 caption). Then
       p_phase / p_bit -> T_bit / T_phase for t << T, the noise bias.
8. Pi pulse at a small, unknown detuning. The pulse Omega_x(t) was made for Delta = 0 and
   rotates the qubit about x by phi(t) = 2 pi int_0^t Omega_x, with phi(T) = pi. Write
   Delta |1><1| = Delta (1 - sigma_z) / 2 and drop the identity (a global phase). In the frame
   of the ideal pulse, sigma_z becomes cos(phi) sigma_z + sin(phi) sigma_y (up to signs), so to
   first order in Delta the gate is X exp(-i v.sigma) with
       v = pi Delta (int_0^T cos(phi) dt, int_0^T sin(phi) dt).
   For U = X exp(-i v.sigma), Tr(X^dagger U) = 2 cos|v|, and pedersen2007 Eq. (3) with
   Tr(M M^dagger) = 2 gives 1 - F = (2/3) sin^2|v| = (2/3)|v|^2 + O(|v|^4).
   A symmetric envelope has phi(T - t) = pi - phi(t), so the cosine integral vanishes and
       1 - F = (2/3) (pi Delta tau)^2,   tau = int_0^T sin(phi(t)) dt.
   Square pulse of length T = 1/(2 Omega): tau = 2T/pi, so 1 - F = (2/3)(Delta/Omega)^2,
   which the exact Rabi solution of step 1 confirms below.
   A real pulse also has its own small error at Delta = 0, a vector v0. Then
   1 - F = (2/3)|v0 + v|^2: the cross term (4/3) v0.v is odd in Delta, of relative size about
   2 |v0| / |v| = 2 sqrt(e0 / (1 - F)) with e0 the error at Delta = 0. The mean of 1 - F over
   +Delta and -Delta cancels it.

Run this file to redo steps 1, 3, 4, 6, 7 and 8 with SymPy.
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

# 7. Pauli channel: <P> scales by R_P. Check p_bit and p_phase from R_Z and R_X.
pI, pX, pY, pZ = sp.symbols("p_I p_X p_Y p_Z", nonnegative=True)
X = sp.Matrix([[0, 1], [1, 0]])
Y = sp.Matrix([[0, -sp.I], [sp.I, 0]])
Z = sp.Matrix([[1, 0], [0, -1]])
I2 = sp.eye(2)


def pauli_channel(r: sp.Matrix) -> sp.Matrix:
    return pI * r + pX * X * r * X + pY * Y * r * Y + pZ * Z * r * Z


R_Z = sp.expand((Z * pauli_channel(Z)).trace() / 2).subs(pI, 1 - pX - pY - pZ)
R_X = sp.expand((X * pauli_channel(X)).trace() / 2).subs(pI, 1 - pX - pY - pZ)
print("p_bit - (pX + pY) =", sp.simplify((1 - R_Z) / 2 - (pX + pY)))  # 0
print("p_phase - (pZ + pY) =", sp.simplify((1 - R_X) / 2 - (pZ + pY)))  # 0

# 8. Square pi pulse at detuning Delta = eps * Omega: exact 1 - F from pedersen2007 Eq. (3),
# then its leading term in eps. U is the propagator of H from step 1 at t = 1/(2 Omega).
eps = sp.symbols("epsilon", positive=True)
U8 = U.subs({de: eps * om, t: 1 / (2 * om)})
M8 = X * U8  # X^dagger U
F8 = ((M8 * M8.H).trace() + sp.Abs(M8.trace()) ** 2) / 6
err8 = sp.series(sp.simplify(1 - F8.subs(om, 1)), eps, 0, 4).removeO()
print("square pulse 1 - F =", sp.simplify(err8))  # 2 eps^2 / 3
# The general formula with tau = 2T/pi, T = 1/(2 Omega): (2/3)(pi Delta tau)^2 = (2/3) eps^2.
tau8 = 2 * (1 / (2 * om)) / sp.pi
print("formula / exact:", sp.simplify(2 * (sp.pi * eps * om * tau8) ** 2 / 3 / err8))  # 1
