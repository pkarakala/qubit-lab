# From layout to Hamiltonian

A transmon's Hamiltonian has two numbers, E_C and E_J. The layout sets E_C, and the junction sets E_J. This page follows one layout to both, with a field solver as the referee.

**The layout.** A 400 µm square pad on silicon, 30 µm from the ground plane. A junction bridges the gap. Next to the pad is the open end of an 8 mm coplanar waveguide (CPW) resonator: a 10 µm strip in a 6 µm trench.

**Capacitance first.** Palace, an open-source finite-element solver, puts a voltage on each conductor and integrates the charge. This gives the capacitance matrix. The pad's total capacitance is C_Σ = 198 fF, so E_C = e²/2C_Σ is about 98 MHz. The pad sees 10 fF of it through the resonator.

**Testing the solver.** Palace is checked on cases with closed forms first. Two plates 100 times wider than their gap match ε₀A/d to 3%. Plates only 4 times wider do not, because the fringing fields are as big as the field between the plates. For the CPW, Göppl et al. give the capacitance per length with elliptic integrals: 163.5 pF/m. Palace gives 165.3 pF/m. It moves closer as the elements get finer, because the field is singular at the thin metal edges.

**Frequencies.** In the eigenmode solve, the junction becomes an inductor L_J. At 12 nH the qubit mode sits about 1% below 1/(2π√(L_J C_Σ)), and the resonator sits near c/(2l√ε_eff).

**Coupling.** Sweep L_J until the qubit meets the resonator. The two modes repel each other, and half the smallest gap is the coupling g. Koch et al. give g in terms of C_g, C_Σ and the resonator capacitance. The Checks view compares the two.

**Try it.** In *Architectures*, open *Simulated transmon* and explode it. Its levels come from the Palace capacitance. Then open *Checks* and scroll to the chip rows.

Sources: goeppl2008 (Eqs. 1, 3-5, 12), koch2007 (Eqs. 2.1, 2.11, 3.3, 3.4), palace2026, geuzaine2009, qubitlab_hand_calcs.
