# Dispersive readout

To read a qubit without touching it, ask a resonator.

**The idea.** Couple the qubit to a microwave resonator, detuned by Δ, far more than the coupling g. They cannot swap energy, but each shifts the other's frequency. The resonator sits at f_r − χ when the qubit is in |g⟩ and at f_r + χ when it is in |e⟩. Drive the resonator and the reflected wave carries a phase that depends on the qubit.

**How big is χ?** For a two-level qubit, χ = g²/Δ. A transmon's third level pulls the other way, and Koch et al. give χ = g₀₁²/Δ − g₁₂²/(2(Δ + α)). The two terms nearly cancel, so χ is several times smaller than g²/Δ. A common shortcut sets g₁₂ = √2 g₀₁. The exact transmon has g₁₂/g₀₁ ≈ 1.39 even at E_J/E_C = 200, and our checks show the shortcut missing χ by 6 to 64%.

**Pointer states.** With the qubit in |g⟩ or |e⟩, the resonator settles into one of two coherent states. Their separation in the IQ plane is the signal. For a fixed photon number, the signal-to-noise ratio peaks at 2χ = κ.

**The price of looking.** Photons in the resonator leak which-state information, so the qubit dephases at Γ_m = 2κχ²ε²/(κ²/4 + χ²)². The simulation matches this to 0.1%. The popular small-χ form, 8χ²n̄/κ, is 4 times too high at χ = κ/2.

**Too many photons.** The dispersive picture holds only for photon numbers well below the critical number Δ²/4g². Beyond it, the approximation breaks down.

**Try it.** In *Readout*, raise E_J/E_C and watch the √2 shortcut drift from the exact χ. Then set κ near 2|χ| and see the pointer states separate furthest.

Sources: koch2007 (Eqs. 22, 27, 28), blais2004 (Eq. 12, Sec. VI.1), gambetta2006 (Eqs. 27, 28), blais2021 (Eqs. 43, 117), qubitlab_hand_calcs.
