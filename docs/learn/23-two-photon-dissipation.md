# Two-photon dissipation

Left alone, a resonator leaks photons and every state decays to vacuum. A cat qubit needs something that pushes the state back onto |±α⟩.

**The trick.** Couple the memory resonator to a lossy buffer resonator through a nonlinear element, so that photons only leave the memory *in pairs*. Losing pairs never changes parity, and it stops exactly when the state reaches |±α⟩. The effective equation is

ρ̇ = κ₂ D[a² − α²] ρ

where D is the Lindblad dissipator and κ₂ the two-photon rate. In hardware κ₂ ≈ 4|g₂|²/κ_b, set by the coupling g₂ and the buffer's loss κ_b.

**What it does.** Start from vacuum. The state grows into an even cat, because vacuum is even and pair loss keeps parity. Start from a coherent state at +α and it stays there.

**With ordinary loss.** Add single-photon loss κ₁. The cat shrinks slightly to |α∞|² = |α|² − κ₁/(2κ₂). Our simulation agrees to within 1% when κ₁ ≪ κ₂|α|² and misses by 12% when |α|² = 1 and κ₁ = κ₂/2, as predicted.

**Kerr cat, the alternative.** A Hamiltonian, H = −K a†²a² + ε₂(a†² + a²), holds the same two states as degenerate eigenstates with |α|² = ε₂/K. With this sign of H they sit at the top of the spectrum; papers often flip the sign and call them ground states. The Phase space view has a Kerr toggle.

**Try it.** In *Phase space*, pick the lesson *Grow a cat* and watch vacuum become an even cat. Then add photon loss.

Sources: lescanne2020 (Eq. 1), mirrahimi2014 (Eq. 1), grimm2020 (Eq. 1), hand calcs in `docs/hand_calcs/`.
