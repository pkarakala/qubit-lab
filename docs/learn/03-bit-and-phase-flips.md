# Bit-flips and phase-flips

A qubit can fail two ways. A cat qubit trades one for the other on purpose.

**Bit-flip (|+α⟩ ↔ |−α⟩).** Noise must carry the state across phase space. Each extra photon makes that harder. In theory the rate drops by e² ≈ 7.4 per photon. Our simulation of the ideal model gives 7.4 for |α|² from 5 to 7. Alice & Bob's 2020 experiment saw 4.2 per photon, then a ceiling at 1 ms. The 2024 device reached over 10 s at |α|² = 11.3.

**Phase-flip (even cat ↔ odd cat).** Losing one photon flips parity, and parity is the qubit's phase. Bigger cats have more photons to lose, so the rate grows linearly: Γ_Z = κ₁|α|². Simulation agrees to 0.2%.

**The trade.** Raising |α|² buys exponentially fewer bit-flips for linearly more phase-flips. That is *biased noise*. With bit-flips negligible, only phase-flips need correcting, and a one-dimensional repetition code is enough.

**Dephasing.** Frequency noise also causes bit-flips, at κ_φ|α|²/(2 sinh 2|α|²). Also exponentially small. Simulation agrees within 10%.

**Watch out for names.** Mirrahimi et al. 2014 label the basis the other way round, so their "bit-flip" is our phase-flip.

**Try it.** In the explorer's *Flip rates* view, drag |α|² and watch the two curves cross.

Sources: guillaud2019 (Sec. II), mirrahimi2014 (Sec. 2), lescanne2020 (Fig. 3), reglade2024 (abstract).
