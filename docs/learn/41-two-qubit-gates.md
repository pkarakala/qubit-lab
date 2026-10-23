# Two-qubit gates

One-qubit gates rotate a single Bloch vector. Computing needs gates that entangle, and every design does that differently.

**Transmon CZ.** Two transmons, coupled by g. Tune qubit 1 until |11⟩ has the same energy as |20⟩. Those two states are coupled by √2 g, so |11⟩ swings into |20⟩ and back. After one full cycle, t = π/(√2 g), it returns with a phase of −1. No other computational state has a partner at that frequency, so only |11⟩ picks up the sign. That is a controlled-Z.

**What goes wrong.** While qubit 1 is parked, |01⟩ and |10⟩ are detuned by about |α|. The weaker g is compared with |α|, the cleaner the gate. In our simulation, g = 5 MHz with α = −300 MHz gives a phase within 0.6% of π. At g = 50 MHz the phase misses by 6% and 1.4% of |11⟩ leaks out.

**Always-on ZZ.** Even when idle, the coupling shifts |11⟩ by ζ = 2g²(1/(Δ − α₂) − 1/(Δ + α₁)). It is a slow, unwanted CZ. Near Δ = −α₁ the formula fails, because the levels it treats as far apart are not.

**Cat CNOT.** The target cat's two-photon pump is turned half a circle, but only when the control is in |−α⟩. The pump never switches off, so bit-flips stay exponentially rare during the gate. Only phase-flips remain: from photon loss, n̄κ₁T, and from going too fast, 1/(2πn̄κ₂T). Their trade-off sets the best gate time, T* = 1/(2n̄√(πκ₁κ₂)). Our simulation matches the papers' phase-flip chances to within 8%, even for gates four times faster than T*.

**Try it.** In *Two-qubit gates*, raise g and watch the CZ phase drift from π. Then raise κ₁/κ₂ and watch the best cat gate get shorter and noisier.

Sources: krantz2019 (Eqs. 116, 119, 123, Sec. IV.F), guillaud2019 (Sec. IV, Sec. VI), chamberland2022 (Eqs. 22 to 27), qubitlab_hand_calcs.
