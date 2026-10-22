# A cat qubit in one page

A cat qubit stores one bit in a microwave resonator, not in an atom-like level.

**The two states.** A resonator can hold a coherent state |α⟩: a blob of light with amplitude α. The qubit uses two of them, |+α⟩ and |−α⟩, on opposite sides of phase space. Alice & Bob call these |0⟩ and |1⟩.

**Why they barely overlap.** The overlap is |⟨α|−α⟩|² = e^(−4|α|²). With 4 photons (|α|² = 4) that is about 10⁻⁷. Noise has to move the state a long way to turn 0 into 1.

**The superpositions.** |0⟩ + |1⟩ and |0⟩ − |1⟩ are the even and odd *cat states* (Schrödinger's cat: two classical-looking states at once). They differ only in photon number parity: even cats hold 0, 2, 4... photons, odd cats 1, 3, 5...

**How to see it.** The Wigner function W(x, p) is a map of the state over phase space. A cat shows two positive blobs and a stripe of fringes between them. The fringes are the quantum part: at the origin, W = +2/π for the even cat and −2/π for the odd one. In the explorer, the height of the 3D surface is W.

**Try it.** Open the explorer, pick *Even cat*, and raise |α|². The blobs separate and the fringes get finer.

Sources: gerry2005 (Ch. 3, 7). Checked in `cat.wigner` and `cat.photon_number`.
