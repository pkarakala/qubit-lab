# Open systems: T1, T2 and fidelity

A real qubit talks to its surroundings. That is where errors come from.

**Density matrix.** A state we are not sure about is a density matrix ρ, a weighted mix of pure states. Its purity, Tr ρ², is 1 for a pure state and drops as information leaks out.

**The Lindblad equation.** For memoryless (Markovian) noise,

ρ̇ = −i[H, ρ] + Σₖ D[Lₖ]ρ,  D[L]ρ = LρL† − ½(L†Lρ + ρL†L).

H is the Hamiltonian. Each jump operator Lₖ is one way the environment acts: losing a photon, scrambling a phase. Every simulation here solves this equation with Dynamiqs.

**T1.** Energy relaxation, jump √(1/T1)|0⟩⟨1|. The excited state empties as P₁ = e^(−t/T1).

**T2.** Coherence between |0⟩ and |1⟩ decays as e^(−t/T2), with 1/T2 = 1/(2T1) + 1/T_φ. T_φ is pure dephasing. So T2 can never exceed 2T1.

**Fidelity.** Idling for time t keeps a random state with average fidelity F̄ = 1/2 + e^(−t/T2)/3 + e^(−t/T1)/6. The chance of a bit-flip is (1 − e^(−t/T1))/2, and of a phase-flip (1 − e^(−t/T2))/2.

**Try it.** In *Qubit control*, run the *T1* and *Ramsey* lessons. In *Playground*, add a loss and watch the purity fall.

Sources: qubitlab_hand_calcs (steps 2, 3, 6 and 7), nielsen2002 (Eq. 3), dynamiqs.
