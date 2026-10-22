# A transmon in one page

The transmon is the standard superconducting qubit, and the family Qolab builds on.

**The circuit.** A Josephson junction shunted by a large capacitor. Two energies set everything: the Josephson energy E_J (how easily Cooper pairs tunnel) and the charging energy E_C (the cost of adding one charge). The Hamiltonian is

H = 4E_C (n − n_g)² − E_J cos φ

**The levels.** For E_J ≫ E_C it is almost a harmonic oscillator. The qubit is the bottom two levels, at E₀₁ ≈ √(8E_J E_C) − E_C. Typical numbers: E_J = 15 GHz, E_C = 0.3 GHz, so E₀₁ ≈ 5.7 GHz.

**Anharmonicity.** Level 1→2 sits about E_C lower than 0→1. That gap is what keeps a microwave pulse from leaking out of the qubit. Our exact diagonalization finds it 10 to 15% larger in magnitude than E_C at E_J/E_C = 50 to 100.

**Why "transmon".** The offset charge n_g wanders in real chips. The energies wiggle with n_g by an amount that shrinks as e^(−√(8E_J/E_C)). Raising E_J/E_C from 1 to 50 flattens the bands by six orders of magnitude, at the cost of less anharmonicity.

**Try it.** In the explorer's *Transmon* view, drag E_J/E_C and watch the bands flatten.

Sources: koch2007 (Eqs. 2.1, 2.4, 2.5, 2.10 to 2.12), tqi2024qolab.
