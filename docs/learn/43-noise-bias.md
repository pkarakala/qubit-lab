# Noise bias in numbers

[Two designs, two bets](42-cat-vs-transmon.md) says cats trade one error for the other. Here are measured numbers.

**One idle step.** Let a qubit sit for t. A bit-flip flips Z; a phase-flip flips X. If ⟨Z⟩ decays as e^(−t/T), the bit-flip chance is p = (1 − e^(−t/T))/2. The phase-flip chance follows the same way from ⟨X⟩.

**Which T.** For a transmon, ⟨Z⟩ decays with T1 and ⟨X⟩ with T2. For a cat, Réglade et al. define T_X and T_Z as the decay times of the cat's ⟨Z⟩ and ⟨X⟩. So one formula covers both qubits.

**Noise bias** is p_phase / p_bit. For short steps it is close to T_bit / T_phase.

| Device | T_bit | T_phase | Bias |
| --- | --- | --- | --- |
| Cat, \|α\|² = 2.5 | 313 µs | 2.7 µs | about 116 |
| Cat, \|α\|² = 11.3 | 15 s | 490 ns | about 3 × 10⁷ |
| Transmon, Willow mean | 68 µs | 89 µs | 0.76 |
| Transmon, best Ta device | 0.36 ms | 0.20 ms | 1.8 |

**What it means.** A transmon's two errors come at similar rates, so its code must fix both. A big cat almost never bit-flips, so a code only has to fix phase-flips. That is why a line of cat qubits can do the job of a 2D grid (Guillaud and Mirrahimi 2019).

**The catch.** The cat's phase-flips are more common than a transmon's. In 25 ns the |α|² = 11.3 cat phase-flips with chance 2.5%; the Willow transmon 0.014%.

**Fine print.** Willow's T2 uses dynamical decoupling (CPMG), and its numbers are means over 105 qubits. The Ta device numbers are its best T1 and echo T2. The check `qubit.flip_probability` shows where the formula fails: slow frequency noise makes ⟨X⟩ decay faster than e^(−t/T2).

**Try it.** In the *Noise bias* view, drag the idle time.

Sources: reglade2024 (Fig. 3), willow2025 (Sec. II, Supplement Sec. I), place2021, guillaud2019 (Sec. II), qubitlab_hand_calcs (step 7).
