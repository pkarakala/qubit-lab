# Two designs, two bets

Same fridge, same aluminum junctions, different physics.

| | Transmon | Cat qubit |
| --- | --- | --- |
| Where the bit lives | Lowest two levels of an anharmonic circuit | Two coherent states of a harmonic resonator |
| What holds it there | The circuit's own energy levels | Two-photon dissipation (or a Kerr Hamiltonian) |
| Bit-flips vs phase-flips | Similar rates | Bit-flips exponentially rarer |
| Error correction | 2D surface code | 1D repetition code |
| Main knob | E_J/E_C | Photon number \|α\|² |

**The surface code** protects against both error types, so it needs a 2D grid. Fowler et al. 2012 is the standard reference.

**The repetition code** only fights one error type, so a line of qubits is enough. Guillaud and Mirrahimi 2019 showed that cat qubits make this work, as long as gates also keep bit-flips rare.

**What each design must show.** A transmon processor needs every physical qubit good enough, and alike enough, at scale, because the surface code spends many of them per logical qubit. A cat processor must keep bit-flips suppressed *during* gates, not only at rest, or the repetition code loses its advantage.

**Try it.** In *Noise bias*, compare the bars for a cat and a transmon. Then in *Error correction*, compare the two codes at the same error rate.

Sources: fowler2012, guillaud2019 (Sec. II), koch2007.
