# Cat vs transmon

Same fridge, same aluminum junctions, different bet.

| | Transmon | Cat qubit |
| --- | --- | --- |
| Where the bit lives | Lowest two levels of an anharmonic circuit | Two coherent states of a harmonic resonator |
| What holds it there | The circuit's own energy levels | Two-photon dissipation (or a Kerr Hamiltonian) |
| Bit-flips vs phase-flips | Similar rates | Bit-flips exponentially rarer |
| Error correction | 2D surface code | 1D repetition code |
| Main knob | E_J/E_C | Photon number \|α\|² |
| Company in this repo | Qolab | Alice & Bob |

**The surface code** protects against both error types, so it needs a 2D grid. Fowler et al. 2012 is the standard reference.

**The repetition code** only fights one error type, so a line of qubits is enough. Guillaud and Mirrahimi 2019 showed that cat qubits make this work, as long as gates also keep bit-flips rare.

**What each side must prove.** Transmons must get every physical qubit good and cheap at scale, which is Qolab's focus. Cats must keep bit-flips suppressed *during* gates, which is Alice & Bob's next milestone.

Sources: fowler2012, guillaud2019 (Sec. II), koch2007, tqi2025alicebob, tqi2024qolab.
