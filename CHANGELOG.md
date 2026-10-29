# Changelog

Notable changes per release. Dates are when the tag was pushed.

## 1.0.0

The first public release: an interactive course in superconducting qubits where every equation is cited and checked against a simulation.

**Views (12).** Phase space, Flip rates, Transmon, Qubit control, Readout, Pulse optimizer, Two-qubit gates, Error correction, Noise bias, Architectures, Playground and Checks. Every view has Share (a link that restores its inputs) and Copy as Python (a script that reproduces it).

**Physics.**
- Cat qubits: two-photon dissipation and Kerr cats in Dynamiqs, Liouvillian flip rates, the e^(−2|α|²) bit-flip law.
- Transmons: scqubits spectra against Koch et al. 2007, control on a 4-level transmon (Rabi, DRAG, T1, Ramsey, echo), GRAPE pulse design through Dynamiqs with optax.
- Readout: χ from the dressed spectrum against three textbook formulas, pointer states in the IQ plane, measurement-induced dephasing against Gambetta et al. 2006.
- Two-qubit gates: a transmon CZ through the 11-20 crossing, always-on ZZ, and a bias-preserving cat CNOT against Guillaud and Mirrahimi 2019 and Chamberland et al. 2022.
- Error correction: surface and repetition code memories with Stim and PyMatching, and a repetition code of cats.
- Architecture gallery: six designs (Xmon, tantalum transmon, tunable coupler, fluxonium, dissipative cat, Kerr cat) as 3D chips that explode into layers, every size cited or marked schematic.

**Checks.** 94 comparisons of a formula with a simulation, each with its verdict predicted before the run, all as predicted. Listed in the Checks view and docs/reference/checks.md.

**Course.** A five-part course in docs/learn/, from states and noise to qubit designs, control, gates and error correction, with a Try it on every page.

**Speed.** Every view's defaults and 73 settings from published experiments are solved in the background on start; JAX compilations are cached on disk; flip rates stream in point by point; the check results ship with the code; slow views show a live timer.

**Running it.** `pipx run qubitlab serve` (once on PyPI), a Docker image, or a Hugging Face Space (docs/deploy.md).
