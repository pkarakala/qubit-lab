# Changelog

Notable changes per release. Dates are when the tag was pushed.

## 1.1.0 (2026-11-13)

Two support layers for "wrong model" and "drift" lessons, each checked against the paper it follows.

- **Pulses under the wrong model.** A pulse can be scored on a model other than the one it was designed for (a detuning or a different anharmonicity), and GRAPE can average over several models to make a pulse robust (kobzar2004, Eq. 1). New formula `qubit.detuned_pi_error`, 1 - F = (2/3)(pi Delta tau)^2, derived in hand calc step 8 and checked. At 20 ns, ensemble GRAPE over +-2 MHz cuts the worst-case error from 3.7e-3 to 9.8e-5.
- **Single-shot readout with relaxation and drift.** Heterodyne IQ records for |g> and |e> from Dynamiqs' diffusive stochastic Schrodinger equation, with T1 during the window and a measurement efficiency eta; changing chi, the detuning, eta or T1 between batches gives a drifting readout. Assignment fidelity is checked against gambetta2007 Eqs. (2.6) and (3.8).
- **Checks.** 102 comparisons, all as predicted (up from 94). The shipped results are regenerated for this code.

## 1.0.0 (2026-11-13)

The first public release: an interactive course in superconducting qubits where every equation is cited and checked against a simulation.

**Views (12).** Phase space, Flip rates, Transmon, Qubit control, Readout, Pulse optimizer, Two-qubit gates, Error correction, Noise bias, Architectures, Playground and Checks. Every view has Share (a link that restores its inputs) and Copy as Python (a script that reproduces it).

**Physics.**
- Cat qubits: two-photon dissipation and Kerr cats in Dynamiqs, Liouvillian flip rates, the e^(−2|α|²) bit-flip law.
- Transmons: scqubits spectra against Koch et al. 2007, control on a 4-level transmon (Rabi, DRAG, T1, Ramsey, echo), GRAPE pulse design through Dynamiqs with optax.
- Readout: χ from the dressed spectrum against three textbook formulas, pointer states in the IQ plane, measurement-induced dephasing against Gambetta et al. 2006.
- Two-qubit gates: a transmon CZ through the 11-20 crossing, always-on ZZ, and a bias-preserving cat CNOT against Guillaud and Mirrahimi 2019 and Chamberland et al. 2022.
- Error correction: surface and repetition code memories with Stim and PyMatching, a repetition code of cats, and what a drifting qubit does to a calibrated decoder.
- Architecture gallery: six designs (Xmon, tantalum transmon, tunable coupler, fluxonium, dissipative cat, Kerr cat) as 3D chips that explode into layers, every size cited or marked schematic.

**Checks.** 94 comparisons of a formula with a simulation, each with its verdict predicted before the run, all as predicted. Listed in the Checks view and docs/reference/checks.md.

**Course.** A five-part course in docs/learn/, from states and noise to qubit designs, control, gates and error correction, with a Try it on every page.

**Speed.** Every view's defaults and 73 settings from published experiments are solved in the background on start; JAX compilations are cached on disk; flip rates stream in point by point; the check results ship with the code; slow views show a live timer.

**Running it.** `pipx run qubitlab serve` (once on PyPI) or a Docker image (docs/deploy.md).
