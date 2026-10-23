# Qubit Lab: open source cleanup and curriculum plan

Approved by the maintainer on 2026-10-22. This file is the working plan; ROADMAP.md keeps the one-line status.

## Goal

An open-source simulator for learning superconducting qubits, for students, engineers and researchers alike. It teaches concepts, not companies. Every equation is cited to a paper and an equation number, and checked against a simulation. Hardware numbers come from peer-reviewed papers only.

## Ground rules

1. **Libraries first.** We plug in maintained open-source solvers and cite them. We write our own code only where none fits.
2. **Theory, simulation and checks stay separate** (AGENTS.md). Theory is SymPy compiled to JAX. Simulation uses the libraries below. A check compares the two, with the verdict predicted before the run.
3. **Never cite from memory.** A paper is read before its equation number goes in the code. Anything not checked is marked `unverified` in references.bib.
4. **One PR per item, branched from main, merged one at a time.** No stacking. The next item starts from main after the previous one merges.

## Libraries we use, and what for

| Library | Role | New in this plan |
| --- | --- | --- |
| Dynamiqs (JAX) | Lindblad and Schrödinger solves, Wigner functions, gradients through solves | Readout, two-qubit gates |
| scqubits | Circuit spectra: transmon, tunable transmon, fluxonium, coupled systems (`HilbertSpace`), dispersive shifts | Readout χ, CZ spectrum, architecture gallery |
| JAX + Optax | Autodiff and optimizers (GRAPE already uses them) | Gate tuning |
| Stim + PyMatching | Error-correction circuits and decoding | Unchanged |
| SymPy | Closed forms compiled to JAX | Every new formula |
| SciPy | Root finding, fits, special functions | Readout SNR fits |
| Palace + Gmsh | Electrostatics and eigenmodes of a chip layout (offline, results committed as data) | Item 5 |
| gdsfactory (candidate) | Parametric chip geometry, exported to Gmsh | Item 5, if it fits; otherwise Gmsh alone |

Qiskit Metal was considered for chip geometry, but its last release is 0.1.5 and it pulls in a GUI stack, so it is not the default.

## Curriculum

The learn pages become one path in five parts. Each page stays under 350 words, ends with `Sources:`, and has a **Try it** box: a share link that opens the right view with the right settings.

| Part | Pages | Views |
| --- | --- | --- |
| 1. Foundations | States and Wigner functions; open systems and the Lindblad equation; T1, T2 and fidelity | Phase space, Playground |
| 2. Two qubit designs | The transmon; the cat qubit; two-photon dissipation; bit-flips and phase-flips | Transmon, Phase space, Flip rates |
| 3. Control and readout | Pulses and DRAG; optimal control; dispersive readout (new) | Qubit control, Pulse optimizer, Readout (new) |
| 4. Gates and errors | Two-qubit gates (new); noise bias; error correction | Two-qubit gates (new), Noise bias, Error correction |
| 5. Hardware | Architecture gallery (new); from layout to Hamiltonian (new) | Architectures (new), Chip (new) |

## Items

### 1. Cleanup (about half a day)

- README: the Findings section becomes short **What you can do** and **How we know it's right** sections (linking the Checks page). Neutral tagline. No company framing.
- Remove the company-claim facts (`alicebob2025.tx`, `alicebob2024.boson4`) and the three news entries in references.bib. Keep peer-reviewed papers whatever their affiliation (Lescanne 2020, Réglade 2024, Willow 2025, Place 2021), cited as papers.
- Code comments say "the convention of Réglade et al. 2024", not a company name. The footer drops company names.
- Rename the *Cat vs transmon* view to *Noise bias*. Learn page 5 becomes conceptual.
- Reorder the learn pages into the five parts above and add Try it boxes.
- ROADMAP points to this plan. Version 0.2.0, kept in pyproject.toml only (folds in PR #15).
- Done when: no company names outside citations, all tests pass, reference docs and screenshots regenerated.

### 2. Readout (2 to 3 days)

- **Physics.** Dispersive readout of a transmon through a resonator: the dispersive shift χ, resonator response for |0⟩ and |1⟩, measurement-induced dephasing, and signal-to-noise vs integration time.
- **Sources to read first** (none cited in code until read): Blais et al. 2004, Koch et al. 2007 Sec. III (transmon χ, already in the bib), Gambetta et al. 2006 (measurement-induced dephasing), the Blais et al. 2021 review of circuit QED.
- **Theory:** χ for a two-level and a multilevel transmon; the steady resonator field for each qubit state; the dephasing rate.
- **Simulation:** scqubits `HilbertSpace` (transmon plus `Oscillator`) for the exact χ. Dynamiqs for the driven, damped resonator coupled to the transmon, giving the IQ blobs and their separation over time.
- **Checks (predicted first):** two-level χ agrees far from resonance and diverges near it. The multilevel χ agrees with scqubits. The dephasing formula agrees at low photon number and diverges at high photon number.
- **View:** *Readout*. The two resonator responses, IQ blobs separating in real time, SNR vs time, with a slider for the drive photon number. **Learn page:** dispersive readout.

### 3. Two-qubit gates (3 to 5 days)

- **Transmon CZ.** Flux-tune |11⟩ into the |20⟩ avoided crossing and back. scqubits for the two-transmon spectrum and the gap; Dynamiqs for the gate. Optax tunes the pulse (reusing the GRAPE code). Theory: the avoided-crossing gap and the conditional phase. Sources to read: Strauch et al. 2003, DiCarlo et al. 2009, Krantz et al. 2019 review.
- **Cat CNOT.** A bias-preserving CNOT between two dissipative cats. Dynamiqs two-mode simulation; theory from Guillaud and Mirrahimi 2019 (in the bib, gate section to read) and Puri et al. 2020 for the bias-preserving argument. Checks: the phase-flip probability per gate against the paper's formula; the bit-flip suppression kept during the gate.
- **View:** *Two-qubit gates*. Populations and conditional phase during the gate, gate error vs duration, and for cats the bias before and after. **Learn page:** two-qubit gates.

### 4. Architecture gallery with exploded 3D views (3 to 5 days)

- **What it shows.** The leading superconducting architectures as 3D chip models that **explode** into layers: substrate, ground plane, junctions, capacitor pads, readout resonator, control and flux lines, couplers, airbridges and package. Click a part to see what it does and the parameter it sets.
- **Architectures (first set):** fixed-frequency transmon with readout resonator; tunable transmon with a tunable coupler; fluxonium; dissipative cat (storage mode plus buffer and nonlinear element); Kerr cat (SNAIL based). Later: dual-rail and 3D cavity bosonic qubits.
- **Physics beside each model:** the spectrum from scqubits (`Transmon`, `TunableTransmon`, `Fluxonium`) or the cat model already in the repo, with each design's key parameters cited to its paper.
- **Geometry:** one TOML scene per architecture (the pattern from the oicd twin). Each part carries a dimension, an `explode` offset and a `dims_class`: MEASURED (from a paper, with locator) or SCHEMATIC (with a one-line reason). The view labels schematic parts as not to scale.
- **Sources to read:** Koch 2007; Yan et al. 2018 (tunable coupler); Manucharyan et al. 2009 and a recent fluxonium paper; Lescanne 2020 and Réglade 2024 (cat device); Grimm 2020 (Kerr cat).
- **Done when:** five scenes explode and reassemble smoothly, every MEASURED dimension has a locator, and a Playwright test screenshots the gallery.

### 5. Chip layer with Palace (1 to 2 weeks)

- Gmsh (and gdsfactory if it fits) geometry of a transmon pad and readout resonator. A Palace electrostatic solve gives the capacitance matrix and E_C; an eigenmode solve gives the frequencies and participation ratios. Check first against a parallel-plate or CPW closed form.
- Results are committed as data, so the explorer never needs Palace. The transmon scene in the gallery shows the computed E_C and frequency next to the geometry that produced them.

## Order and timing

1, then 2, 3, 4 and 5, each merged before the next starts. About 3 to 4 weeks of part-time work. While a PR waits for review, the next item's papers get read, so no time is lost.

## Open questions for the maintainer

- Which career direction matters most: device design, control and calibration, or error correction? It sets how deep items 2 to 5 go.
- Whether to add a photonics bridge later (microwave-to-optical transduction), to connect this work with optical interconnects.
