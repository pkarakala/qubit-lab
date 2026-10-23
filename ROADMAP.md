# Roadmap

What is built, what is next, and the decisions behind it. Update this file when a milestone lands or a decision changes. Rules for doing the work are in [AGENTS.md](AGENTS.md). The detailed plan for the next items is [docs/plan.md](docs/plan.md).

## Done

| Milestone | What it gives |
| --- | --- |
| 1. Cat phase space | Wigner function of a cat qubit growing and decaying, ideal cats from theory as a ghost |
| 2. Bit-flip vs phase-flip | Liouvillian flip rates vs \|α\|², the e^(−2\|α\|²) law, Lescanne 2020 as a measured fact |
| 3. Transmon spectrum | scqubits bands vs Koch et al. closed forms, charge dispersion in 3D |
| 3b. Transmon control | Rabi, π pulse, DRAG, T1, Ramsey, echo on a 4-level transmon; Bloch sphere view |
| 4. Noise bias (cat vs transmon) | Flip chance per idle step and noise bias at measured numbers (Réglade 2024, Willow 2025, Place 2021) |
| 6. Repetition vs surface code | Stim and PyMatching memory experiments, logical error vs distance, 3D space-time decoding |
| Extras | Pulse optimizer (GRAPE through Dynamiqs), Playground for typed one-mode models, share links, Copy as Python |
| Release tooling | `qubitlab serve` ships the explorer in the wheel; release workflow to PyPI; Docker image and Hugging Face Space workflow |

## Next

In order, one PR each, merged before the next starts. Details in [docs/plan.md](docs/plan.md).

| # | Milestone | Scope | Estimate |
| --- | --- | --- | --- |
| 7 | Release v0.2.0 | Turn on PyPI trusted publishing and the Space (docs/deploy.md), tag v0.2.0, Zenodo DOI. The code is ready; the rest is setup. | 1 day |
| 9 | Readout | Dispersive shift from scqubits, resonator and transmon in Dynamiqs, IQ blobs and SNR vs time, checked against the papers. | 2 to 3 days |
| 10 | Two-qubit gates | Transmon CZ through the 11 to 20 avoided crossing; bias-preserving cat CNOT. | 3 to 5 days |
| 11 | Architecture gallery | The leading superconducting architectures as 3D chips that explode into layers, each with its spectrum and cited parameters. | 3 to 5 days |
| 5 | Chip layer (Palace) | Gmsh geometry of a transmon pad and readout resonator. Palace electrostatic solve gives the capacitance matrix and E_C; eigenmode solve gives frequencies and participation. Check against an analytic parallel-plate or CPW formula first. Results committed as data, so the explorer never needs Palace. | 1 to 2 weeks |
| 8 | Release v1.0 | A demo video, a written walk through the course, outreach to the Dynamiqs and scqubits communities. | 2 to 3 days |

Later: upstream a transmon example to Palace (open an issue first), a transmon mode in the Playground, a photonics bridge (microwave-to-optical).

## Decisions

- **Libraries first.** Dynamiqs for open-system dynamics, scqubits for circuit spectra, SymPy compiled to JAX for theory, Palace for electromagnetics. Our own code only where none fits (AGENTS.md rule 7).
- **Theory, simulation and checks stay in separate layers**, enforced by `tests/test_layers.py`.
- **Every formula is cited** to an equation number, or derived in `docs/hand_calcs/`.
- **Commits carry no AI co-author lines.** The repo is meant to read as open to any contributor.
- **Coding happens in a local agent session**; architecture decisions and progress are tracked with the maintainer.

## Open questions

- Date of the DuQuantum hackathon (sets whether the release or the chip layer comes first).
