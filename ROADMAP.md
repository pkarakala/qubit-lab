# Roadmap

What is built, what is next, and the decisions behind it. Update this file when a milestone lands or a decision changes. Rules for doing the work are in [AGENTS.md](AGENTS.md).

## Done

| Milestone | What it gives |
| --- | --- |
| 1. Cat phase space | Wigner function of a cat qubit growing and decaying, ideal cats from theory as a ghost |
| 2. Bit-flip vs phase-flip | Liouvillian flip rates vs \|α\|², the e^(−2\|α\|²) law, Lescanne 2020 as a measured fact |
| 3. Transmon spectrum | scqubits bands vs Koch et al. closed forms, charge dispersion in 3D |
| 3b. Transmon control | Rabi, π pulse, DRAG, T1, Ramsey, echo on a 4-level transmon; Bloch sphere view |

## Next

| # | Milestone | Scope | Estimate |
| --- | --- | --- | --- |
| 4 | Cat vs transmon | One view: error rates side by side at matched hardware numbers, and qubits per logical qubit. Surface code from Fowler 2012 (Eqs. 10 and 11, n_q = (2d−1)²). Repetition code for cats needs a verified source first; Guillaud 2019 did not give a closed form. | 2 to 3 days |
| 5 | Chip layer (Palace) | Gmsh geometry of a transmon pad and readout resonator. Palace electrostatic solve gives the capacitance matrix and E_C; eigenmode solve gives frequencies and participation. Check against an analytic parallel-plate or CPW formula first. Palace runs offline on a laptop; results are committed as data, so the explorer never needs Palace. | 1 to 2 weeks |
| 6 | Repetition vs surface code | Logical error vs code distance for both qubit types, with the overhead in physical qubits. | 3 to 5 days |
| 7 | Release v1.0 | Host the explorer (static front end plus precomputed data, or a small API host), a short demo video, tagged release with CITATION.cff. | 2 to 3 days |

Total to v1.0: about 3 to 4 weeks of part-time work. Milestone 5 carries most of the uncertainty.

Later: upstream a transmon example to Palace (open an issue first), bias-preserving gates for cats.

## Decisions

- **Libraries first.** Dynamiqs for open-system dynamics, scqubits for circuit spectra, SymPy compiled to JAX for theory, Palace for electromagnetics. Our own code only where none fits (AGENTS.md rule 7).
- **Theory, simulation and checks stay in separate layers**, enforced by `tests/test_layers.py`.
- **Every formula is cited** to an equation number, or derived in `docs/hand_calcs/`.
- **Commits carry no AI co-author lines.** The repo is meant to read as open to any contributor.
- **Coding happens in a local agent session**; architecture decisions and progress are tracked with the maintainer.

## Open questions

- Date of the DuQuantum hackathon (sets whether 4 or 5 comes first).
- Where to host the explorer for v1.0.
