# Qubit Lab

Cat qubits and transmons, from closed-form theory to Dynamiqs simulation, in a 3D explorer.

A learning project. Alice & Bob build cat qubits; Qolab builds transmon-family qubits. Both are superconducting circuits, but they bet on different physics. qubitlab puts the equations from the papers next to a full simulation, so you can see where each formula holds and where it breaks.

![A cat losing its fringes to photon loss, with the ideal cat from theory as a wireframe](docs/img/phase_space.png)

| | |
| --- | --- |
| ![Flip rates](docs/img/flip_rates.png) | ![Transmon](docs/img/transmon.png) |

## Findings

1. In the ideal two-photon model, each added photon cuts the bit-flip rate by e² ≈ 7.4 once |α|² ≳ 5. The 2020 experiment by Lescanne et al. measured 4.2.
2. The phase-flip rate κ₁|α|² matches simulation to 0.2% for |α|² ≥ 2, and is off by 86% at |α|² = 0.3, where there is no cat to speak of.
3. Koch et al.'s transmon frequency holds to 1% for E_J/E_C ≥ 20, but the true anharmonicity is 10 to 15% larger than E_C at E_J/E_C = 50 to 100.

## Run

```bash
uv sync --all-groups
cd web && npm install && cd ..
./scripts/dev.sh           # API on :8000, explorer on :5173
uv run qubitlab checks       # every formula vs the simulation, in the terminal
```

## How it is built

Three layers, kept apart by a test:

- **Theory** (`src/qubitlab/theory/`): each equation is a SymPy expression with its paper and equation number, compiled to JAX.
- **Simulation** (`src/qubitlab/sim/`): Dynamiqs for the cat's Lindblad equation, JAX for the transmon's spectrum.
- **Checks** (`src/qubitlab/checks/`): theory vs simulation, with the verdict predicted before running.

## Learn

Start with [docs/learn/](docs/learn/): five one-screen pages. Every equation is listed with its source in [docs/reference/formulas.md](docs/reference/formulas.md), and every comparison in [docs/reference/checks.md](docs/reference/checks.md).

## Limits

The cat model is the standard effective one: a single mode with two-photon dissipation, no buffer mode, no gates yet. Rates are in units of κ₂, not hertz. Hardware numbers are shown as cited facts, never fitted. Not affiliated with Alice & Bob, Qolab, or any company cited.

Parts of this repo were written with AI coding assistants, directed and reviewed by me. Agent rules are in [AGENTS.md](AGENTS.md).

Apache-2.0 · Pranav Reddy
