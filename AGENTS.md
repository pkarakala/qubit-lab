# AGENTS.md

Rules for coding agents (and people) working in this repo. Humans start at [README.md](README.md) and [docs/learn/](docs/learn/), not here.

## The three layers

| Layer | Folder | Contains | Must not |
| --- | --- | --- | --- |
| Theory | `src/qubitlab/theory/` | SymPy expressions with sources, compiled to JAX | import Dynamiqs or `qubitlab.sim` |
| Simulation | `src/qubitlab/sim/` | Dynamiqs and JAX solvers | import SymPy or `qubitlab.theory` |
| Checks | `src/qubitlab/checks/` | Theory vs simulation, with a predicted verdict per regime | hide a disagreement |

`tests/test_layers.py` enforces the import rules.

## Rules

1. **No unsourced physics.** A `Formula` needs at least one `Source` with a key in `references.bib` and a locator (equation, section or figure). A derivation of our own is `Kind.DERIVED` and points at a script in `docs/hand_calcs/`.
2. **Never cite from memory.** Read the paper before you write a locator. If a detail was not checked, say so in the bib entry's `unverified` field.
3. **Predict before you run.** Each check regime states `AGREE` or `DIVERGE` from the formula's `valid_when`, before you see the numbers. If the result surprises you, that is a finding: report it, do not tune the tolerance to hide it.
4. **Expected test values come from hand calcs or papers,** never from running the code.
5. **Conventions are fixed:** beta = x + ip, hbar = 1/2 Wigner convention, alpha real and positive, rates in units of kappa_2 (or K), transmon energies in units of E_C or GHz. Alice & Bob's basis: |0>, |1> ~ |+alpha>, |-alpha>. Some papers swap bit and phase flip; note it in the `Source.note`.
6. **Human docs stay short.** A `docs/learn/` page is at most 350 words and ends with a `Sources:` line. `tests/test_docs.py` enforces both. Long material goes in `docs/reference/`, which is generated.
7. **Keep diffs small and on one topic.** Write plainly: short sentences, no filler, no em dashes.

## Map

```
src/qubitlab/provenance.py      Source, Kind, bib parser
src/qubitlab/theory/            cat.py, transmon.py, qubit.py (control and decoherence), registry.py
src/qubitlab/sim/               cat.py (Lindblad, Wigner, Liouvillian rates), transmon.py (charge basis),
                                transmon_dynamics.py (pulses, T1/T2, Ramsey, echo; 4 levels)
src/qubitlab/checks/            base.py, cat_checks.py, transmon_checks.py, qubit_checks.py
src/qubitlab/api.py             FastAPI for the explorer
docs/learn/                   human layer, one screen per page
docs/reference/               generated: formulas, checks (scripts/make_reference.py)
docs/hand_calcs/              derivations behind DERIVED sources
web/                          React + three.js explorer
```

## Commands

```bash
uv sync --all-groups
uv run ruff format . && uv run ruff check . && uv run mypy && uv run pytest -q
uv run pytest -q -m "not slow"                 # quick loop, under a minute
uv run python scripts/make_reference.py        # after changing a formula or check
cd web && npm run typecheck && npm run test:unit && npm run e2e
```
