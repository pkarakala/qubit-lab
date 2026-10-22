# Contributing

Corrections are the most useful contribution: a wrong equation, a misread paper, a check that should fail.

## Report a mistake

Open an issue. Say what is wrong, where you saw it (explorer, a `docs/learn/` page, or a file), and what it should be, with a source. No code needed.

## Set up

```bash
git clone https://github.com/pkarakala/qubit-lab && cd qubit-lab
uv sync --all-groups
cd web && npm install && cd ..
./scripts/dev.sh        # API on :8000, explorer on :5173
```

## Add or fix a formula

1. Read the paper. Note the equation number.
2. Add the BibTeX entry to `references.bib` if it is new.
3. Write the `Formula` in `src/qubitlab/theory/` with its `Source` and `valid_when`.
4. Add a check in `src/qubitlab/checks/`: pick regimes where it should agree and one where it should not, and write the expected verdict first.
5. Run `uv run python scripts/make_reference.py` and the tests.

## Before a pull request

```bash
uv run ruff format . && uv run ruff check . && uv run mypy && uv run pytest -q
cd web && npm run typecheck && npm run test:unit
```

If you used an AI assistant, point it at [AGENTS.md](AGENTS.md) and say so in the PR.

## Release (maintainer)

1. Set the version: `uv version 0.2.0`, commit, merge to main.
2. Tag and push: `git tag v0.2.0 && git push origin v0.2.0`.
3. The release workflow builds the explorer into the wheel, smoke-tests `qubitlab serve`, publishes to PyPI and makes a GitHub release.

PyPI needs a one-time trusted publisher for this repo (workflow `release.yml`, environment `pypi`).
