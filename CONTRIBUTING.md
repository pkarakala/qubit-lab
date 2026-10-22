# Contributing

Corrections are the most useful contribution: a wrong equation, a misread paper, a check that should fail.

## Report a mistake

Open an issue. Say what is wrong, where you saw it (explorer, a `docs/learn/` page, or a file), and what it should be, with a source. No code needed.

## Set up

```bash
git clone https://github.com/pkarakala/qubitlab && cd qubitlab
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
