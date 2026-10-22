"""The human layer stays short and sourced; the reference stays generated."""

import re
from pathlib import Path

from catlab.provenance import bib_entries
from scripts.make_reference import formulas_md

ROOT = Path(__file__).resolve().parents[1]
LEARN = ROOT / "docs" / "learn"
MAX_WORDS = 350


def test_learn_pages_are_short_and_sourced() -> None:
    pages = sorted(LEARN.glob("0*.md"))
    assert pages
    keys = set(bib_entries())
    for page in pages:
        text = page.read_text(encoding="utf-8")
        assert len(text.split()) <= MAX_WORDS, f"{page.name} is over {MAX_WORDS} words"
        match = re.search(r"^Sources: (.+)$", text, flags=re.M)
        assert match, f"{page.name} has no Sources line"
        cited = set(re.findall(r"\b([a-z]+\d{4}[a-z]*|catlab_hand_calcs)\b", match.group(1)))
        assert cited, f"{page.name} cites nothing"
        assert cited <= keys, f"{page.name} cites unknown keys {cited - keys}"


def test_no_em_dashes_in_human_docs() -> None:
    for page in [*LEARN.glob("*.md"), ROOT / "README.md"]:
        assert "—" not in page.read_text(encoding="utf-8"), page.name


def test_formula_reference_is_up_to_date() -> None:
    current = (ROOT / "docs" / "reference" / "formulas.md").read_text(encoding="utf-8")
    assert current == formulas_md(), "run: uv run python scripts/make_reference.py"
