"""Where every equation and number comes from.

Each formula in `qubitlab.theory` carries one or more `Source` objects. A test checks that
every `Source.key` exists in `references.bib`, so nothing in the theory layer is unsourced.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from functools import cache
from pathlib import Path

BIB_PATH = Path(__file__).resolve().parents[2] / "references.bib"


class Kind(StrEnum):
    """How much to trust a source, strongest first."""

    PEER_REVIEWED = "peer_reviewed"
    TEXTBOOK = "textbook"
    PREPRINT = "preprint"
    COMPANY_CLAIM = "company_claim"
    DERIVED = "derived"  # worked out in docs/hand_calcs from a cited starting point
    ASSUMPTION = "assumption"


@dataclass(frozen=True)
class Source:
    key: str  # BibTeX key in references.bib
    kind: Kind
    locator: str  # equation, section or figure, e.g. "Eq. (2.5)"
    note: str = ""

    def short(self) -> str:
        return f"{self.key}, {self.locator}"


@cache
def bib_entries() -> dict[str, dict[str, str]]:
    """Parse references.bib into {key: {field: value}}. Small parser, no dependency."""
    text = BIB_PATH.read_text(encoding="utf-8")
    entries: dict[str, dict[str, str]] = {}
    for match in re.finditer(r"@(\w+)\{([^,]+),(.*?)\n\}", text, flags=re.S):
        kind, key, body = match.groups()
        fields = {"type": kind.lower()}
        for f in re.finditer(r"(\w+)\s*=\s*\{(.*?)\},?\s*$", body, flags=re.M):
            fields[f.group(1).lower()] = re.sub(r"\s+", " ", f.group(2)).strip()
        entries[key.strip()] = fields
    return entries


def cite(key: str) -> dict[str, str]:
    """Return the bib entry for `key`, or raise if it is missing."""
    entries = bib_entries()
    if key not in entries:
        raise KeyError(f"{key!r} is not in references.bib")
    return entries[key]
