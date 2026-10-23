"""Every preset is a valid request, cites a paper in references.bib, and is warmed."""

from qubitlab.provenance import bib_entries
from qubitlab.warm import preset_steps, presets

VIEWS = {"phase", "rates", "transmon", "qubit", "pulse", "qec", "qec_cat", "compare"}


def test_presets_cite_papers_in_the_bib() -> None:
    bib = bib_entries()
    for p in presets():
        assert p["view"] in VIEWS, p["name"]
        assert p["refs"], p["name"]
        assert all(r in bib for r in p["refs"]), (p["name"], p["refs"])


def test_presets_are_valid_requests() -> None:
    # Builds every request model; a parameter out of range raises here.
    cached = [p for p in presets() if p["view"] not in ("transmon", "qec")]
    assert len(preset_steps()) == len(cached)


def test_preset_names_are_unique_per_view() -> None:
    keys = [(p["view"], p["name"]) for p in presets()]
    assert len(keys) == len(set(keys))
