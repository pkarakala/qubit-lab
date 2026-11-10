"""Every preset is a valid request, cites a paper in references.bib, and is warmed."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from qubitlab import cache
from qubitlab.api import app
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


@pytest.mark.slow
def test_every_phase_preset_solves(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Long Kerr and two-photon solves (Grimm 2020, Hajr 2024, Chamberland 2022) once ran out
    # of solver steps and answered 500.
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)  # solve; do not read a cached answer
    client = TestClient(app)
    for p in presets():
        if p["view"] == "phase":
            r = client.post("/api/cat/evolve", json=p["params"])
            assert r.status_code == 200, (p["name"], r.text)
