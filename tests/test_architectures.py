"""The gallery: every part says where its size comes from, and the scenes load and solve."""

import re

import pytest
from fastapi.testclient import TestClient

from qubitlab.api import app
from qubitlab.architectures import scenes
from qubitlab.provenance import bib_entries

client = TestClient(app)
KEY = re.compile(r"\b([a-z]+\d{4}[a-z]*)\b")


def test_every_part_is_sourced_or_marked_schematic() -> None:
    keys = set(bib_entries())
    for scene in scenes().values():
        assert set(scene.refs) <= keys, scene.id
        for part in scene.parts:
            assert part.dims_source.strip(), f"{scene.id}.{part.id} has no dims_source"
            if part.dims_class == "MEASURED":
                cited = set(KEY.findall(part.dims_source))
                assert cited and cited <= keys, f"{scene.id}.{part.id} cites {cited}"
            if part.enlarged:
                assert part.draw_size is not None, (
                    f"{scene.id}.{part.id} enlarged without draw_size"
                )


def test_physics_says_where_its_numbers_come_from() -> None:
    for scene in scenes().values():
        assert scene.physics["source"] and scene.physics["kind"], scene.id


def test_gallery_api_and_a_spectrum() -> None:
    data = client.get("/api/architectures").json()
    assert {s["id"] for s in data} >= {"xmon", "fluxonium", "kerr_cat"}
    sp = client.get("/api/architectures/xmon/spectrum").json()["sim"]
    # barends2013 (main text p. 2): about 6 GHz unbiased. The tunable transmon at zero flux:
    zero = len(sp["x"]) // 2
    assert sp["levels"][zero][1] == pytest.approx(6.0, abs=0.15)
    assert client.get("/api/architectures/nope/spectrum").status_code == 404
