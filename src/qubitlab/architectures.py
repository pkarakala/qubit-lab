"""The architecture gallery: one 3D scene per qubit design, each part with a cited size.

Scenes live in data/architectures/*.toml, in micrometres. A part's `dims_class` says where its
size comes from: MEASURED (a paper and a locator in `dims_source`) or SCHEMATIC (with the reason).
Parts too small to see are drawn at `draw_size` and flagged `enlarged`; `size` stays the true one.
"""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field
from functools import cache
from pathlib import Path
from typing import Any, Literal

SCENES = Path(__file__).resolve().parent / "data" / "architectures"
Vec = tuple[float, float, float]


@dataclass(frozen=True)
class Part:
    id: str
    name: str
    role: str
    size: Vec
    position: Vec
    material: str
    explode: Vec
    dims_class: Literal["MEASURED", "SCHEMATIC"]
    dims_source: str
    draw_size: Vec | None = None
    enlarged: bool = False


@dataclass(frozen=True)
class Scene:
    id: str
    name: str
    family: str
    summary: str
    refs: tuple[str, ...]
    physics: dict[str, Any]
    parts: tuple[Part, ...] = field(default_factory=tuple)

    def describe(self) -> dict[str, Any]:
        return asdict(self)


def _vec(v: list[float]) -> Vec:
    if len(v) != 3:
        raise ValueError(f"expected 3 numbers, got {v}")
    return (float(v[0]), float(v[1]), float(v[2]))


def _load(path: Path) -> Scene:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    parts = []
    for p in raw["part"]:
        if p["dims_class"] not in ("MEASURED", "SCHEMATIC"):
            raise ValueError(f"{path.name}: {p['id']} has dims_class {p['dims_class']!r}")
        parts.append(
            Part(
                id=p["id"],
                name=p["name"],
                role=p["role"],
                size=_vec(p["size"]),
                position=_vec(p["position"]),
                material=p["material"],
                explode=_vec(p["explode"]),
                dims_class=p["dims_class"],
                dims_source=p["dims_source"],
                draw_size=_vec(p["draw_size"]) if "draw_size" in p else None,
                enlarged=bool(p.get("enlarged", False)),
            )
        )
    s = raw["scene"]
    return Scene(
        s["id"],
        s["name"],
        s["family"],
        s["summary"],
        tuple(s["refs"]),
        raw["physics"],
        tuple(parts),
    )


@cache
def scenes() -> dict[str, Scene]:
    out = {}
    for path in sorted(SCENES.glob("*.toml")):
        scene = _load(path)
        out[scene.id] = scene
    return out
