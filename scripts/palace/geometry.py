"""Gmsh geometry and meshes for the Palace runs. Lengths in micrometres (Palace L0 = 1e-6).

Three models:
- `parallel_plate`: two square plates in a vacuum box, the box as ground.
- `cpw`: a straight piece of coplanar waveguide on silicon, half model as below. Two lengths
  give the capacitance per length with the end effects cancelled.
- `device`: a grounded transmon pad on silicon, coupled to the open end of a half-wave CPW
  resonator. Metal is zero thickness on the substrate surface. The junction is a small
  rectangle across the pad's gap, used as a lumped inductor in the eigenmode solve.
  The layout is mirror symmetric about y = 0, so only y >= 0 is meshed. The plane y = 0 gets
  no boundary condition: Palace then imposes zero normal field for the potential and zero
  tangential magnetic field for the eigenmodes, which is the even (fundamental) symmetry.
  Capacitances of the half model are half the full ones, and the half junction carries
  half the current, so its port inductance is 2 L_J.

Each function writes a Gmsh 2.2 mesh and returns the physical group numbers it used.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import gmsh


@dataclass(frozen=True)
class Device:
    """Sizes in micrometres. Defaults: Si substrate, 10/6 um CPW as in goeppl2008 (w, ~s)."""

    pad: float = 400.0  # qubit pad, square side
    pad_gap: float = 30.0  # pad to ground plane
    junction_width: float = 10.0  # lumped port width, along y
    w: float = 10.0  # CPW centre width
    s: float = 6.0  # CPW gap
    length: float = 8000.0  # half-wave resonator length
    coupler_len: float = 100.0  # coupling bar at the resonator end, along y
    coupler_width: float = 20.0  # along x
    coupling_gap: float = 15.0  # coupler bar to pad
    margin: float = 300.0  # ground plane beyond the structures
    substrate: float = 500.0  # silicon thickness
    vacuum: float = 500.0  # vacuum above the chip
    mesh_fine: float = 2.0  # element size at metal edges
    mesh_coarse: float = 100.0  # element size far away
    mesh_grow: float = 150.0  # distance over which the size grows from fine to coarse


DEFAULT = Device()


def _init(name: str) -> None:
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(name)


def _size_field(edges: list[int], fine: float, coarse: float, dist_max: float) -> None:
    f = gmsh.model.mesh.field
    dist = f.add("Distance")
    f.setNumbers(dist, "CurvesList", edges)
    # Distance is measured to points sampled along each curve: space them a mesh size apart,
    # or long edges (the 8 mm resonator) end up with coarse elements between samples.
    longest = max(gmsh.model.occ.getMass(1, c) for c in edges)
    f.setNumber(dist, "Sampling", int(longest / fine) + 2)
    thr = f.add("Threshold")
    f.setNumber(thr, "InField", dist)
    f.setNumber(thr, "SizeMin", fine)
    f.setNumber(thr, "SizeMax", coarse)
    f.setNumber(thr, "DistMin", fine)
    f.setNumber(thr, "DistMax", dist_max)
    f.setAsBackgroundMesh(thr)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)


def _write(path: Path) -> dict[str, int]:
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.option.setNumber("Mesh.Algorithm3D", 10)  # HXT, parallel
    gmsh.model.mesh.generate(3)
    path.parent.mkdir(parents=True, exist_ok=True)
    gmsh.write(str(path))
    _, tets, _ = gmsh.model.mesh.getElements(3)
    n = sum(len(t) for t in tets)
    gmsh.finalize()
    return {"tetrahedra": int(n)}


def parallel_plate(path: Path, side: float, gap: float, thickness: float) -> dict[str, object]:
    """Plates of side `side` at z = +-(gap/2 + thickness/2), in a box 3 sides wide."""
    _init("parallel_plate")
    occ = gmsh.model.occ
    half = 1.5 * side + 5 * gap
    box = occ.addBox(-half, -half, -half, 2 * half, 2 * half, 2 * half)
    top = occ.addBox(-side / 2, -side / 2, gap / 2, side, side, thickness)
    bot = occ.addBox(-side / 2, -side / 2, -gap / 2 - thickness, side, side, thickness)
    air, _ = occ.cut([(3, box)], [(3, top), (3, bot)])
    occ.synchronize()
    vol = air[0][1]
    surfaces = [t for _, t in gmsh.model.getBoundary([(3, vol)], oriented=False)]
    groups: dict[str, list[int]] = {"outer": [], "top": [], "bottom": []}
    for sf in surfaces:
        x0, y0, z0, x1, y1, z1 = gmsh.model.getBoundingBox(2, sf)
        on_box = any(
            abs(a - b) < 1e-6 and abs(abs(a) - half) < 1e-6
            for a, b in ((x0, x1), (y0, y1), (z0, z1))
        )
        if on_box:
            groups["outer"].append(sf)
            continue
        groups["top" if (z0 + z1) / 2 > 0 else "bottom"].append(sf)
    tags = {"air": 1, "outer": 2, "top": 3, "bottom": 4}
    gmsh.model.addPhysicalGroup(3, [vol], tags["air"])
    for name in ("outer", "top", "bottom"):
        gmsh.model.addPhysicalGroup(2, groups[name], tags[name])
    plate_edges = [
        c
        for sf in groups["top"] + groups["bottom"]
        for _, c in gmsh.model.getBoundary([(2, sf)], oriented=False)
    ]
    _size_field(plate_edges, fine=gap / 3, coarse=half / 6, dist_max=max(side, 10 * gap))
    stats = _write(path)
    return {"attributes": tags, "side": side, "gap": gap, "thickness": thickness, **stats}


def device(path: Path, dev: Device = DEFAULT) -> dict[str, object]:
    """The transmon and resonator. x runs along the resonator; z = 0 is the chip surface."""
    _init("device")
    occ = gmsh.model.occ
    g = dev.pad_gap
    pad_x0, pad_x1 = g, g + dev.pad
    # The coupler bar sits in the clearance, coupling_gap from the pad's right edge.
    cpl_x0 = pad_x1 + dev.coupling_gap
    cpl_x1 = cpl_x0 + dev.coupler_width
    # Clearance around pad and coupler: x in [0, coupler + g], y in [-(pad/2 + g), pad/2 + g].
    clear_x1 = cpl_x1 + g
    res_x0 = cpl_x1  # the centre line starts at the coupler
    res_x1 = res_x0 + dev.length
    x_min, x_max = -dev.margin, res_x1 + dev.s + dev.margin
    y_half = max(dev.pad / 2 + g, dev.w / 2 + dev.s) + dev.margin
    # Metal pieces on z = 0, y >= 0.
    pad = occ.addRectangle(pad_x0, 0, 0, dev.pad, dev.pad / 2)
    coupler = occ.addRectangle(cpl_x0, 0, 0, dev.coupler_width, dev.coupler_len / 2)
    line = occ.addRectangle(res_x0, 0, 0, dev.length, dev.w / 2)
    centre, _ = occ.fuse([(2, coupler)], [(2, line)])
    junction = occ.addRectangle(0, 0, 0, g, dev.junction_width / 2)
    # Non-metal openings in the ground plane: the pad clearance and the CPW gap.
    hole = occ.addRectangle(0, 0, 0, clear_x1, dev.pad / 2 + g)
    trench = occ.addRectangle(clear_x1 - 1, 0, 0, res_x1 + dev.s - clear_x1 + 1, dev.w / 2 + dev.s)
    opening, _ = occ.fuse([(2, hole)], [(2, trench)])
    sheet = occ.addRectangle(x_min, 0, 0, x_max - x_min, y_half)
    ground, _ = occ.cut([(2, sheet)], opening, removeTool=False)
    # Volumes.
    sub = occ.addBox(x_min, 0, -dev.substrate, x_max - x_min, y_half, dev.substrate)
    vac = occ.addBox(x_min, 0, 0, x_max - x_min, y_half, dev.vacuum)
    pieces = [(3, sub), (3, vac)] + ground + opening + [(2, pad)] + centre + [(2, junction)]
    _, children = occ.fragment(pieces, [])
    occ.synchronize()
    kids = dict(zip(pieces, children, strict=True))

    def of(inputs: list[tuple[int, int]]) -> set[int]:
        return {t for dt in inputs for d, t in kids[dt] if d == 2}

    junction_s = of([(2, junction)])
    pad_s = of([(2, pad)]) - junction_s
    centre_s = of(centre)
    ground_s = of(ground)
    gaps_s = of(opening) - pad_s - centre_s - junction_s
    sub_v = sorted({t for d, t in kids[(3, sub)] if d == 3})
    vac_v = sorted({t for d, t in kids[(3, vac)] if d == 3})
    surface = pad_s | centre_s | ground_s | gaps_s | junction_s
    if len(surface) != sum(map(len, (pad_s, centre_s, ground_s, gaps_s, junction_s))):
        raise ValueError("metal pieces overlap; Palace needs each boundary face in one group")

    def on_mirror(t: int) -> bool:
        _, y0, _, _, y1, _ = gmsh.model.getBoundingBox(2, t)
        return abs(y0) < 1e-6 and abs(y1) < 1e-6

    rest = [t for _, t in gmsh.model.getEntities(2) if t not in surface]
    groups = {
        "walls": sorted(t for t in rest if not on_mirror(t)),
        "mirror": sorted(t for t in rest if on_mirror(t)),
        "ground": sorted(ground_s),
        "pad": sorted(pad_s),
        "centre": sorted(centre_s),
        "junction": sorted(junction_s),
        "gaps": sorted(gaps_s),
    }
    tags = {
        "substrate": 1,
        "vacuum": 2,
        "walls": 3,
        "ground": 4,
        "pad": 5,
        "centre": 6,
        "junction": 7,
        "gaps": 8,
        "mirror": 9,
    }
    gmsh.model.addPhysicalGroup(3, sub_v, tags["substrate"])
    gmsh.model.addPhysicalGroup(3, vac_v, tags["vacuum"])
    for name in ("walls", "ground", "pad", "centre", "junction", "gaps", "mirror"):
        if groups[name]:
            gmsh.model.addPhysicalGroup(2, groups[name], tags[name])
    # Refine at the metal edges, but not along the mirror line, where the field is smooth.
    edges = [
        c
        for name in ("pad", "centre", "junction")
        for sf in groups[name]
        for _, c in gmsh.model.getBoundary([(2, sf)], oriented=False)
        if gmsh.model.getBoundingBox(1, c)[4] > 1e-6
    ]
    _size_field(edges, fine=dev.mesh_fine, coarse=dev.mesh_coarse, dist_max=dev.mesh_grow)
    stats = _write(path)
    return {
        "attributes": tags,
        "device": asdict(dev),
        "counts": {k: len(v) for k, v in groups.items()},
        **stats,
    }


def cpw(path: Path, length: float, dev: Device = DEFAULT, fine: float = 1.0) -> dict[str, object]:
    """A straight centre strip of `length` in its trench, ground plane around, y >= 0 only."""
    _init("cpw")
    occ = gmsh.model.occ
    margin, depth = 150.0, 300.0
    x_min, x_max = -margin, length + margin
    y_max = dev.w / 2 + dev.s + margin
    strip = occ.addRectangle(0, 0, 0, length, dev.w / 2)
    trench = occ.addRectangle(-dev.s, 0, 0, length + 2 * dev.s, dev.w / 2 + dev.s)
    sheet = occ.addRectangle(x_min, 0, 0, x_max - x_min, y_max)
    ground, _ = occ.cut([(2, sheet)], [(2, trench)], removeTool=False)
    sub = occ.addBox(x_min, 0, -depth, x_max - x_min, y_max, depth)
    vac = occ.addBox(x_min, 0, 0, x_max - x_min, y_max, depth)
    pieces = [(3, sub), (3, vac)] + ground + [(2, trench), (2, strip)]
    _, children = occ.fragment(pieces, [])
    occ.synchronize()
    kids = dict(zip(pieces, children, strict=True))
    strip_s = {t for d, t in kids[(2, strip)] if d == 2}
    ground_s = {t for dt in ground for d, t in kids[dt] if d == 2}
    gap_s = {t for d, t in kids[(2, trench)] if d == 2} - strip_s
    surface = strip_s | ground_s | gap_s
    rest = [t for _, t in gmsh.model.getEntities(2) if t not in surface]
    walls = [t for t in rest if gmsh.model.getBoundingBox(2, t)[4] > 1e-6]
    tags = {"substrate": 1, "vacuum": 2, "walls": 3, "ground": 4, "strip": 5}
    gmsh.model.addPhysicalGroup(3, [t for d, t in kids[(3, sub)] if d == 3], tags["substrate"])
    gmsh.model.addPhysicalGroup(3, [t for d, t in kids[(3, vac)] if d == 3], tags["vacuum"])
    gmsh.model.addPhysicalGroup(2, sorted(walls), tags["walls"])
    gmsh.model.addPhysicalGroup(2, sorted(ground_s), tags["ground"])
    gmsh.model.addPhysicalGroup(2, sorted(strip_s), tags["strip"])
    gmsh.model.addPhysicalGroup(2, sorted(gap_s), 6)
    edges = [
        c
        for sf in strip_s | ground_s
        for _, c in gmsh.model.getBoundary([(2, sf)], oriented=False)
        if gmsh.model.getBoundingBox(1, c)[4] > 1e-6
        and gmsh.model.getBoundingBox(1, c)[1] < y_max - 1e-6
    ]
    _size_field(edges, fine=fine, coarse=60.0, dist_max=100.0)
    stats = _write(path)
    return {"attributes": tags, "length": length, "fine": fine, **stats}
