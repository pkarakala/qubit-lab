"""Mesh, solve and collect the Palace runs behind `src/qubitlab/data/chip.json`.

Usage (Palace on PATH, or PALACE=/path/to/palace):
    uv run --group chip python scripts/palace/run.py [--np 4] [--work runs/palace]

Steps, each skipped when its output already exists in the work folder:
1. Parallel plates, side/gap = 100 and 4: electrostatic, mutual capacitance.
2. A straight CPW at two lengths and two element orders: capacitance per length.
3. The device: electrostatic, the Maxwell capacitance matrix of pad and resonator.
4. The device: eigenmodes at a typical junction, then a sweep of the junction inductance L_J
   through the resonator.
Then everything is written to src/qubitlab/data/chip.json, with the Palace version.

The version is recorded as Palace reports it. "-dirty" on the first run means the build
script was changed to read dependency patches from disk (the cloud machine could not download
them); the solver code was not changed.

Memory: the largest solve (device eigenmodes, order 2) needs about 8 GB.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

import geometry

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "src" / "qubitlab" / "data" / "chip.json"
EPS_SI = 11.45  # silicon at low temperature; an input, same in theory and simulation
PLATES = ((1000.0, 10.0), (40.0, 10.0))  # (side, gap) in um
CPW_LENGTHS = (200.0, 400.0)  # um
L_J_GALLERY = 12e-9  # a typical transmon junction, for the gallery numbers
SWEEP = (0.90, 0.95, 1.0, 1.05, 1.10)  # L_J over the harmonic crossing estimate
# Element orders. The device eigenmodes stop at 2: order 3 does not fit in 16 GB.
ORDER = {"plates": 2, "cpw": (2, 3), "device_electrostatic": 3, "device_eigenmode": 2}


def palace(np: int, folder: Path, config: dict[str, Any]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.json").write_text(json.dumps(config, indent=2))
    exe = os.environ.get("PALACE", "palace")
    with (folder / "palace.log").open("w") as f:
        subprocess.run(
            [exe, "-np", str(np), "config.json"],
            cwd=folder,
            stdout=f,
            stderr=subprocess.STDOUT,
            check=True,
        )


def read_csv(path: Path) -> list[list[float]]:
    with path.open() as f:
        rows = list(csv.reader(f))
    return [[float(x) for x in r] for r in rows[1:] if r]


def link_mesh(run: Path, mesh: Path) -> None:
    run.mkdir(parents=True, exist_ok=True)
    if not (run / "mesh.msh").exists():
        os.symlink(os.path.relpath(mesh, run), run / "mesh.msh")


def meshed(folder: Path, make: Any) -> dict[str, Any]:
    """Mesh once into folder/mesh.msh; the returned info is kept in folder/mesh.json."""
    folder.mkdir(parents=True, exist_ok=True)
    if not (folder / "mesh.json").exists():
        (folder / "mesh.json").write_text(json.dumps(make(folder / "mesh.msh")))
    info: dict[str, Any] = json.loads((folder / "mesh.json").read_text())
    return info


def silicon(a: dict[str, int]) -> list[dict[str, Any]]:
    return [
        {"Attributes": [a["substrate"]], "Permittivity": EPS_SI},
        {"Attributes": [a["vacuum"]], "Permittivity": 1.0},
    ]


def electrostatic(
    np: int,
    run: Path,
    materials: list[dict[str, Any]],
    ground: list[int],
    terminals: list[int],
    order: int,
) -> list[list[float]]:
    """The Maxwell capacitance matrix in farads (Palace's terminal-C.csv)."""
    if not (run / "postpro" / "terminal-C.csv").exists():
        palace(
            np,
            run,
            {
                "Problem": {"Type": "Electrostatic", "Verbose": 2, "Output": "postpro"},
                "Model": {"Mesh": "mesh.msh", "L0": 1e-6},
                "Domains": {"Materials": materials},
                "Boundaries": {
                    "Ground": {"Attributes": ground},
                    "Terminal": [
                        {"Index": i + 1, "Attributes": [t]} for i, t in enumerate(terminals)
                    ],
                },
                "Solver": {
                    "Order": order,
                    "Electrostatic": {"Save": 0},
                    "Linear": {"Type": "BoomerAMG", "KSPType": "CG", "Tol": 1e-10, "MaxIts": 1000},
                },
            },
        )
    return [r[1:] for r in read_csv(run / "postpro" / "terminal-C.csv")]


def plates(np: int, work: Path) -> list[dict[str, Any]]:
    out = []
    for side, gap in PLATES:
        folder = work / f"plates_{side:g}_{gap:g}"
        info = meshed(
            folder, lambda p, s=side, g=gap: geometry.parallel_plate(p, s, g, thickness=2.0)
        )
        a = info["attributes"]
        c = electrostatic(
            np,
            folder,
            [{"Attributes": [a["air"]], "Permittivity": 1.0}],
            [a["outer"]],
            [a["top"], a["bottom"]],
            ORDER["plates"],
        )
        out.append(
            {
                "side_um": side,
                "gap_um": gap,
                "thickness_um": 2.0,
                "maxwell_F": c,
                "tetrahedra": info["tetrahedra"],
            }
        )
    return out


def cpw(np: int, work: Path) -> dict[str, Any]:
    """Self-capacitance of the half-model strip, doubled to the full line, per length and order."""
    by_order: dict[str, list[float]] = {}
    tets = []
    for length in CPW_LENGTHS:
        folder = work / f"cpw_{length:g}"
        info = meshed(folder, lambda p, n=length: geometry.cpw(p, n))
        tets.append(info["tetrahedra"])
        a = info["attributes"]
        for order in ORDER["cpw"]:
            run = folder / f"order_{order}"
            link_mesh(run, folder / "mesh.msh")
            c = electrostatic(np, run, silicon(a), [a["walls"], a["ground"]], [a["strip"]], order)
            by_order.setdefault(str(order), []).append(2 * c[0][0])
    return {
        "lengths_um": list(CPW_LENGTHS),
        "C_F": by_order,
        "fine_um": info["fine"],
        "tetrahedra": tets,
    }


def device_modes(
    np: int, folder: Path, a: dict[str, int], L_J: float, target_ghz: float
) -> list[dict[str, float]]:
    run = folder / f"eigen_{L_J * 1e9:.4f}nH"
    link_mesh(run, folder / "mesh.msh")
    if not (run / "postpro" / "eig.csv").exists():
        palace(
            np,
            run,
            {
                "Problem": {"Type": "Eigenmode", "Verbose": 2, "Output": "postpro"},
                "Model": {"Mesh": "mesh.msh", "L0": 1e-6},
                "Domains": {"Materials": silicon(a)},
                "Boundaries": {
                    "PEC": {"Attributes": [a["walls"], a["ground"], a["pad"], a["centre"]]},
                    # Half of the junction, carrying half the current: 2 L_J (see geometry.py).
                    "LumpedPort": [
                        {"Index": 1, "Attributes": [a["junction"]], "Direction": "+X", "L": 2 * L_J}
                    ],
                },
                "Solver": {
                    "Order": ORDER["device_eigenmode"],
                    "Eigenmode": {"Target": target_ghz, "N": 3, "Save": 0, "Tol": 1e-8},
                    # The default here is a sparse direct solve, which needs more memory than 16 GB.
                    # AMS is the auxiliary-space multigrid for the curl-curl operator.
                    "Linear": {"Type": "AMS", "KSPType": "GMRES", "Tol": 1e-8, "MaxIts": 1000},
                },
            },
        )
    eig = read_csv(run / "postpro" / "eig.csv")
    epr = {int(r[0]): r[1] for r in read_csv(run / "postpro" / "port-EPR.csv")}
    # Palace signs the participation by the port's orientation; the magnitude is the fraction.
    return [{"f_GHz": r[1], "Q": r[3], "epr_junction": abs(epr[int(r[0])])} for r in eig]


def palace_version() -> str:
    exe = os.environ.get("PALACE", "palace")
    out = subprocess.run([exe, "--version"], capture_output=True, text=True).stdout
    lines = [
        ln.split(":", 1)[1].strip() for ln in out.splitlines() if ln.startswith("Palace version:")
    ]
    return lines[0] if lines else "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--np", type=int, default=4)
    ap.add_argument("--work", type=Path, default=ROOT / "runs" / "palace")
    args = ap.parse_args()
    work: Path = args.work
    np = args.np

    plate_results = plates(np, work)
    line = cpw(np, work)
    folder = work / "device"
    info = meshed(folder, geometry.device)
    a = info["attributes"]
    run = folder / "electrostatic"
    link_mesh(run, folder / "mesh.msh")
    half = electrostatic(
        np,
        run,
        silicon(a),
        [a["walls"], a["ground"]],
        [a["pad"], a["centre"]],
        ORDER["device_electrostatic"],
    )
    maxwell = [[2 * x for x in row] for row in half]  # the half model holds half the charge
    far = device_modes(np, folder, a, L_J_GALLERY, target_ghz=2.0)
    # The qubit holds nearly all its energy in the junction; the resonator's fundamental is the
    # lowest of the other modes.
    qubit = max(far, key=lambda m: m["epr_junction"])
    f_res = min(m["f_GHz"] for m in far if m is not qubit) * 1e9
    # Sweep L_J around where the harmonic qubit 1/(2 pi sqrt(L_J C_Sigma)) meets the resonator.
    L_star = 1 / ((2 * math.pi * f_res) ** 2 * maxwell[0][0])
    sweep = [
        {
            "L_J_H": L_star * k,
            "modes": device_modes(np, folder, a, L_star * k, target_ghz=0.8 * f_res / 1e9),
        }
        for k in SWEEP
    ]
    data = {
        "palace": {
            "version": palace_version(),
            "order": ORDER,
            "machine": f"{platform.system()} {platform.machine()}",
            "date": dt.date.today().isoformat(),
            "script": "scripts/palace/run.py",
        },
        "eps_r_substrate": EPS_SI,
        "plates": plate_results,
        "cpw": line,
        "device": {
            "geometry_um": info["device"],
            "tetrahedra": info["tetrahedra"],
            "maxwell_F": maxwell,
            "gallery": {"L_J_H": L_J_GALLERY, "modes": far},
            "sweep": sweep,
        },
    }
    OUT.write_text(json.dumps(data, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
