"""Palace results for the chip layer: capacitances and eigenmodes of a transmon and its resonator.

Palace (awslabs/palace) is a finite-element solver that runs offline, so the explorer never
needs it. `scripts/palace/run.py` meshes the geometry with Gmsh, runs the solves and writes
data/chip.json, which this module reads. SI units: farads, henries, hertz, metres.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np

DATA = Path(__file__).resolve().parents[1] / "data" / "chip.json"


@cache
def load() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(DATA.read_text(encoding="utf-8"))
    return data


def plate_capacitance(side_um: float) -> float:
    """Mutual capacitance between the two plates of the case with this side, from the
    Maxwell matrix: C_12 is negative, the capacitance between the plates is -C_12."""
    for case in load()["plates"]:
        if case["side_um"] == side_um:
            return float(-case["maxwell_F"][0][1])
    raise KeyError(f"no parallel-plate run with side {side_um} um")


def plates() -> list[dict[str, Any]]:
    return list(load()["plates"])


def cpw_capacitance_per_length(order: int = 3) -> float:
    """C per length of the straight CPW: the difference of two lengths over their difference,
    so the end fields cancel. F/m."""
    line = load()["cpw"]
    (l1, l2), (c1, c2) = line["lengths_um"], line["C_F"][str(order)]
    return float((c2 - c1) / ((l2 - l1) * 1e-6))


def cpw_orders() -> list[int]:
    return sorted(int(k) for k in load()["cpw"]["C_F"])


@dataclass(frozen=True)
class Capacitances:
    """From the Maxwell matrix of terminals 1 = qubit pad, 2 = resonator centre conductor."""

    C_sigma: float  # everything the pad sees: C_11
    C_g: float  # pad to resonator: -C_12
    C_line: float  # the whole centre conductor: C_22


def capacitances() -> Capacitances:
    m = load()["device"]["maxwell_F"]
    return Capacitances(C_sigma=float(m[0][0]), C_g=float(-m[0][1]), C_line=float(m[1][1]))


def geometry() -> dict[str, float]:
    """Device sizes in micrometres."""
    return dict(load()["device"]["geometry_um"])


def eps_r() -> float:
    return float(load()["eps_r_substrate"])


def _split(modes: list[dict[str, float]]) -> tuple[dict[str, float], dict[str, float]]:
    """(qubit-like, resonator fundamental): the junction holds most of the qubit mode's energy,
    and the fundamental is the lowest of the others."""
    qubit = max(modes, key=lambda m: m["epr_junction"])
    return qubit, min((m for m in modes if m is not qubit), key=lambda m: m["f_GHz"])


def gallery_modes() -> tuple[float, float, float]:
    """(L_J, qubit mode frequency, resonator mode frequency) at the gallery junction, far from
    resonance."""
    g = load()["device"]["gallery"]
    qubit, res = _split(g["modes"])
    return float(g["L_J_H"]), qubit["f_GHz"] * 1e9, res["f_GHz"] * 1e9


@dataclass(frozen=True)
class Crossing:
    f_r: float  # bare resonator frequency from the fit
    g: float  # coupling: half the smallest splitting
    L_J: np.ndarray  # the sweep
    f_lower: np.ndarray
    f_upper: np.ndarray


def crossing() -> Crossing:
    """Fit the avoided crossing of the qubit and resonator modes over the L_J sweep.

    Two coupled modes have f_+ + f_- = f_q + f_r and (f_+ - f_-)^2 = (f_q - f_r)^2 + 4 g^2.
    Eliminating f_q: (f_+ - f_-)^2 - (f_+ + f_-)^2 = -4 f_r (f_+ + f_-) + 4 (f_r^2 + g^2),
    a straight line in S = f_+ + f_-, fit by least squares (numpy.polyfit).
    """
    L, lo, hi = [], [], []
    for point in load()["device"]["sweep"]:
        two = sorted(point["modes"], key=lambda m: -m["epr_junction"])[:2]
        f = sorted(m["f_GHz"] * 1e9 for m in two)
        L.append(point["L_J_H"])
        lo.append(f[0])
        hi.append(f[1])
    f_lo, f_hi = np.array(lo), np.array(hi)
    S, D = f_hi + f_lo, f_hi - f_lo
    slope, intercept = np.polyfit(S, D**2 - S**2, 1)
    f_r = -slope / 4
    g = float(np.sqrt(intercept / 4 - f_r**2))
    return Crossing(float(f_r), g, np.array(L), f_lo, f_hi)


def provenance() -> dict[str, Any]:
    return dict(load()["palace"])
