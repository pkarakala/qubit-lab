"""FastAPI backend for the explorer. Every response keeps theory and simulation apart."""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from catlab import __version__
from catlab.cache import disk_cached
from catlab.facts import FACTS
from catlab.provenance import bib_entries
from catlab.sim import cat as cat_sim
from catlab.sim import transmon as tr_sim
from catlab.theory import FORMULAS
from catlab.theory import cat as cat_th
from catlab.theory import transmon as tr_th

app = FastAPI(title="catlab", version=__version__)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class CatRequest(BaseModel):
    alpha_sq: float = Field(4.0, ge=0.0, le=9.0, description="target |alpha|^2")
    kappa_1: float = Field(0.0, ge=0.0, le=1.0, description="single-photon loss / kappa_2")
    n_th: float = Field(0.0, ge=0.0, le=0.5)
    kappa_phi: float = Field(0.0, ge=0.0, le=1.0, description="dephasing / kappa_2")
    scheme: Literal["dissipative", "kerr", "none"] = "dissipative"

    def params(self) -> cat_sim.CatParams:
        return cat_sim.CatParams(
            alpha_sq=self.alpha_sq,
            kappa_1=self.kappa_1,
            n_th=self.n_th,
            kappa_phi=self.kappa_phi,
            scheme=self.scheme,
        )


class EvolveRequest(CatRequest):
    initial: Literal["vacuum", "coherent_plus", "coherent_minus", "even_cat", "odd_cat"] = "vacuum"
    t_final: float = Field(3.0, gt=0.0, le=200.0)
    n_frames: int = Field(31, ge=2, le=81)
    grid: int = Field(61, ge=21, le=101)


class RatesRequest(CatRequest):
    alpha_sq_values: list[float] = Field(
        default_factory=lambda: [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        max_length=20,
    )


def _round(a: Any, digits: int = 5) -> Any:
    return np.round(np.asarray(a, dtype=np.float64), digits).tolist()


@disk_cached
def _evolve(body: dict[str, Any]) -> dict[str, Any]:
    req = EvolveRequest(**body)
    p = req.params()
    ev = cat_sim.evolve(p, req.initial, req.t_final, req.n_frames, req.grid)
    alpha = math.sqrt(req.alpha_sq)
    X, P = np.meshgrid(ev.xvec, ev.xvec, indexing="ij")
    # Theory overlay: the ideal even and odd cats of the same size, from closed forms.
    even = np.asarray(cat_th.wigner_even_cat(x=X, p=P, alpha=alpha)) if alpha > 0 else None
    odd = np.asarray(cat_th.wigner_odd_cat(x=X, p=P, alpha=alpha)) if alpha > 0 else None
    theory: dict[str, Any] = {
        "even_cat_wigner": _round(even, 4) if even is not None else None,
        "odd_cat_wigner": _round(odd, 4) if odd is not None else None,
    }
    if req.scheme == "dissipative" and req.kappa_1 > 0:
        theory["stabilized_photon_number"] = float(
            cat_th.stabilized_photon_number(alpha=alpha, kappa_1=req.kappa_1, kappa_2=1.0)
        )
        theory["parity_decay_rate"] = float(
            cat_th.parity_decay_rate(kappa_1=req.kappa_1, alpha=alpha)
        )
        if req.initial in ("even_cat", "odd_cat"):
            sign = 1.0 if req.initial == "even_cat" else -1.0
            rate = theory["parity_decay_rate"]
            theory["parity_curve"] = _round(sign * np.exp(-rate * ev.times))
    return {
        "sim": {
            "n_fock": p.n_fock,
            "times": _round(ev.times),
            "photon_number": _round(ev.photon_number),
            "parity": _round(ev.parity),
            "sign_x": _round(ev.sign_x),
            "purity": _round(ev.purity),
            "xvec": _round(ev.xvec),
            "wigner": _round(ev.wigner, 4),
        },
        "theory": theory,
    }


@disk_cached
def _rates(body: dict[str, Any]) -> dict[str, Any]:
    req = RatesRequest(**body)
    base = req.params()
    rows = []
    for n in sorted(set(req.alpha_sq_values)):
        if n <= 0:
            continue
        p = cat_sim.CatParams(**{**base.__dict__, "alpha_sq": n})
        r = cat_sim.liouvillian_rates(p)
        alpha = math.sqrt(n)
        theory: dict[str, float | None] = {"phase_flip_rate": None, "bit_flip_rate_dephasing": None}
        if req.scheme == "dissipative" and req.n_th == 0:
            if req.kappa_1 > 0:
                theory["phase_flip_rate"] = float(
                    cat_th.phase_flip_rate(kappa_1=req.kappa_1, alpha=alpha)
                )
            if req.kappa_phi > 0 and req.kappa_1 == 0:
                theory["bit_flip_rate_dephasing"] = float(
                    cat_th.bit_flip_rate_dephasing(kappa_phi=req.kappa_phi, alpha=alpha)
                )
        rows.append(
            {
                "alpha_sq": n,
                "sim": {
                    "bit_flip_rate": r.bit_flip_rate if r.z_decay_resolved else None,
                    "phase_flip_rate": r.phase_flip_rate,
                    "n_fock": r.n_fock,
                },
                "theory": theory,
            }
        )
    return {
        "rows": rows,
        "theory": {
            "bit_flip_suppression_per_photon": float(cat_th.bit_flip_suppression(Delta_n=1.0))
        },
        "rate_floor": cat_sim.RATE_FLOOR / 2,
    }


@disk_cached
def _checks() -> list[dict[str, Any]]:
    from catlab.checks import run_all

    return [r.describe() for r in run_all()]


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.get("/api/formulas")
def formulas() -> list[dict[str, Any]]:
    return [f.describe() for f in FORMULAS.values()]


@app.get("/api/references")
def references() -> dict[str, dict[str, str]]:
    return bib_entries()


@app.get("/api/facts")
def facts() -> list[dict[str, object]]:
    return [f.describe() for f in FACTS]


@app.post("/api/cat/evolve")
def cat_evolve(req: EvolveRequest) -> dict[str, Any]:
    try:
        return _evolve(req.model_dump())
    except ValueError as e:
        raise HTTPException(422, str(e)) from e


@app.post("/api/cat/rates")
def cat_rates(req: RatesRequest) -> dict[str, Any]:
    if max(req.alpha_sq_values, default=0) > 8:
        raise HTTPException(422, "Liouvillian rates are only resolved up to |alpha|^2 = 8")
    return _rates(req.model_dump())


@app.get("/api/transmon/bands")
def transmon_bands(
    E_J: float = 15.0, E_C: float = 0.3, n_levels: int = 4, n_points: int = 101
) -> dict[str, Any]:
    if not (0 < E_J <= 200 and 0 < E_C <= 10 and 3 <= n_levels <= 6 and 11 <= n_points <= 401):
        raise HTTPException(422, "parameters out of range")
    p = tr_sim.TransmonParams(E_J=E_J, E_C=E_C)
    b = tr_sim.bands(p, n_points=n_points, n_levels=n_levels)
    half = tr_sim.levels(p, 0.5, n_levels)
    zero = tr_sim.levels(p, 0.0, n_levels)
    duffing = [
        float(
            tr_th.level_energy(m=float(m), E_J=E_J, E_C=E_C)
            - tr_th.level_energy(m=0.0, E_J=E_J, E_C=E_C)
        )
        for m in range(n_levels)
    ]
    return {
        "sim": {
            "n_g": _round(b.n_g),
            "energies": _round(b.energies, 6),
            "dispersion": [float(half[m] - zero[m]) for m in range(n_levels)],
            "e01_mean": float(np.mean(b.transition(0, 1))),
            "anharmonicity": float(np.mean(b.transition(1, 2) - b.transition(0, 1))),
        },
        "theory": {
            "level_spacing_from_ground": duffing,
            "e01": float(tr_th.qubit_frequency(E_J=E_J, E_C=E_C)),
            "anharmonicity": float(tr_th.anharmonicity(E_C=E_C)),
            "dispersion": [
                float(tr_th.charge_dispersion(m=float(m), E_J=E_J, E_C=E_C))
                for m in range(n_levels)
            ],
        },
    }


@app.get("/api/checks")
def checks() -> list[dict[str, Any]]:
    return _checks()


@disk_cached
def _transmon_surface(
    E_C: float, ratios: list[float], n_levels: int, n_points: int
) -> dict[str, Any]:
    energies, ripples = [], []
    for r in ratios:
        b = tr_sim.bands(tr_sim.TransmonParams(E_J=r * E_C, E_C=E_C), n_points, n_levels)
        e01 = float(np.mean(b.transition(0, 1)))
        energies.append(b.energies / e01)  # in units of the mean E_01
        ripples.append(
            (b.energies - b.energies.mean(axis=0)) / E_C
        )  # wiggle with n_g, units of E_C
    ng = np.linspace(-1.0, 1.0, n_points)
    return {
        "sim": {
            "ratios": ratios,
            "n_g": _round(ng),
            "energies_over_e01": _round(energies, 6),
            "ripple_over_ec": _round(ripples, 6),
        }
    }


@app.get("/api/transmon/surface")
def transmon_surface(E_C: float = 1.0, n_levels: int = 3, n_points: int = 61) -> dict[str, Any]:
    """Bands over (n_g, E_J/E_C) on a log grid of E_J/E_C from 1 to 100, scaled by E_01."""
    ratios = [float(r) for r in np.round(np.logspace(0, 2, 41), 4)]
    return _transmon_surface(E_C, ratios, n_levels, n_points)
