"""FastAPI backend for the explorer. Every response keeps theory and simulation apart."""

from __future__ import annotations

import json
import math
import os
import threading
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from typing import Any, Literal

import anyio
import jax
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, ValidationError
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from qubitlab import __version__, snippets
from qubitlab.architectures import scenes as architecture_scenes
from qubitlab.cache import CACHE_DIR, disk_cached
from qubitlab.facts import DEVICES, FACT, FACTS
from qubitlab.provenance import bib_entries
from qubitlab.sim import cat as cat_sim
from qubitlab.sim import custom as custom_sim
from qubitlab.sim import pulse_optimizer as po_sim
from qubitlab.sim import qec as qec_sim
from qubitlab.sim import readout as ro_sim
from qubitlab.sim import spectra as spectra_sim
from qubitlab.sim import transmon as tr_sim
from qubitlab.sim import transmon_dynamics as qb_sim
from qubitlab.sim import two_qubit as tq_sim
from qubitlab.theory import FORMULAS
from qubitlab.theory import cat as cat_th
from qubitlab.theory import gates as gate_th
from qubitlab.theory import qec as qec_th
from qubitlab.theory import qubit as qb_th
from qubitlab.theory import readout as ro_th
from qubitlab.theory import transmon as tr_th

# Public hosting knobs, both off by default (see docs/deploy.md):
# QUBITLAB_SOLVES   most simulations running at once; the rest wait their turn.
# QUBITLAB_RATE     requests per client, e.g. "30/minute".
# QUBITLAB_WARM     "1": precompute the default views in the background on start (qubitlab.warm).
SOLVES = int(os.environ.get("QUBITLAB_SOLVES", "0"))
RATE = os.environ.get("QUBITLAB_RATE")
WARM = os.environ.get("QUBITLAB_WARM") == "1"

# JAX keeps compiled solvers on disk, so a restarted server skips most of the first-run compile.
jax.config.update("jax_compilation_cache_dir", str(CACHE_DIR / "jax"))  # type: ignore[no-untyped-call]
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.1)  # type: ignore[no-untyped-call]


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    if SOLVES:
        # Sync endpoints run in AnyIO's thread pool, so its size caps concurrent solves.
        anyio.to_thread.current_default_thread_limiter().total_tokens = SOLVES
    if WARM:
        from qubitlab.warm import warm

        threading.Thread(target=warm, name="qubitlab-warm", daemon=True).start()
    yield


app = FastAPI(title="qubitlab", version=__version__, lifespan=lifespan)


@app.middleware("http")
async def warm_steps_aside(request: Request, call_next: Any) -> Any:
    """Hold the background warm-up between its steps while a request is being answered."""
    from qubitlab.warm import request_finished, request_started

    request_started()
    try:
        return await call_next(request)
    finally:
        request_finished()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def add_rate_limit(target: FastAPI) -> None:
    """Limit requests per visitor when QUBITLAB_RATE is set (also used by `qubitlab.serve`)."""
    if RATE:
        target.state.limiter = Limiter(key_func=get_remote_address, default_limits=[RATE])
        target.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]
        target.add_middleware(SlowAPIMiddleware)


add_rate_limit(app)


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
    # The Liouvillian block method reads the flip rates off the slowest modes. That is only
    # right when the two cats are the slowest manifold, i.e. under two-photon dissipation. With
    # a Kerr cat or no stabilization, faster-looking intra-well modes would be mislabeled.
    scheme: Literal["dissipative"] = "dissipative"
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
def _rate_row(body: dict[str, Any]) -> dict[str, Any]:
    """Liouvillian flip rates at one cat size (body's alpha_sq), with the closed forms there."""
    req = RatesRequest(**body)
    n = req.alpha_sq
    p = req.params()
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
    return {
        "alpha_sq": n,
        "sim": {
            "bit_flip_rate": r.bit_flip_rate if r.z_decay_resolved else None,
            "phase_flip_rate": r.phase_flip_rate if r.x_decay_resolved else None,
            "n_fock": r.n_fock,
        },
        "theory": theory,
    }


def _rate_bodies(req: RatesRequest) -> list[dict[str, Any]]:
    """One request per cat size, so each row is cached on its own."""
    sizes = [n for n in sorted(set(req.alpha_sq_values)) if n > 0]
    return [req.model_dump() | {"alpha_sq": n, "alpha_sq_values": [n]} for n in sizes]


def _rate_header() -> dict[str, Any]:
    return {
        "theory": {
            "bit_flip_suppression_per_photon": float(cat_th.bit_flip_suppression(Delta_n=1.0))
        },
        "rate_floor": cat_sim.RATE_FLOOR / 2,
    }


def _rates(body: dict[str, Any]) -> dict[str, Any]:
    rows = [_rate_row(b) for b in _rate_bodies(RatesRequest(**body))]
    return {"rows": rows, **_rate_header()}


@disk_cached
def _checks() -> list[dict[str, Any]]:
    from qubitlab.checks import run_all

    return [r.describe() for r in run_all()]


def _checks_fast() -> list[dict[str, Any]]:
    """The shipped results when they match this code (qubitlab.snapshot), else a live run."""
    from qubitlab import snapshot

    return snapshot.load() or _checks()


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


@app.post("/api/cat/rates/stream")
def cat_rates_stream(req: RatesRequest) -> StreamingResponse:
    """The same rates as /api/cat/rates, one JSON line per cat size as each is solved.

    The first line is the header (theory constants and the number of rows to come), so the
    explorer can draw the curve point by point instead of waiting for the slowest size.
    """
    if max(req.alpha_sq_values, default=0) > 8:
        raise HTTPException(422, "Liouvillian rates are only resolved up to |alpha|^2 = 8")
    bodies = _rate_bodies(req)

    def lines() -> Iterator[str]:
        yield json.dumps({**_rate_header(), "total": len(bodies)}) + "\n"
        for b in bodies:
            yield json.dumps({"row": _rate_row(b)}) + "\n"

    return StreamingResponse(lines(), media_type="application/x-ndjson")


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


@app.get("/api/presets")
def presets() -> list[dict[str, Any]]:
    """Settings from published experiments, each citing its paper. The cache warms them."""
    from qubitlab.warm import presets as load

    return load()


@app.get("/api/checks")
def checks() -> list[dict[str, Any]]:
    return _checks_fast()


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


class QubitRequest(BaseModel):
    experiment: Literal["rabi", "pulse", "t1", "ramsey", "echo"] = "rabi"
    alpha_ghz: float = Field(-0.3, ge=-1.0, le=-0.05, description="anharmonicity")
    t1_ns: float = Field(20_000.0, ge=1.0, le=1e6)
    t2_ns: float = Field(20_000.0, ge=1.0, le=2e6)
    sigma_ghz: float = Field(0.0, ge=0.0, le=0.02, description="quasi-static noise spread")
    delta_ghz: float = Field(0.0, ge=-0.1, le=0.1, description="qubit minus drive frequency")
    omega_ghz: float = Field(0.02, ge=0.0, le=0.2, description="Rabi frequency")
    duration_ns: float = Field(20.0, ge=2.0, le=200.0, description="pi pulse length")
    drag: float = Field(0.0, ge=-2.0, le=3.0)
    t_max_ns: float = Field(200.0, gt=0.0, le=1e6)
    n_points: int = Field(121, ge=11, le=301)


@disk_cached
def _qubit(body: dict[str, Any]) -> dict[str, Any]:
    req = QubitRequest(**body)
    if req.t2_ns > 2 * req.t1_ns:
        raise ValueError("T2 cannot exceed 2 T1")
    q = qb_sim.TransmonQubit(
        alpha_ghz=req.alpha_ghz, t1_ns=req.t1_ns, t2_ns=req.t2_ns, sigma_ghz=req.sigma_ghz
    )
    n_points = min(req.n_points, 41) if req.experiment == "echo" else req.n_points
    traj = qb_sim.run(
        req.experiment,
        q,
        delta_ghz=req.delta_ghz,
        omega_ghz=req.omega_ghz,
        duration_ns=req.duration_ns,
        drag=req.drag,
        t_max_ns=req.t_max_ns,
        n_points=n_points,
    )
    t = traj.times_ns
    curve: Any = None
    label = ""
    if req.experiment == "rabi":
        curve = qb_th.rabi_population(Omega=req.omega_ghz, Delta=req.delta_ghz, t=t)
        label = "P1, two-level Rabi formula (no decay)"
    elif req.experiment == "t1":
        curve = qb_th.t1_decay(t=t, T_1=req.t1_ns)
        label = "P1 = exp(-t/T1)"
    elif req.experiment == "ramsey":
        curve = qb_th.ramsey(t=t, T_2=req.t2_ns, sigma=req.sigma_ghz, Delta=req.delta_ghz)
        label = "<X>, Ramsey formula"
    elif req.experiment == "echo":
        curve = qb_th.echo(t=t, T_2=req.t2_ns)
        label = "<X> = exp(-t/T2)"
    return {
        "sim": {
            "times": _round(t),
            "bloch": _round(traj.bloch),
            "populations": _round(traj.populations),
            "leakage": _round(traj.leakage),
            "omega_x": _round(traj.omega_x),
            "omega_y": _round(traj.omega_y),
        },
        "theory": {
            "curve": _round(curve) if curve is not None else None,
            "curve_label": label,
            "idle_fidelity_at_end": float(
                qb_th.idle_fidelity(t=float(t[-1]), T_1=req.t1_ns, T_2=req.t2_ns)
            ),
        },
    }


@app.post("/api/qubit/run")
def qubit_run(req: QubitRequest) -> dict[str, Any]:
    """One lab experiment on a transmon kept to 4 levels: Bloch path, populations, theory."""
    try:
        return _qubit(req.model_dump())
    except ValueError as e:
        raise HTTPException(422, str(e)) from e


class PulseRequest(BaseModel):
    alpha_ghz: float = Field(-0.2, ge=-1.0, le=-0.05, description="anharmonicity")
    duration_ns: float = Field(6.0, ge=2.0, le=40.0, description="X gate length")
    omega_max_ghz: float = Field(0.3, ge=0.05, le=1.0, description="largest drive allowed")
    t1_ns: float = Field(50_000.0, ge=1.0, le=1e6)
    t2_ns: float = Field(50_000.0, ge=1.0, le=2e6)
    steps: int = Field(300, ge=10, le=1000)


@disk_cached
def _pulse(body: dict[str, Any]) -> dict[str, Any]:
    req = PulseRequest(**body)
    if req.t2_ns > 2 * req.t1_ns:
        raise ValueError("T2 cannot exceed 2 T1")
    q = qb_sim.TransmonQubit(alpha_ghz=req.alpha_ghz, t1_ns=req.t1_ns, t2_ns=req.t2_ns)
    slices = round(4 * req.duration_ns)  # 0.25 ns per slice, a 4 GS/s waveform generator
    edges, dx, dy = po_sim.drag_slices(q, req.duration_ns, slices)
    opt = po_sim.optimize(
        q, req.duration_ns, slices=slices, omega_max_ghz=req.omega_max_ghz, steps=req.steps
    )

    def score(ox: Any, oy: Any, error: float, leakage: float) -> dict[str, Any]:
        H = qb_sim.driven_h(q, 0.0, edges, ox, oy)
        noisy = 1 - qb_sim.average_gate_fidelity(q, H, req.duration_ns, po_sim.X_GATE)
        traj = qb_sim.run_pulse(q, edges, ox, oy)
        return {
            "omega_x": _round(ox),
            "omega_y": _round(oy),
            "error": error,
            "leakage": leakage,
            "error_with_noise": noisy,
            "times": _round(traj.times_ns),
            "bloch": _round(traj.bloch),
            "populations": _round(traj.populations),
        }

    return {
        "sim": {
            "edges": _round(edges),
            "drag": score(dx, dy, *po_sim.gate_error(q, edges, dx, dy)),
            "optimized": {
                **score(opt.omega_x, opt.omega_y, opt.error, opt.leakage),
                "history": opt.history.tolist(),
            },
        }
    }


@app.post("/api/pulse/optimize")
def pulse_optimize(req: PulseRequest) -> dict[str, Any]:
    """An X gate on a 4-level transmon: the DRAG pulse vs a pulse found by gradient descent."""
    try:
        return _pulse(req.model_dump())
    except ValueError as e:
        raise HTTPException(422, str(e)) from e


MIN_FAILURES = 10
P_GRID = [0.001, 0.0015, 0.002, 0.003, 0.005, 0.007, 0.01, 0.015]


@disk_cached
def _qec_curves(shots: int) -> dict[str, Any]:
    curves = []
    for code in ("surface", "repetition"):
        for d in (3, 5, 7):
            runs = [qec_sim.memory(code, d, p, shots) for p in P_GRID]
            curves.append(
                {
                    "code": code,
                    "d": d,
                    "qubits": qec_sim.physical_qubits(code, d),
                    # Under 10 failures the estimate is mostly sampling noise: not reported.
                    "per_round": [
                        r.per_round if r.failures >= MIN_FAILURES else None for r in runs
                    ],
                    "failures": [r.failures for r in runs],
                }
            )
    fowler = {
        str(d): [
            float(qec_th.surface_code_logical_error(p=p, p_th=0.0057, d=float(d))) for p in P_GRID
        ]
        for d in (3, 5, 7)
    }
    return {
        "p": P_GRID,
        "shots": shots,
        "sim": {"curves": curves},
        "theory": {
            "fowler_surface": fowler,
            "fowler_p_th": 0.0057,
            "surface_qubits": {
                str(d): float(qec_th.surface_code_qubits(d=float(d))) for d in (3, 5, 7)
            },
        },
    }


@app.get("/api/qec/curves")
def qec_curves() -> dict[str, Any]:
    """Logical error per round vs physical error, Stim + PyMatching, d = 3, 5, 7."""
    return _qec_curves(100_000)


class CatCodeRequest(BaseModel):
    p_phase: float = Field(0.005, gt=0.0, le=0.05, description="cat phase-flip prob. per round")
    p_bit: float = Field(1e-6, ge=0.0, le=0.01, description="cat bit-flip prob. per round")
    shots: int = Field(50_000, ge=1000, le=500_000)


@disk_cached
def _cat_code(body: dict[str, Any]) -> dict[str, Any]:
    req = CatCodeRequest(**body)
    rows = []
    for d in (1, 3, 5, 7, 9, 11, 13):
        if d == 1:
            corrected: float | None = req.p_phase
            failures = None
        else:
            m = qec_sim.memory("repetition", d, req.p_phase, req.shots)
            corrected = m.per_round if m.failures >= MIN_FAILURES else None
            failures = m.failures
        rows.append(
            {
                "d": d,
                "qubits": 2 * d - 1 if d > 1 else 1,
                "sim": {"phase_flip_logical": corrected, "failures": failures},
                "theory": {
                    "bit_flip_floor": float(
                        qec_th.repetition_code_bit_flip_floor(d=float(d), p_bit=req.p_bit)
                    )
                },
            }
        )
    return {"rows": rows, "shots": req.shots}


@app.post("/api/qec/cat")
def qec_cat(req: CatCodeRequest) -> dict[str, Any]:
    """A repetition code of cats: corrected phase-flips (sim) plus uncorrected bit-flips (theory).

    The repetition circuit applies p to every operation, so p_phase here plays the role of the
    cat's phase-flip probability per round; bit-flips are added from the closed form.
    """
    return _cat_code(req.model_dump())


@app.get("/api/qec/shot")
def qec_shot(d: int = 5, p: float = 0.005, seed: int = 0) -> dict[str, Any]:
    """One surface code run: detection events in space-time and the decoder's matching."""
    if d not in (3, 5, 7) or not 0 < p <= 0.05:
        raise HTTPException(422, "d must be 3, 5 or 7 and 0 < p <= 0.05")
    s = qec_sim.sample_shot(d, p, seed)
    return {
        "sim": {
            "qubits": s.qubits,
            "detectors": s.detectors,
            "fired": s.fired,
            "matched": s.matched,
            "logical_error": s.logical_error,
        }
    }


@disk_cached
def _compare(t_ns: float) -> dict[str, Any]:
    rows = []
    for d in DEVICES:
        t_bit = FACT[d.t_bit].value * 1e9  # facts are in seconds
        t_phase = FACT[d.t_phase].value * 1e9
        p_bit = float(qb_th.flip_probability(t=t_ns, T=t_bit))
        p_phase = float(qb_th.flip_probability(t=t_ns, T=t_phase))
        theory: dict[str, float | None] = {
            "p_bit": p_bit,
            "p_phase": p_phase,
            "bias": p_phase / p_bit,
            "idle_error": None,
        }
        sim: dict[str, float] | None = None
        if d.qubit == "transmon":
            theory["idle_error"] = 1 - float(qb_th.idle_fidelity(t=t_ns, T_1=t_bit, T_2=t_phase))
            q = qb_sim.TransmonQubit(t1_ns=t_bit, t2_ns=t_phase)
            s_bit, s_phase = qb_sim.flip_probabilities(q, t_ns)
            sim = {
                "p_bit": s_bit,
                "p_phase": s_phase,
                "idle_error": 1 - qb_sim.average_idle_fidelity(q, t_ns),
            }
        rows.append(
            {
                "key": d.key,
                "label": d.label,
                "qubit": d.qubit,
                "t_bit": d.t_bit,
                "t_phase": d.t_phase,
                "theory": theory,
                "sim": sim,
            }
        )
    return {"t_ns": t_ns, "devices": rows}


@app.get("/api/compare")
def compare(t_ns: float = 25.0) -> dict[str, Any]:
    """Bit- and phase-flip chance per idle step of length t for measured cats and transmons.

    The default t is the single-qubit gate time of Acharya et al. 2025
    (willow2025, Supplement Sec. I).
    """
    if not 1.0 <= t_ns <= 1e5:
        raise HTTPException(422, "t_ns must be between 1 ns and 100 us")
    return _compare(t_ns)


class JumpIn(BaseModel):
    op: str = Field(max_length=custom_sim.MAX_CHARS)
    rate: str = Field("1", max_length=custom_sim.MAX_CHARS)


class PlaygroundRequest(BaseModel):
    hamiltonian: str = Field("0", max_length=custom_sim.MAX_CHARS)
    jumps: list[JumpIn] = Field(default_factory=list, max_length=6)
    params: dict[str, float] = Field(default_factory=dict, max_length=12)
    dim: int = Field(30, ge=4, le=60, description="Fock states kept")
    initial: Literal["vacuum", "coherent_plus", "coherent_minus", "even_cat", "odd_cat"] = "vacuum"
    alpha_sq: float = Field(4.0, ge=0.0, le=12.0, description="|alpha|^2 of the initial state")
    t_final: float = Field(3.0, gt=0.0, le=100.0)
    n_frames: int = Field(31, ge=2, le=81)
    grid: int = Field(61, ge=21, le=101)
    extent: float = Field(4.0, ge=1.0, le=8.0, description="half-width of the Wigner window")

    def model(self) -> custom_sim.Model:
        jumps = tuple(custom_sim.Jump(j.op, j.rate) for j in self.jumps)
        return custom_sim.Model(self.hamiltonian, jumps, self.params, self.dim)


@disk_cached
def _playground(body: dict[str, Any]) -> dict[str, Any]:
    req = PlaygroundRequest(**body)
    model = req.model()
    ev = custom_sim.evolve(
        model, req.initial, req.alpha_sq, req.t_final, req.n_frames, req.grid, req.extent
    )
    script = custom_sim.python_script(model, req.initial, req.alpha_sq, req.t_final, req.n_frames)
    return {
        "sim": {
            "n_fock": req.dim,
            "times": _round(ev.times),
            "photon_number": _round(ev.photon_number),
            "parity": _round(ev.parity),
            "sign_x": _round(ev.sign_x),
            "purity": _round(ev.purity),
            "xvec": _round(ev.xvec),
            "wigner": _round(ev.wigner, 4),
            "edge_population": _round(ev.edge_population),
        },
        "python": script,
    }


@app.post("/api/playground")
def playground(req: PlaygroundRequest) -> dict[str, Any]:
    """Any one-mode Lindblad model the user types, solved with Dynamiqs. No theory layer."""
    try:
        return _playground(req.model_dump())
    except custom_sim.ModelError as e:
        raise HTTPException(422, str(e)) from e
    except RuntimeError as e:  # Dynamiqs hit its step limit
        msg = "the solver gave up: the model is too fast or too stiff for this duration"
        raise HTTPException(422, msg) from e


class TransmonInputs(BaseModel):
    E_J: float = Field(15.0, gt=0.0, le=200.0)
    E_C: float = Field(0.3, gt=0.0, le=10.0)


class QecInputs(BaseModel):
    p: float = Field(0.005, gt=0.0, le=0.05, description="physical error rate, surface code")
    p_phase: float = Field(0.005, gt=0.0, le=0.05)
    p_bit: float = Field(5e-7, ge=0.0, le=0.01)


class CompareInputs(BaseModel):
    t_ns: float = Field(25.0, ge=1.0, le=1e5)


class ReadoutRequest(BaseModel):
    ratio: float = Field(50.0, ge=20.0, le=200.0, description="E_J / E_C")
    E_C: float = Field(0.3, ge=0.1, le=0.5, description="charging energy, GHz")
    f_r: float = Field(7.0, ge=4.0, le=9.0, description="bare resonator frequency, GHz")
    g_n: float = Field(0.05, ge=0.005, le=0.15, description="coupling to n (a + a+), GHz")
    kappa_mhz: float = Field(1.0, ge=0.1, le=20.0, description="resonator linewidth / 2 pi")
    n_bar: float = Field(1.0, ge=0.05, le=10.0, description="photons for a resonant drive")
    delta_r_mhz: float = Field(0.0, ge=-20.0, le=20.0, description="omega_r - omega_d, / 2 pi")


@disk_cached
def _readout(body: dict[str, Any]) -> dict[str, Any]:
    req = ReadoutRequest(**body)
    d = ro_sim.dressed_chi(req.ratio * req.E_C, req.E_C, req.f_r, req.g_n)
    delta = d.f01 - req.f_r
    theory_chi = {
        "transmon": float(ro_th.chi_transmon(g=d.g01, g_12=d.g12, Delta=delta, alpha=d.alpha)),
        "harmonic": float(ro_th.chi_transmon_harmonic(g=d.g01, Delta=delta, alpha=d.alpha)),
        "two_level": float(ro_th.chi_two_level(g=d.g01, Delta=delta)),
    }
    # Dynamics in rad/us: chi and kappa from MHz, the drive set by the resonant photon number.
    two_pi = 2 * math.pi
    chi = two_pi * d.chi * 1e3
    kappa = two_pi * req.kappa_mhz
    delta_r = two_pi * req.delta_r_mhz
    eps = math.sqrt(req.n_bar) * kappa / 2
    t_max = 12 / kappa
    p = ro_sim.pointer_trajectories(chi, eps, kappa, delta_r, t_max, n_points=61)
    probe = np.linspace(-4, 4, 161) * max(abs(chi), kappa / 2)
    return {
        "sim": {
            "chi_mhz": d.chi * 1e3,
            "f01_ghz": d.f01,
            "alpha_mhz": d.alpha * 1e3,
            "g01_mhz": d.g01 * 1e3,
            "g12_mhz": d.g12 * 1e3,
            "times_us": _round(p.times),
            "field_g": [_round(p.field_g.real), _round(p.field_g.imag)],
            "field_e": [_round(p.field_e.real), _round(p.field_e.imag)],
            "photons_g": _round(p.photons_g),
            "photons_e": _round(p.photons_e),
            "dephasing_per_us": ro_sim.dephasing_rate(chi, eps, kappa),
        },
        "theory": {
            "chi_mhz": {k: v * 1e3 for k, v in theory_chi.items()},
            "probe_mhz": _round(probe / two_pi),
            "response_g": _round(
                ro_th.pointer_photons(epsilon=eps, kappa=kappa, delta_r=probe, chi=-chi)
            ),
            "response_e": _round(
                ro_th.pointer_photons(epsilon=eps, kappa=kappa, delta_r=probe, chi=chi)
            ),
            "dephasing_per_us": float(ro_th.dephasing_rate(kappa=kappa, chi=chi, epsilon=eps)),
            "dephasing_weak_per_us": float(
                ro_th.dephasing_rate_weak(kappa=kappa, chi=chi, epsilon=eps)
            ),
        },
    }


@app.post("/api/readout")
def readout(req: ReadoutRequest) -> dict[str, Any]:
    """Dispersive readout of a transmon: chi from the dressed spectrum, pointer states over time."""
    return _readout(req.model_dump())


class GatesRequest(BaseModel):
    g_mhz: float = Field(10.0, ge=1.0, le=60.0, description="transmon exchange coupling / 2 pi")
    alpha_mhz: float = Field(-300.0, ge=-400.0, le=-100.0, description="anharmonicity, both")
    n: Literal[2, 4] = Field(4, description="|alpha|^2 of each cat (even: no geometric phase)")
    log_kappa_ratio: float = Field(-3.0, ge=-4.0, le=-2.0, description="log10 kappa_1/kappa_2")


@disk_cached
def _cz(g_mhz: float, alpha_mhz: float) -> dict[str, Any]:
    two_pi = 2 * math.pi
    g, a = two_pi * g_mhz * 1e-3, two_pi * alpha_mhz * 1e-3  # rad/ns
    t_cz = float(gate_th.cz_time(g=g))
    times = np.linspace(0, 2 * t_cz, 81)
    phase, leak = tq_sim.cz_sweep(two_pi * 5.0, a, a, g, times)
    deltas = np.linspace(-0.6, 0.6, 121) * abs(alpha_mhz) * 1e-3 * 2  # GHz around 5 GHz
    # Leave out the resonances (|Delta| = 0 or |alpha|), where dressed states cannot be labeled.
    gap = 2 * g_mhz * 1e-3
    deltas = deltas[(np.abs(deltas) > gap) & (np.abs(np.abs(deltas) - abs(alpha_mhz) * 1e-3) > gap)]
    gz = g_mhz * 1e-3
    a_ghz = alpha_mhz * 1e-3
    return {
        "sim": {
            "times_ns": _round(times),
            "phase": _round(np.abs(np.unwrap(phase))),  # -pi and pi are the same CZ
            "leakage": _round(leak),
            "zz_delta_mhz": _round(deltas * 1e3),
            "zz_mhz": _round(
                [tq_sim.static_zz(5.0 + d, 5.0, a_ghz, a_ghz, gz) * 1e3 for d in deltas]
            ),
        },
        "theory": {
            "cz_time_ns": t_cz,
            "zz_mhz": _round(
                gate_th.static_zz(g=gz, Delta=deltas, alpha_1=a_ghz, alpha_2=a_ghz) * 1e3
            ),
        },
    }


@disk_cached
def _cnot(n: int, log_kappa_ratio: float) -> dict[str, Any]:
    k1 = 10**log_kappa_ratio
    T = np.logspace(-1, 1.5, 81)
    z1 = np.asarray(gate_th.cnot_z1(n=float(n), kappa_1=k1, kappa_2=1.0, T=T))
    z2 = np.asarray(gate_th.cnot_z2(n=float(n), kappa_1=k1, T=T))
    t_opt = 1 / (2 * n * math.sqrt(math.pi * k1))  # docs/hand_calcs/two_qubit_gates.py step 3
    sim_T = [0.5 * t_opt, t_opt, 3 * t_opt]
    runs = [tq_sim.cat_cnot(float(n), k1, t) for t in sim_T]
    return {
        "sim": {
            "T": sim_T,
            "control": [r.control_flip for r in runs],
            "target": [r.target_flip for r in runs],
        },
        "theory": {
            "T": _round(T),
            "z1": _round(z1, 7),
            "z2": _round(z2, 7),
            "control": _round(z1 + z2, 7),
            "target": _round(2 * z2, 7),
            "t_opt": t_opt,
        },
    }


@app.post("/api/gates")
def gates(req: GatesRequest) -> dict[str, Any]:
    """A transmon CZ and a cat CNOT: simulation next to the papers' formulas."""
    return {"cz": _cz(req.g_mhz, req.alpha_mhz), "cnot": _cnot(req.n, req.log_kappa_ratio)}


@app.get("/api/architectures")
def architectures() -> list[dict[str, Any]]:
    """Every scene in the gallery, with each part's size and where it comes from."""
    return [s.describe() for s in architecture_scenes().values()]


@disk_cached
def _spectrum(scene_id: str) -> dict[str, Any] | None:
    p = architecture_scenes()[scene_id].physics
    model = p["model"]
    if model == "tunable_transmon":
        sp = spectra_sim.tunable_transmon(p["E_J_max"], p["E_C"], p["d"])
    elif model == "transmon":
        sp = spectra_sim.transmon(p["E_J"], p["E_C"])
    elif model == "fluxonium":
        sp = spectra_sim.fluxonium(p["E_J"], p["E_C"], p["E_L"])
    elif model == "kerr_cat":
        sp = spectra_sim.kerr_cat(p["K"], 1.5 * p["eps2"])
    else:
        return None  # the dissipative cat has its own views
    return {
        "sim": {
            "x": _round(sp.x),
            "levels": _round(sp.levels),
            "x_label": sp.x_label,
            "unit": sp.unit,
        }
    }


@app.get("/api/architectures/{scene_id}/spectrum")
def architecture_spectrum(scene_id: str) -> dict[str, Any] | None:
    """The scene's energy levels against its control knob (flux, charge or drive)."""
    if scene_id not in architecture_scenes():
        raise HTTPException(404, f"no scene {scene_id!r}")
    return _spectrum(scene_id)


INPUTS: dict[snippets.View, type[BaseModel]] = {
    "phase": EvolveRequest,
    "rates": RatesRequest,
    "transmon": TransmonInputs,
    "qubit": QubitRequest,
    "pulse": PulseRequest,
    "qec": QecInputs,
    "compare": CompareInputs,
    "readout": ReadoutRequest,
    "gates": GatesRequest,
}


@app.post("/api/python/{view}")
def python_script(view: snippets.View, body: dict[str, Any]) -> dict[str, str]:
    """A Python script that reproduces the view with these inputs (validated like the view)."""
    try:
        inputs = INPUTS[view](**body).model_dump()
    except ValidationError as e:
        raise HTTPException(422, str(e)) from e
    return {"python": snippets.script(view, inputs)}
