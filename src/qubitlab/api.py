"""FastAPI backend for the explorer. Every response keeps theory and simulation apart."""

from __future__ import annotations

import math
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Literal

import anyio
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from qubitlab import __version__
from qubitlab.cache import disk_cached
from qubitlab.facts import DEVICES, FACT, FACTS
from qubitlab.provenance import bib_entries
from qubitlab.sim import cat as cat_sim
from qubitlab.sim import custom as custom_sim
from qubitlab.sim import pulse_optimizer as po_sim
from qubitlab.sim import qec as qec_sim
from qubitlab.sim import transmon as tr_sim
from qubitlab.sim import transmon_dynamics as qb_sim
from qubitlab.theory import FORMULAS
from qubitlab.theory import cat as cat_th
from qubitlab.theory import qec as qec_th
from qubitlab.theory import qubit as qb_th
from qubitlab.theory import transmon as tr_th

# Public hosting knobs, all off by default (see docs/deploy.md):
# QUBITLAB_SOLVES   most simulations running at once; the rest wait their turn.
# QUBITLAB_RATE     requests per client, e.g. "30/minute".
# QUBITLAB_STATIC   the built explorer (web/dist), served from the same process.
SOLVES = int(os.environ.get("QUBITLAB_SOLVES", "0"))
RATE = os.environ.get("QUBITLAB_RATE")
STATIC = os.environ.get("QUBITLAB_STATIC")


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    if SOLVES:
        # Sync endpoints run in AnyIO's thread pool, so its size caps concurrent solves.
        anyio.to_thread.current_default_thread_limiter().total_tokens = SOLVES
    yield


app = FastAPI(title="qubitlab", version=__version__, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)
if RATE:
    app.state.limiter = Limiter(key_func=get_remote_address, default_limits=[RATE])
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]
    app.add_middleware(SlowAPIMiddleware)


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
                    "phase_flip_rate": r.phase_flip_rate if r.x_decay_resolved else None,
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
    from qubitlab.checks import run_all

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

    The default t is the Willow single-qubit gate time (willow2025, Supplement Sec. I).
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


# Mounted last, so every /api route above takes precedence.
if STATIC:
    app.mount("/", StaticFiles(directory=STATIC, html=True), name="explorer")
