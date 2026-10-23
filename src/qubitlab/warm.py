"""Precompute what the explorer asks for on first open, so the default views load at once.

    python -m qubitlab.warm

The Docker build runs it, and `qubitlab serve` and scripts/dev.sh run it in the background on
start (QUBITLAB_WARM=1). Each call matches a view's default request in web/src: if a view's
defaults change, update its line here. Results land in the disk cache (qubitlab.cache).
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

log = logging.getLogger("qubitlab.warm")

FLIP_RATE_SIZES = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7]  # views/FlipRates.tsx
COMPARE_SIZES = [1, 2, 3, 4, 5, 6, 7]  # views/CatVsTransmon.tsx
CAT = {"alpha_sq": 4, "kappa_phi": 0, "n_th": 0, "scheme": "dissipative"}
PRESETS_PATH = Path(__file__).resolve().parent / "data" / "presets.json"


def presets() -> list[dict[str, Any]]:
    """Settings from published experiments, each citing its paper (data/presets.json)."""
    data: list[dict[str, Any]] = json.loads(PRESETS_PATH.read_text(encoding="utf-8"))["presets"]
    return data


def preset_steps() -> list[Callable[[], object]]:
    """One solve per preset, built as its view sends it (web/src), so the cache answers it.

    The request models are built here, so a bad preset fails at once, not in the background.
    """
    from qubitlab import api

    steps: list[Callable[[], object]] = []
    for preset in presets():
        view, p = preset["view"], preset["params"]
        step: Callable[[], object]
        if view == "phase":
            step = partial(api.cat_evolve, api.EvolveRequest(**p))
        elif view == "rates":  # views/FlipRates.tsx sends its knobs plus the fixed sizes
            rates = api.RatesRequest(**{**CAT, **p, "alpha_sq_values": FLIP_RATE_SIZES})
            step = partial(api.cat_rates, rates)
        elif view == "qubit":
            step = partial(api.qubit_run, api.QubitRequest(**p))
        elif view == "pulse":
            step = partial(api.pulse_optimize, api.PulseRequest(**p))
        elif view == "qec_cat":
            step = partial(api.qec_cat, api.CatCodeRequest(**p))
        elif view == "compare":
            step = partial(api.compare, t_ns=float(p["t_ns"]))
        elif view in ("transmon", "qec"):  # those two answer in well under a second
            continue
        else:
            raise ValueError(f"preset for unknown view {view!r}: {preset['name']}")
        steps.append(step)
    return steps


def warm() -> None:
    """Fill the cache: each view's defaults, then the published presets, then the checks."""
    from qubitlab import api

    steps = [
        lambda: api.cat_evolve(
            api.EvolveRequest(**CAT, kappa_1=0, initial="vacuum", t_final=3, n_frames=31, grid=61)
        ),
        lambda: api.cat_rates(
            api.RatesRequest(**CAT, kappa_1=0.1, alpha_sq_values=FLIP_RATE_SIZES)
        ),
        lambda: api.transmon_surface(n_levels=3),
        lambda: api.qubit_run(
            api.QubitRequest(t1_ns=20000, t2_ns=20000, t_max_ns=100, n_points=121)
        ),
        lambda: api.compare(t_ns=25.0),
        lambda: api.cat_rates(api.RatesRequest(**CAT, kappa_1=0.01, alpha_sq_values=COMPARE_SIZES)),
        lambda: api.qec_shot(d=5, p=0.005, seed=1),
        lambda: api.qec_cat(api.CatCodeRequest(p_phase=0.005, p_bit=0.005 / 1e4)),
        lambda: api.pulse_optimize(api.PulseRequest()),
        api.qec_curves,
        *preset_steps(),
        api.checks,
    ]
    start = time.perf_counter()
    for step in steps:
        try:
            step()
        except Exception:  # a failed warm-up only costs speed; the request will report it
            log.exception("warm-up step failed")
    log.info("warm-up done in %.0f s", time.perf_counter() - start)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    warm()
