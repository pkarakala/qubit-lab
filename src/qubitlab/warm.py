"""Precompute what the explorer asks for on first open, so the default views load at once.

    python -m qubitlab.warm

The Docker build runs it, and `qubitlab serve` and scripts/dev.sh run it in the background on
start (QUBITLAB_WARM=1). Each call matches a view's default request in web/src: if a view's
defaults change, update its line here. Results land in the disk cache (qubitlab.cache).
"""

from __future__ import annotations

import logging
import time

log = logging.getLogger("qubitlab.warm")

FLIP_RATE_SIZES = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7]  # views/FlipRates.tsx
COMPARE_SIZES = [1, 2, 3, 4, 5, 6, 7]  # views/CatVsTransmon.tsx
CAT = {"alpha_sq": 4, "kappa_phi": 0, "n_th": 0, "scheme": "dissipative"}


def warm() -> None:
    """Fill the cache, quickest and most-visited views first, the full checks last."""
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
