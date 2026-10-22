"""Fill the disk cache with what the explorer asks for on first open (used by the Dockerfile).

    uv run python scripts/warm_cache.py

Each call below matches a view's default request in web/src, so the first visitor to a
hosted copy gets cached answers. If a view's defaults change, update its line here.
"""

from qubitlab import api

FLIP_RATE_SIZES = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7]  # views/FlipRates.tsx
COMPARE_SIZES = [1, 2, 3, 4, 5, 6, 7]  # views/CatVsTransmon.tsx
CAT = {"alpha_sq": 4, "kappa_phi": 0, "n_th": 0, "scheme": "dissipative"}

if __name__ == "__main__":
    api.checks()
    api.qec_curves()
    api.qec_shot(d=5, p=0.005, seed=1)
    api.qec_cat(api.CatCodeRequest(p_phase=0.005, p_bit=0.005 / 1e4))
    api.compare(t_ns=25.0)
    api.transmon_surface(n_levels=3)
    api.cat_evolve(api.EvolveRequest(**CAT, kappa_1=0, initial="vacuum", t_final=3, n_frames=31, grid=61))
    api.cat_rates(api.RatesRequest(**CAT, kappa_1=0.1, alpha_sq_values=FLIP_RATE_SIZES))
    api.cat_rates(api.RatesRequest(**CAT, kappa_1=0.01, alpha_sq_values=COMPARE_SIZES))
    api.qubit_run(api.QubitRequest(t1_ns=20000, t2_ns=20000, t_max_ns=100, n_points=121))
    api.pulse_optimize(api.PulseRequest())
