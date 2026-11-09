"""Ship the explorer's slowest solves in src/qubitlab/data/solves.json.

uv run python scripts/make_snapshots.py   # about half an hour on 4 cores

The cat CNOT at every position of the Gates view's sliders, and the QEC curves. The API serves
them only while the theory, sim and checks code matches the fingerprint written with them
(qubitlab.snapshot); after a code change it runs them live again until this is rerun.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from qubitlab import api, snapshot
from qubitlab.sim import qec as qec_sim
from qubitlab.sim import two_qubit as tq_sim


def main() -> None:
    rows: list[dict[str, Any]] = []
    for n, ratio in api.CNOT_GRID:
        k1, _, times = api.cnot_times(n, ratio)
        for T in times:
            run = tq_sim.cat_cnot(float(n), k1, T)
            rows.append({"key": api.solve_key("cat_cnot", n, k1, T), "result": asdict(run)})
        print(f"cat CNOT, n = {n}, log10 kappa_1 = {ratio}", flush=True)
    for code, d in api.QEC_CURVES:
        for p in api.P_GRID:
            m = qec_sim.memory(code, d, p, api.QEC_SHOTS)
            key = api.solve_key("memory", code, d, p, api.QEC_SHOTS)
            rows.append({"key": key, "result": asdict(m)})
    snapshot.write(rows, api.SHIPPED)
    print(f"{len(rows)} solves written to {api.SHIPPED}")


if __name__ == "__main__":
    main()
