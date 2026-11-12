import subprocess
import sys

import pytest
import stim

from qubitlab.sim import qec


def test_qubit_counts_match_the_layouts() -> None:
    assert [qec.physical_qubits("repetition", d) for d in (3, 5)] == [5, 9]
    assert [qec.physical_qubits("surface", d) for d in (3, 5)] == [17, 49]  # rotated: 2d^2 - 1


def test_bigger_code_helps_below_threshold() -> None:
    small = qec.memory("surface", 3, 0.002, shots=50_000)
    large = qec.memory("surface", 5, 0.002, shots=50_000)
    assert large.per_round < small.per_round


def test_per_round_inverts_the_total() -> None:
    # An odd number of flips in 3 rounds: P = [1 - (1 - 2 eps)^3] / 2 (willow2025, Sec. IV).
    m = qec.Memory("surface", 3, 0.01, rounds=3, shots=1000, failures=271)
    assert (1 - (1 - 2 * m.per_round) ** 3) / 2 == pytest.approx(0.271)


def test_a_random_logical_qubit_errs_half_the_time_per_round() -> None:
    # Above threshold P can sample above 1/2; (1 - 2P)^(1/r) would then be complex.
    m = qec.Memory("repetition", 13, 0.05, rounds=13, shots=1000, failures=512)
    assert m.per_round == 0.5


def test_shot_matches_every_fired_detector() -> None:
    shot = qec.sample_shot(5, 0.01, seed=1)
    assert sum(q[2] for q in shot.qubits) == 25  # d^2 data qubits
    in_pairs = {i for pair in shot.matched for i in pair if i >= 0}
    assert in_pairs == set(shot.fired)


CONCURRENT_FIRST_CALLS = """
import threading
from qubitlab.sim import qec
ts = [threading.Thread(target=qec.sample_shot, args=(5, 0.005, s)) for s in range(4)]
ts.append(threading.Thread(target=qec.memory, args=("surface", 3, 0.005, 1000)))
for t in ts: t.start()
for t in ts: t.join()
"""


def test_concurrent_first_samples_do_not_deadlock() -> None:
    # Before Stim's NumPy bindings were set up at import, about 2 runs in 5 of this hung forever:
    # the server's request threads and its warm-up make these first calls at the same time.
    for _ in range(3):
        subprocess.run([sys.executable, "-c", CONCURRENT_FIRST_CALLS], check=True, timeout=120)


def test_hot_qubit_scales_only_its_own_noise() -> None:
    c = qec.circuit("surface", 3, 0.001)
    q = qec.centre_data_qubit(c)
    hot = qec.with_hot_qubit(c, q, 10)
    # Same detectors, so the old decoder can read the new shots.
    assert hot.num_detectors == c.num_detectors and hot.num_observables == c.num_observables
    for inst in hot.flattened():
        if inst.name in qec.NOISE:
            touches = any(t.value == q for t in inst.targets_copy())
            # 0.001 * 10 by hand; every other target keeps 0.001.
            assert inst.gate_args_copy() == pytest.approx([0.01 if touches else 0.001])


def test_hot_qubit_caps_at_full_randomness() -> None:
    c = qec.circuit("surface", 3, 0.04)
    hot = qec.with_hot_qubit(c, qec.centre_data_qubit(c), 30)  # 1.2 asked; caps apply
    for inst in hot.flattened():
        if inst.name in qec.NOISE:
            assert inst.gate_args_copy()[0] <= qec.NOISE[inst.name][1]


def test_factor_one_changes_nothing() -> None:
    c = qec.circuit("surface", 3, 0.002)
    same = qec.with_hot_qubit(c, qec.centre_data_qubit(c), 1)

    def errors(x: stim.Circuit) -> list[str]:
        # Splitting an instruction reorders the model's lines; the set of errors must not change.
        return sorted(str(e) for e in x.detector_error_model().flattened() if e.type == "error")

    assert errors(same) == errors(c)


def test_drift_lights_up_the_neighbours_and_costs_the_old_decoder() -> None:
    r = qec.drift(3, 0.003, 20, shots=40_000)
    assert r.hot == (3.0, 3.0)  # the middle of a d = 3 patch, data qubits at odd coordinates
    worst = sorted(r.rates, key=lambda s: s[3] / s[2])[-4:]
    # The four measure qubits next to it, at distance sqrt(2) on Stim's grid.
    assert {(x, y) for x, y, _, _ in worst} == {(2.0, 2.0), (2.0, 4.0), (4.0, 2.0), (4.0, 4.0)}
    assert r.stale > r.informed > r.before
