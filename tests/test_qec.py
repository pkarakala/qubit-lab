import subprocess
import sys

import pytest

from qubitlab.sim import qec


def test_qubit_counts_match_the_layouts() -> None:
    assert [qec.physical_qubits("repetition", d) for d in (3, 5)] == [5, 9]
    assert [qec.physical_qubits("surface", d) for d in (3, 5)] == [17, 49]  # rotated: 2d^2 - 1


def test_bigger_code_helps_below_threshold() -> None:
    small = qec.memory("surface", 3, 0.002, shots=50_000)
    large = qec.memory("surface", 5, 0.002, shots=50_000)
    assert large.per_round < small.per_round


def test_per_round_inverts_the_total() -> None:
    m = qec.Memory("surface", 3, 0.01, rounds=3, shots=1000, failures=271)
    assert 1 - (1 - m.per_round) ** 3 == pytest.approx(0.271)


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
