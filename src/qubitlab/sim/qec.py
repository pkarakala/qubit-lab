"""Memory experiments for the surface code and the repetition code, with Stim and PyMatching.

Stim builds each circuit (`stim.Circuit.generated`) with uniform circuit noise of strength p
after every gate, reset and measurement, samples it, and PyMatching decodes the syndromes.
Nothing here is a closed form; `qubitlab.checks` compares the scaling to Fowler et al.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pymatching
import stim

Code = Literal["surface", "repetition"]
TASK: dict[Code, str] = {
    "surface": "surface_code:rotated_memory_z",
    "repetition": "repetition_code:memory",
}


def circuit(code: Code, distance: int, p: float, rounds: int | None = None) -> stim.Circuit:
    return stim.Circuit.generated(
        TASK[code],
        distance=distance,
        rounds=rounds or distance,
        after_clifford_depolarization=p,
        before_round_data_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
    )


def physical_qubits(code: Code, distance: int) -> int:
    """Qubits the circuit acts on. Stim's num_qubits is the largest index plus one, and its
    surface code layout skips indices, so count the targets instead."""
    flat = circuit(code, distance, 0.0).flattened()
    return len({t.value for op in flat for t in op.targets_copy() if t.is_qubit_target})


@dataclass(frozen=True)
class Memory:
    code: Code
    distance: int
    p: float
    rounds: int
    shots: int
    failures: int

    @property
    def logical_error(self) -> float:
        """Probability the logical qubit is wrong after all rounds."""
        return self.failures / self.shots

    @property
    def per_round(self) -> float:
        """Logical error per round, from (1 - P_round)^rounds = 1 - P_total."""
        return float(1 - (1 - self.logical_error) ** (1 / self.rounds))


def memory(code: Code, distance: int, p: float, shots: int = 100_000, seed: int = 0) -> Memory:
    c = circuit(code, distance, p)
    dem = c.detector_error_model(decompose_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(dem)
    det, obs = c.compile_detector_sampler(seed=seed).sample(shots, separate_observables=True)
    wrong = np.any(matcher.decode_batch(det) != obs, axis=1)
    return Memory(code, distance, p, distance, shots, int(wrong.sum()))


def slope(
    code: Code, distance: int, p_lo: float, p_hi: float, shots: int, min_failures: int = 100
) -> float:
    """Slope of log P_L against log p between two noise strengths.

    Each P_L carries a relative sampling error of about 1/sqrt(failures), so with fewer than
    `min_failures` the slope is mostly noise; that raises instead of returning a number.
    """
    lo, hi = memory(code, distance, p_lo, shots), memory(code, distance, p_hi, shots)
    if min(lo.failures, hi.failures) < min_failures:
        raise ValueError(f"under {min_failures} failures sampled; raise shots or p")
    return float(np.log(hi.per_round / lo.per_round) / np.log(p_hi / p_lo))


def threshold(
    code: Code, d_small: int, d_large: int, p_lo: float, p_hi: float, shots: int
) -> float:
    """Noise strength where the two distances give the same logical error (bisection)."""

    def gap(p: float) -> float:
        return memory(code, d_large, p, shots).per_round - memory(code, d_small, p, shots).per_round

    lo, hi = p_lo, p_hi
    if gap(lo) >= 0 or gap(hi) <= 0:
        raise ValueError("threshold is not bracketed")
    for _ in range(12):
        mid = (lo * hi) ** 0.5
        if gap(mid) < 0:
            lo = mid
        else:
            hi = mid
    return float((lo * hi) ** 0.5)


@dataclass(frozen=True)
class Shot:
    """One sampled run of the surface code, as a space-time picture of the decoding."""

    qubits: list[tuple[float, float, bool]]  # x, y, is_data
    detectors: list[tuple[float, float, float]]  # x, y, round
    fired: list[int]  # detectors that flagged an error
    matched: list[tuple[int, int]]  # pairs PyMatching joined; -1 is the boundary
    logical_error: bool


def sample_shot(distance: int, p: float, seed: int = 0) -> Shot:
    c = circuit("surface", distance, p)
    matcher = pymatching.Matching.from_detector_error_model(
        c.detector_error_model(decompose_errors=True)
    )
    det, obs = c.compile_detector_sampler(seed=seed).sample(1, separate_observables=True)
    # In Stim's rotated layout, data qubits sit at odd coordinates and measure qubits at even.
    qubits = [(x, y, int(x) % 2 == 1) for x, y in c.get_final_qubit_coordinates().values()]
    coords = c.get_detector_coordinates()
    detectors = [tuple(coords[i][:3]) for i in range(c.num_detectors)]
    pairs = [(int(a), int(b)) for a, b in matcher.decode_to_edges_array(det[0])]
    wrong = bool(np.any(matcher.decode(det[0]) != obs[0]))
    return Shot(qubits, detectors, [int(i) for i in np.flatnonzero(det[0])], pairs, wrong)
