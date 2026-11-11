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

# Stim's first conversion to a NumPy array sets up pybind11's NumPy bindings behind a C++ static
# guard while holding the GIL. Two threads making that first call at once deadlock the whole
# process (the server's request threads and its warm-up do exactly that). Make it here, once,
# while the module is imported on one thread.
stim.Circuit("M 0\nDETECTOR rec[-1]").compile_detector_sampler().sample(1)

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
        """Logical error per round, eps, from the error after all rounds, P.

        Each round flips the logical qubit with probability eps, and two flips cancel, so the
        qubit ends up wrong when the number of flips is odd: P = [1 - (1 - 2 eps)^rounds] / 2.
        Inverted as in willow2025 (arXiv v1, Sec. IV, unnumbered display):
        eps = [1 - (1 - 2P)^(1/rounds)] / 2. At P >= 1/2 the qubit is random and eps is 1/2.
        """
        return float((1 - max(0.0, 1 - 2 * self.logical_error) ** (1 / self.rounds)) / 2)


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


# Each Stim noise channel, the number of qubits it acts on, and its largest meaningful
# probability: beyond it the channel stops adding randomness (DEPOLARIZE1 at 3/4, DEPOLARIZE2 at
# 15/16, a bit or phase flip at 1/2).
NOISE: dict[str, tuple[int, float]] = {
    "DEPOLARIZE1": (1, 0.75),
    "DEPOLARIZE2": (2, 15 / 16),
    "X_ERROR": (1, 0.5),
    "Z_ERROR": (1, 0.5),
}


def with_hot_qubit(c: stim.Circuit, qubit: int, factor: float) -> stim.Circuit:
    """The same circuit with every noise channel that touches `qubit` made `factor` times likelier.

    A defect that drifts onto one qubit (a two-level system, say) does this. Stim has no
    transform for it, so this walks the flattened circuit and splits each noise instruction
    into the targets it leaves alone and the ones it scales. Detectors and the observable are
    untouched, so a decoder built for `c` still reads the shots of the result.
    """
    out = stim.Circuit()
    for inst in c.flattened():
        spec = NOISE.get(inst.name)
        if spec is None:
            out.append(inst)
            continue
        arity, cap = spec
        (p,) = inst.gate_args_copy()
        t = inst.targets_copy()
        groups = [t[i : i + arity] for i in range(0, len(t), arity)]
        hit = [any(x.value == qubit for x in g) for g in groups]
        cold = [x for g, h in zip(groups, hit, strict=True) if not h for x in g]
        hot = [x for g, h in zip(groups, hit, strict=True) if h for x in g]
        if cold:
            out.append(inst.name, cold, p)
        if hot:
            out.append(inst.name, hot, min(cap, p * factor))
    return out


def centre_data_qubit(c: stim.Circuit) -> int:
    """The data qubit nearest the middle of the patch (data qubits sit at odd coordinates)."""
    coords = c.get_final_qubit_coordinates()
    data = [q for q, (x, y) in coords.items() if int(x) % 2 == 1 and int(y) % 2 == 1]
    mx = float(np.mean([coords[q][0] for q in data]))
    my = float(np.mean([coords[q][1] for q in data]))
    return int(min(data, key=lambda q: (coords[q][0] - mx) ** 2 + (coords[q][1] - my) ** 2))


@dataclass(frozen=True)
class Drift:
    """A surface code before and after one qubit got worse, and three decoders' luck with it.

    `before` decodes calibration-time shots with the calibrated model. `stale` decodes the
    drifted shots with that same, now wrong, model. `informed` decodes them with the true new
    noise, the best a refit can do. Rates are logical error per shot over `shots` runs; `stale`
    and `informed` read the same shots, so their difference is not sampling noise.
    `rates` lists each measure qubit as (x, y, detection rate before, after).
    """

    distance: int
    p: float
    factor: float
    hot: tuple[float, float]
    shots: int
    before: float
    stale: float
    informed: float
    rates: list[tuple[float, float, float, float]]


def drift(distance: int, p: float, factor: float, shots: int = 100_000, seed: int = 0) -> Drift:
    good = circuit("surface", distance, p)
    q = centre_data_qubit(good)
    bad = with_hot_qubit(good, q, factor)

    def matcher(c: stim.Circuit) -> pymatching.Matching:
        dem = c.detector_error_model(decompose_errors=True)
        return pymatching.Matching.from_detector_error_model(dem)

    det0, obs0 = good.compile_detector_sampler(seed=seed).sample(shots, separate_observables=True)
    det1, obs1 = bad.compile_detector_sampler(seed=seed + 1).sample(
        shots, separate_observables=True
    )

    def wrong(m: pymatching.Matching, det: np.ndarray, obs: np.ndarray) -> float:
        return float(np.mean(np.any(m.decode_batch(det) != obs, axis=1)))

    # Average each measure qubit's detectors over rounds: detector coordinates are (x, y, round).
    where = [tuple(v[:2]) for v in good.get_detector_coordinates().values()]
    sites = sorted(set(where))
    rates = []
    for s in sites:
        cols = [i for i, w in enumerate(where) if w == s]
        rates.append((s[0], s[1], float(det0[:, cols].mean()), float(det1[:, cols].mean())))
    x, y = good.get_final_qubit_coordinates()[q]
    calibrated = matcher(good)
    return Drift(
        distance,
        p,
        factor,
        (x, y),
        shots,
        before=wrong(calibrated, det0, obs0),
        stale=wrong(calibrated, det1, obs1),
        informed=wrong(matcher(bad), det1, obs1),
        rates=rates,
    )
