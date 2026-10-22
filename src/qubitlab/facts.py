"""Measured numbers from experiments, with the exact source.

These are hardware results, in physical units. The simulation is dimensionless, so the explorer
shows them beside the curves, never fitted to them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from qubitlab.provenance import Kind


@dataclass(frozen=True)
class Fact:
    key: str
    label: str
    value: float
    unit: str
    alpha_sq: float | None  # cat size the number was measured at, if any
    cite: str
    kind: Kind
    locator: str
    qubit: str = "cat"  # "cat" or "transmon"

    def describe(self) -> dict[str, object]:
        d = asdict(self)
        d["kind"] = self.kind.value
        return d


PR = Kind.PEER_REVIEWED
CLAIM = Kind.COMPANY_CLAIM

FACTS: tuple[Fact, ...] = (
    Fact(
        "lescanne2020.tx_gain",
        "Bit-flip time gain per added photon",
        4.2,
        "x",
        None,
        "lescanne2020",
        PR,
        "Fig. 3",
    ),
    Fact(
        "lescanne2020.tx_max",
        "Bit-flip time ceiling",
        1e-3,
        "s",
        5.0,
        "lescanne2020",
        PR,
        "Fig. 3, saturates for |alpha|^2 >= 5",
    ),
    Fact("reglade2024.tx_2.5", "Bit-flip time", 313e-6, "s", 2.5, "reglade2024", PR, "Fig. 3 text"),
    Fact("reglade2024.tx_5.6", "Bit-flip time", 56e-3, "s", 5.6, "reglade2024", PR, "Fig. 3 text"),
    Fact("reglade2024.tx_11.3", "Bit-flip time", 15.0, "s", 11.3, "reglade2024", PR, "Fig. 3 text"),
    Fact(
        "reglade2024.tz_2.5", "Phase-flip time", 2.7e-6, "s", 2.5, "reglade2024", PR, "Fig. 3 text"
    ),
    Fact(
        "reglade2024.tz_11.3", "Phase-flip time", 490e-9, "s", 11.3, "reglade2024", PR, "abstract"
    ),
    Fact(
        "alicebob2025.tx",
        "Bit-flip time, preliminary",
        3600.0,
        "s",
        None,
        "tqi2025alicebob",
        CLAIM,
        "over one hour, Galvanic Cat design",
    ),
    Fact(
        "alicebob2024.boson4",
        "Bit-flip time, Boson 4",
        430.0,
        "s",
        None,
        "tqi2025alicebob",
        CLAIM,
        "previous record, 2024",
    ),
    Fact(
        "willow2025.t1",
        "T1, mean of 105 qubits",
        68e-6,
        "s",
        None,
        "willow2025",
        PR,
        "Sec. II, first paragraph",
        "transmon",
    ),
    Fact(
        "willow2025.t2",
        "T2 (CPMG), mean of 105 qubits",
        89e-6,
        "s",
        None,
        "willow2025",
        PR,
        "Sec. II, first paragraph",
        "transmon",
    ),
    Fact(
        "willow2025.gate_1q",
        "Single-qubit gate duration",
        25e-9,
        "s",
        None,
        "willow2025",
        PR,
        "Supplement Sec. I",
        "transmon",
    ),
    Fact(
        "place2021.t1",
        "T1, best device, peak",
        0.36e-3,
        "s",
        None,
        "place2021",
        PR,
        "main text, 3rd paragraph, Fig. 1C",
        "transmon",
    ),
    Fact(
        "place2021.t2",
        "T2 (echo), best device, time-averaged",
        0.20e-3,
        "s",
        None,
        "place2021",
        PR,
        "main text, 4th paragraph, Fig. 2C",
        "transmon",
    ),
)

FACT = {f.key: f for f in FACTS}


@dataclass(frozen=True)
class Device:
    """One measured qubit: the time for a bit-flip and for a phase-flip, from the same paper.

    For a cat these are T_X and T_Z, decay times of <Z_alpha> and <X_alpha> (reglade2024,
    Fig. 3 caption). For a transmon they are T1 and T2, the decay times of <Z> and <X>.
    """

    key: str
    label: str
    qubit: str
    t_bit: str  # Fact key
    t_phase: str  # Fact key


DEVICES: tuple[Device, ...] = (
    Device("reglade2024.2.5", "Cat, |α|² = 2.5", "cat", "reglade2024.tx_2.5", "reglade2024.tz_2.5"),
    Device(
        "reglade2024.11.3", "Cat, |α|² = 11.3", "cat", "reglade2024.tx_11.3", "reglade2024.tz_11.3"
    ),
    Device("willow2025", "Transmon, Willow mean", "transmon", "willow2025.t1", "willow2025.t2"),
    Device("place2021", "Transmon, best Ta device", "transmon", "place2021.t1", "place2021.t2"),
)
