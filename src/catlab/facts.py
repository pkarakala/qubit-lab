"""Measured numbers from experiments, with the exact source.

These are hardware results, in physical units. The simulation is dimensionless, so the explorer
shows them beside the curves, never fitted to them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from catlab.provenance import Kind


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
)
