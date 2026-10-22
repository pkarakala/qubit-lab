from qubitlab.checks.base import REGISTRY, run_all
from qubitlab.provenance import bib_entries
from qubitlab.theory import FORMULAS


def test_every_formula_source_is_in_the_bib() -> None:
    entries = bib_entries()
    for formula in FORMULAS.values():
        for source in formula.sources:
            assert source.key in entries, f"{formula.key} cites missing {source.key}"
            assert source.locator, f"{formula.key}: source {source.key} has no locator"


def test_every_bib_entry_can_be_found() -> None:
    for key, entry in bib_entries().items():
        findable = {"doi", "url", "howpublished", "publisher"} & set(entry)
        assert findable, f"{key} has no DOI, URL or publisher"


def test_every_check_points_at_a_real_formula() -> None:
    run_all(include_slow=False)  # registers the check modules
    assert REGISTRY
    for result in run_all(include_slow=False):
        assert result.formula in FORMULAS, result.key


def test_facts_and_devices_are_sourced() -> None:
    from qubitlab.facts import DEVICES, FACT, FACTS

    keys = set(bib_entries())
    assert {f.cite for f in FACTS} <= keys
    for d in DEVICES:
        assert FACT[d.t_bit].qubit == FACT[d.t_phase].qubit == d.qubit
        assert FACT[d.t_bit].unit == FACT[d.t_phase].unit == "s"
