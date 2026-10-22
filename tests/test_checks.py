import pytest

from catlab.checks import run_all
from catlab.checks.base import REGISTRY


def _all_as_expected(include_slow: bool) -> None:
    results = run_all(include_slow=include_slow)
    surprises = [r.describe() for r in results if not r.as_expected]
    assert not surprises, surprises


def test_fast_checks_match_the_model_card() -> None:
    _all_as_expected(include_slow=False)


@pytest.mark.slow
def test_slow_checks_match_the_model_card() -> None:
    for c in REGISTRY:
        if c.slow:
            surprises = [r.describe() for r in c.run() if not r.as_expected]
            assert not surprises, surprises
