"""Where theory meets simulation.

Each check evaluates one closed-form formula from `qubitlab.theory` and the matching quantity
from `qubitlab.sim`, then reports both numbers, the relative error and a verdict. A check that
fails is a finding, not a bug to hide: it marks where a formula's assumptions break.
"""

from qubitlab.checks.base import Check, CheckResult, Verdict, run_all

__all__ = ["Check", "CheckResult", "Verdict", "run_all"]
