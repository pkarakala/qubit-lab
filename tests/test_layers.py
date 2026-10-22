"""The theory / sim / checks split is enforced, not just documented."""

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "qubitlab"


def imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_theory_never_simulates() -> None:
    for path in (SRC / "theory").glob("*.py"):
        bad = {n for n in imports(path) if n.startswith(("dynamiqs", "qubitlab.sim"))}
        assert not bad, f"{path.name} imports {bad}"


def test_sim_never_uses_closed_forms() -> None:
    for path in (SRC / "sim").glob("*.py"):
        bad = {n for n in imports(path) if n.startswith(("qubitlab.theory", "sympy"))}
        assert not bad, f"{path.name} imports {bad}"
