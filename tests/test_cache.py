"""The disk cache: one solve per key under concurrent callers, keys tied to the code."""

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from qubitlab import cache


def test_concurrent_callers_share_one_solve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    calls: list[int] = []

    @cache.disk_cached
    def slow(x: int) -> int:
        calls.append(x)
        time.sleep(0.3)
        return x * 2

    out: list[int] = []
    threads = [threading.Thread(target=lambda: out.append(slow(3))) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert out == [6, 6, 6, 6]
    assert calls == [3]


def test_other_code_does_not_reuse_a_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    calls: list[int] = []

    @cache.disk_cached
    def double(x: int) -> int:
        calls.append(x)
        return x * 2

    assert double(3) == double(3) == 6
    monkeypatch.setattr(cache, "CODE", "edited")  # as after a code change
    assert double(3) == 6
    assert calls == [3, 3]


def test_the_default_folder_is_per_user_not_per_working_directory(tmp_path: Path) -> None:
    env = {k: v for k, v in os.environ.items() if k != "QUBITLAB_CACHE"}
    env["XDG_CACHE_HOME"] = str(tmp_path / "xdg")
    code = "from qubitlab.cache import CACHE_DIR; print(CACHE_DIR)"
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, cwd=tmp_path, capture_output=True, text=True
    )
    assert out.stdout.strip() == str(tmp_path / "xdg" / "qubitlab"), out.stderr
