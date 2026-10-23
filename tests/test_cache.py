"""The disk cache under concurrent callers, as when the warm-up and a click ask for one solve."""

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
