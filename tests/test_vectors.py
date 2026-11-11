import json
from pathlib import Path
from typing import Any

import pytest

from pitch_tracker.vectors import DEFAULT_PATH, build_vectors

COMMITTED = Path(__file__).parent.parent / DEFAULT_PATH


def assert_close(a: Any, b: Any, path: str = "") -> None:
    if isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for key in a:
            assert_close(a[key], b[key], f"{path}/{key}")
    elif isinstance(a, list):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            assert_close(x, y, f"{path}[{i}]")
    elif isinstance(a, float) or isinstance(b, float):
        assert a == pytest.approx(b, rel=1e-9, abs=1e-9), path
    else:
        assert a == b, path


def test_committed_vectors_are_up_to_date() -> None:
    committed = json.loads(COMMITTED.read_text())
    assert_close(build_vectors(), committed)


def test_vectors_cover_every_method_and_the_no_pitch_case() -> None:
    vectors = build_vectors()
    assert len(vectors["tones"]) == 2 * 6 * 3 + 1
    silence = vectors["tones"][-1]
    assert silence["expected"]["autocorr"] is None
    assert all(set(t["expected"]) == {"peak", "hps", "autocorr"} for t in vectors["tones"])
