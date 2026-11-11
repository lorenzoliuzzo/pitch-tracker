import numpy as np
import pytest

from pitch_tracker.live import Tuner, format_reading, needle
from pitch_tracker.notes import Note
from pitch_tracker.synth import harmonic_tone

FS = 44100.0
WEAK = [0.2, 1.0, 0.8]


def feed(tuner: Tuner, x: np.ndarray, block: int = 1024) -> Note | None:
    note = None
    for start in range(0, x.size, block):
        tuner.push(x[start : start + block])
        note = tuner.reading()
    return note


def test_no_reading_until_the_window_is_full() -> None:
    tuner = Tuner()
    tuner.push(harmonic_tone(110.0, WEAK, 0.05, FS))
    assert tuner.reading() is None


def test_low_e_with_a_weak_fundamental_is_named_from_the_stream() -> None:
    note = feed(Tuner(), harmonic_tone(82.41, WEAK, 1.0, FS))
    assert note is not None
    assert note.name == "E2"
    assert abs(note.cents) < 2.0


@pytest.mark.parametrize("method", ["hps", "autocorr"])
def test_both_weak_fundamental_methods_work_live(method: str) -> None:
    note = feed(Tuner(method=method), harmonic_tone(196.0 * 2 ** (-20 / 1200), WEAK, 1.0, FS))
    assert note is not None
    assert note.name == "G3"
    assert note.cents == pytest.approx(-20.0, abs=3.0)


def test_quiet_input_is_gated_out() -> None:
    quiet = 0.001 * harmonic_tone(110.0, WEAK, 1.0, FS)
    assert feed(Tuner(), quiet) is None


def test_gate_forgets_the_previous_note() -> None:
    tuner = Tuner()
    assert feed(tuner, harmonic_tone(110.0, WEAK, 1.0, FS)) is not None
    assert feed(tuner, np.zeros(8192)) is None
    assert len(tuner.recent) == 0


def test_median_smoothing_ignores_a_single_outlier_frame() -> None:
    tuner = Tuner(smoothing=5)
    for f in (110.0, 110.0, 220.0, 110.0, 110.0):
        tuner.recent.append(f)
    tuner.buffer = harmonic_tone(110.0, WEAK, 1.0, FS)[: tuner.buffer.size]
    tuner.filled = tuner.buffer.size
    note = tuner.reading()
    assert note is not None
    assert note.name == "A2"


def test_pushing_more_than_a_window_keeps_the_latest_samples() -> None:
    tuner = Tuner(window=8)
    tuner.push(np.arange(20, dtype=float))
    assert tuner.buffer.tolist() == [12, 13, 14, 15, 16, 17, 18, 19]


def test_needle_marks_the_cents() -> None:
    assert needle(0.0).count("O") == 1
    assert needle(None) == "-" * 20 + "|" + "-" * 20
    assert needle(-50.0).index("o") == 0
    assert needle(500.0).rindex("o") == 40


def test_format_reading_names_the_note_and_the_direction() -> None:
    flat = format_reading(Note("A2", 110.0, -23.8))
    assert flat.startswith("A2")
    assert "-23.8c" in flat
    assert "listening" in format_reading(None)
