import math

import pytest

from pitch_tracker.notes import GUITAR_OPEN_STRINGS_HZ, cents_between, nearest_note


def test_octave_is_1200_cents() -> None:
    assert cents_between(220.0, 110.0) == pytest.approx(1200.0)


def test_equal_frequencies_are_zero_cents() -> None:
    assert cents_between(82.41, 82.41) == 0.0


def test_semitone_is_100_cents() -> None:
    assert cents_between(440.0 * 2 ** (1 / 12), 440.0) == pytest.approx(100.0)


def test_flat_a2() -> None:
    note = nearest_note(108.5)
    assert note.name == "A2"
    assert note.cents == pytest.approx(-23.8, abs=0.05)


@pytest.mark.parametrize("name", GUITAR_OPEN_STRINGS_HZ)
def test_open_strings_name_themselves(name: str) -> None:
    note = nearest_note(GUITAR_OPEN_STRINGS_HZ[name])
    assert note.name == name
    assert abs(note.cents) < 1.0


def test_halfway_note_stays_within_50_cents() -> None:
    for f in (90.0, 100.0, 130.0, 300.0):
        assert abs(nearest_note(f).cents) <= 50.0 + 1e-9


def test_a4_reference_shifts_the_note_frequency() -> None:
    assert nearest_note(442.0, a4=442.0).cents == pytest.approx(0.0, abs=1e-9)
    assert math.isclose(nearest_note(440.0, a4=442.0).cents, cents_between(440.0, 442.0))


def test_nonpositive_frequency_is_rejected() -> None:
    with pytest.raises(ValueError):
        nearest_note(0.0)
