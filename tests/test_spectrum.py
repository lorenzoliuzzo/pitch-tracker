import numpy as np
import pytest

from pitch_tracker.notes import GUITAR_OPEN_STRINGS_HZ, cents_between
from pitch_tracker.spectrum import bin_range, interpolate_peak, peak_frequency
from pitch_tracker.synth import harmonic_tone

FS = 44100.0
N = 4096
STRINGS = list(GUITAR_OPEN_STRINGS_HZ.items())


def test_interpolation_recovers_the_vertex_of_a_parabola() -> None:
    bins = np.arange(10, dtype=float)
    parabola = -((bins - 4.3) ** 2)
    assert interpolate_peak(parabola, 4) == pytest.approx(4.3)


def test_interpolation_stays_within_half_a_bin_when_the_middle_is_not_a_peak() -> None:
    rising = np.array([0.0, 1.0, 1.1])
    assert interpolate_peak(rising, 1) == pytest.approx(1.5)
    assert interpolate_peak(np.array([1.1, 1.0, 0.0]), 1) == pytest.approx(0.5)


def test_interpolation_of_a_flat_top_stays_on_the_bin() -> None:
    assert interpolate_peak(np.array([1.0, 1.0, 1.0]), 1) == 1.0


@pytest.mark.parametrize(("name", "f"), STRINGS)
@pytest.mark.parametrize("offset_cents", [-30.0, 0.0, 30.0])
def test_pure_tone_is_within_four_cents(name: str, f: float, offset_cents: float) -> None:
    f_true = f * 2.0 ** (offset_cents / 1200.0)
    x = harmonic_tone(f_true, [1.0], 1.0, FS)[:N]
    assert abs(cents_between(peak_frequency(x, FS, 70.0, 1400.0), f_true)) < 4.0


def test_a_stronger_second_harmonic_is_read_an_octave_high() -> None:
    x = harmonic_tone(82.41, [0.2, 1.0, 0.8], 1.0, FS)[:N]
    f_hat = peak_frequency(x, FS, 70.0, 1400.0)
    assert cents_between(f_hat, 82.41) == pytest.approx(1200.0, abs=10.0)


def test_search_range_excludes_outside_peaks() -> None:
    loud_low = harmonic_tone(50.0, [1.0], 1.0, FS)[:N]
    quiet_in_range = harmonic_tone(200.0, [0.01], 1.0, FS)[:N]
    f_hat = peak_frequency(loud_low + quiet_in_range, FS, 100.0, 400.0)
    assert f_hat == pytest.approx(200.0, abs=1.0)


def test_silence_does_not_crash() -> None:
    assert np.isfinite(peak_frequency(np.zeros(N), FS, 70.0, 1400.0))


@pytest.mark.parametrize(("f_min", "f_max"), [(0.0, 100.0), (500.0, 100.0), (100.0, FS)])
def test_invalid_range_is_rejected(f_min: float, f_max: float) -> None:
    with pytest.raises(ValueError):
        bin_range(N, FS, f_min, f_max)


def test_record_too_short_for_the_range_is_rejected() -> None:
    with pytest.raises(ValueError):
        bin_range(64, FS, 70.0, 100.0)
