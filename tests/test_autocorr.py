import numpy as np
import pytest

from pitch_tracker.autocorr import autocorr_frequency, autocorrelation
from pitch_tracker.notes import GUITAR_OPEN_STRINGS_HZ, cents_between
from pitch_tracker.synth import harmonic_tone

FS = 44100.0
N = 4096
WEAK = [0.2, 1.0, 0.8]


def test_fft_autocorrelation_matches_the_direct_sum() -> None:
    x = np.random.default_rng(1).standard_normal(64)
    direct = np.array([np.dot(x[: x.size - m], x[m:]) for m in range(x.size)])
    assert np.allclose(autocorrelation(x), direct)


def test_lag_zero_is_the_energy() -> None:
    x = np.random.default_rng(2).standard_normal(100)
    assert autocorrelation(x)[0] == pytest.approx(np.sum(x**2))


@pytest.mark.parametrize(("name", "f"), list(GUITAR_OPEN_STRINGS_HZ.items()))
@pytest.mark.parametrize("offset_cents", [-30.0, 0.0, 30.0])
@pytest.mark.parametrize("noise", [0.0, 0.1])
def test_weak_fundamental_is_found_within_two_cents(
    name: str, f: float, offset_cents: float, noise: float
) -> None:
    f_true = f * 2.0 ** (offset_cents / 1200.0)
    x = harmonic_tone(f_true, WEAK, 1.0, FS, noise=noise)[:N]
    f_hat = autocorr_frequency(x, FS, 70.0, 700.0)
    assert f_hat is not None
    assert abs(cents_between(f_hat, f_true)) < 2.0


@pytest.mark.parametrize(("name", "f"), list(GUITAR_OPEN_STRINGS_HZ.items()))
def test_pure_tone_is_within_seven_cents(name: str, f: float) -> None:
    x = harmonic_tone(f, [1.0], 1.0, FS)[:N]
    f_hat = autocorr_frequency(x, FS, 70.0, 700.0)
    assert f_hat is not None
    assert abs(cents_between(f_hat, f)) < 7.0


def test_dc_offset_is_ignored() -> None:
    x = harmonic_tone(196.0, WEAK, 1.0, FS)[:N] + 5.0
    f_hat = autocorr_frequency(x, FS, 70.0, 700.0)
    assert f_hat is not None
    assert abs(cents_between(f_hat, 196.0)) < 2.0


def test_silence_and_white_noise_have_no_pitch() -> None:
    assert autocorr_frequency(np.zeros(N), FS, 70.0, 700.0) is None
    noise = np.random.default_rng(3).standard_normal(N)
    assert autocorr_frequency(noise, FS, 70.0, 700.0) is None


def test_finite_record_shrinks_the_peak_so_a_high_threshold_rejects_the_low_e() -> None:
    x = harmonic_tone(82.41, WEAK, 1.0, FS)[:N]
    assert autocorr_frequency(x, FS, 70.0, 700.0, threshold=0.5) is not None
    assert autocorr_frequency(x, FS, 70.0, 700.0, threshold=0.95) is None


def test_invalid_range_is_rejected() -> None:
    with pytest.raises(ValueError):
        autocorr_frequency(np.zeros(N), FS, 500.0, 100.0)
