import numpy as np
import pytest

from pitch_tracker.hps import harmonic_product_spectrum, hps_frequency
from pitch_tracker.notes import GUITAR_OPEN_STRINGS_HZ, cents_between
from pitch_tracker.synth import harmonic_tone

FS = 44100.0
N = 4096
WEAK = [0.2, 1.0, 0.8]


def test_product_spectrum_matches_the_worked_example() -> None:
    magnitude = np.zeros(65)
    magnitude[[10, 20, 30, 40]] = [0.2, 1.0, 0.8, 0.3]
    product = harmonic_product_spectrum(magnitude, 3)
    assert product[10] == pytest.approx(0.16)
    assert product[20] == 0.0
    assert int(np.argmax(product)) == 10


@pytest.mark.parametrize(("name", "f"), list(GUITAR_OPEN_STRINGS_HZ.items()))
@pytest.mark.parametrize("offset_cents", [-30.0, 0.0, 30.0])
@pytest.mark.parametrize("noise", [0.0, 0.1])
def test_weak_fundamental_is_found_within_three_cents(
    name: str, f: float, offset_cents: float, noise: float
) -> None:
    f_true = f * 2.0 ** (offset_cents / 1200.0)
    x = harmonic_tone(f_true, WEAK, 1.0, FS, noise=noise)[:N]
    assert abs(cents_between(hps_frequency(x, FS, 70.0, 700.0), f_true)) < 3.0


def test_strong_fundamental_is_found_too() -> None:
    f_true = 110.0
    x = harmonic_tone(f_true, [1.0, 0.5, 0.3], 1.0, FS)[:N]
    assert abs(cents_between(hps_frequency(x, FS, 70.0, 700.0), f_true)) < 3.0


def test_a_pure_tone_has_no_harmonics_to_multiply() -> None:
    x = harmonic_tone(146.83, [1.0], 1.0, FS)[:N]
    f_hat = hps_frequency(x, FS, 70.0, 700.0)
    assert abs(cents_between(f_hat, 146.83)) > 50.0


def test_one_factor_is_rejected() -> None:
    with pytest.raises(ValueError):
        hps_frequency(np.zeros(N), FS, 70.0, 700.0, factors=1)


@pytest.mark.parametrize("factors", [2, 3, 4])
def test_random_records_up_to_nyquist_never_index_out_of_bounds(factors: int) -> None:
    rng = np.random.default_rng(4)
    for _ in range(200):
        x = rng.standard_normal(254)
        f_hat = hps_frequency(x, FS, 100.0, FS / 2.0, factors=factors)
        assert 0.0 < f_hat <= FS / 2.0
