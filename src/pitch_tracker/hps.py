import numpy as np

from pitch_tracker.spectrum import bin_range, interpolate_peak, windowed_magnitude


def harmonic_product_spectrum(magnitude: np.ndarray, factors: int) -> np.ndarray:
    n = magnitude.size // factors
    product = np.ones(n)
    for r in range(1, factors + 1):
        product *= magnitude[::r][:n]
    return product


def hps_frequency(x: np.ndarray, fs: float, f_min: float, f_max: float, factors: int = 3) -> float:
    if factors < 2:
        raise ValueError("the product needs at least two factors")
    magnitude = windowed_magnitude(x)
    k_lo, k_hi = bin_range(x.size, fs, f_min, f_max)
    product = harmonic_product_spectrum(magnitude, factors)
    k_hi = min(k_hi, product.size - 1)
    k0 = k_lo + int(np.argmax(product[k_lo : k_hi + 1]))

    # The product only chooses the bin. Interpolating on it is biased (a product of
    # peaks is not a parabola), so refine on the strongest harmonic of the plain spectrum.
    harmonic = max(range(1, factors + 1), key=lambda r: magnitude[r * k0])
    centre = harmonic * k0
    k = centre - 1 + int(np.argmax(magnitude[centre - 1 : centre + 2]))
    k = min(k, magnitude.size - 2)
    log_magnitude = np.log(magnitude + 1e-12)
    return interpolate_peak(log_magnitude, k) * fs / x.size / harmonic
