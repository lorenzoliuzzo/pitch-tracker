import numpy as np

from pitch_tracker.spectrum import check_range, interpolate_peak


def autocorrelation(x: np.ndarray) -> np.ndarray:
    n = x.size
    power = np.abs(np.fft.rfft(x, 2 * n)) ** 2
    return np.fft.irfft(power)[:n]


def autocorr_frequency(
    x: np.ndarray, fs: float, f_min: float, f_max: float, threshold: float = 0.5
) -> float | None:
    check_range(fs, f_min, f_max)
    x = x - x.mean()
    r = autocorrelation(x)
    if r[0] == 0.0:
        return None
    r = r / r[0]

    lag_lo = max(1, int(np.floor(fs / f_max)))
    lag_hi = min(r.size - 2, int(np.ceil(fs / f_min)))
    for lag in range(lag_lo, lag_hi + 1):
        is_peak = r[lag] >= r[lag - 1] and r[lag] > r[lag + 1]
        if is_peak and r[lag] >= threshold:
            return fs / interpolate_peak(r, lag)
    return None
