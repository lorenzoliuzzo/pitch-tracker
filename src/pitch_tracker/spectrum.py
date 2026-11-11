import numpy as np
from scipy import signal


def windowed_magnitude(x: np.ndarray) -> np.ndarray:
    window = signal.get_window("hann", x.size)
    return np.abs(np.fft.rfft(x * window))


def check_range(fs: float, f_min: float, f_max: float) -> None:
    if not 0.0 < f_min < f_max <= fs / 2.0:
        raise ValueError(f"need 0 < f_min < f_max <= fs/2, got {f_min}, {f_max}, fs={fs}")


def bin_range(n_samples: int, fs: float, f_min: float, f_max: float) -> tuple[int, int]:
    check_range(fs, f_min, f_max)
    k_lo = max(1, int(np.ceil(f_min * n_samples / fs)))
    k_hi = min(n_samples // 2 - 1, int(np.floor(f_max * n_samples / fs)))
    if k_hi < k_lo:
        raise ValueError("the record is too short to contain a bin in [f_min, f_max]")
    return k_lo, k_hi


def interpolate_peak(values: np.ndarray, k: int) -> float:
    a, b, c = values[k - 1], values[k], values[k + 1]
    denominator = a - 2.0 * b + c
    if denominator == 0.0:
        return float(k)
    return k + 0.5 * (a - c) / denominator


def peak_frequency(x: np.ndarray, fs: float, f_min: float, f_max: float) -> float:
    magnitude = windowed_magnitude(x)
    k_lo, k_hi = bin_range(x.size, fs, f_min, f_max)
    k = k_lo + int(np.argmax(magnitude[k_lo : k_hi + 1]))
    log_magnitude = np.log(magnitude + 1e-12)
    return interpolate_peak(log_magnitude, k) * fs / x.size
