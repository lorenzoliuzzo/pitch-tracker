from collections.abc import Callable

import numpy as np

from pitch_tracker.autocorr import autocorr_frequency
from pitch_tracker.hps import hps_frequency
from pitch_tracker.spectrum import peak_frequency

F_MIN = 70.0
F_MAX = 700.0

Estimator = Callable[[np.ndarray], float | None]


def make_estimator(method: str, fs: float, f_min: float = F_MIN, f_max: float = F_MAX) -> Estimator:
    if method == "peak":
        return lambda x: peak_frequency(x, fs, f_min, f_max)
    if method == "hps":
        return lambda x: hps_frequency(x, fs, f_min, f_max)
    if method == "autocorr":
        return lambda x: autocorr_frequency(x, fs, f_min, f_max)
    raise ValueError(f"unknown method {method!r}; choose peak, hps or autocorr")


METHODS = ("peak", "hps", "autocorr")
