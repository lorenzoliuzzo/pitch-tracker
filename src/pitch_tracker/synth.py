import numpy as np


def harmonic_tone(
    f0: float,
    amplitudes: list[float],
    duration: float,
    fs: float,
    noise: float = 0.0,
    decay: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    t = np.arange(round(duration * fs)) / fs
    x = np.zeros_like(t)
    for k, a in enumerate(amplitudes, start=1):
        x += a * np.sin(2.0 * np.pi * k * f0 * t)
    x *= np.exp(-decay * t)
    if noise > 0.0:
        x += noise * np.random.default_rng(seed).standard_normal(t.size)
    return x
