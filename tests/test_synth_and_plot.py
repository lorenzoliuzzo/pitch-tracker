from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

from pitch_tracker.record import plot_waveform_and_spectrogram
from pitch_tracker.synth import harmonic_tone


def test_tone_has_requested_length_and_peaks_at_the_fundamental() -> None:
    fs = 8000.0
    x = harmonic_tone(200.0, [1.0], duration=1.0, fs=fs)
    assert x.size == 8000
    spectrum = np.abs(np.fft.rfft(x))
    assert np.argmax(spectrum) * fs / x.size == 200.0


def test_weak_fundamental_has_a_stronger_second_harmonic() -> None:
    fs = 8000.0
    x = harmonic_tone(100.0, [0.2, 1.0, 0.8], duration=1.0, fs=fs)
    spectrum = np.abs(np.fft.rfft(x))
    assert np.argmax(spectrum) * fs / x.size == 200.0


def test_noise_is_reproducible() -> None:
    a = harmonic_tone(100.0, [1.0], 0.1, 8000.0, noise=0.1, seed=3)
    b = harmonic_tone(100.0, [1.0], 0.1, 8000.0, noise=0.1, seed=3)
    assert np.array_equal(a, b)


def test_plot_writes_a_file(tmp_path: Path) -> None:
    x = harmonic_tone(110.0, [1.0, 0.5], duration=1.0, fs=8000.0)
    out = tmp_path / "plots" / "a2.png"
    plot_waveform_and_spectrogram(x, 8000.0, 1024, out)
    assert out.stat().st_size > 0
