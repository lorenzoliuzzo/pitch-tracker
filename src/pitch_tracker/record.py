import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
from scipy import signal

DEFAULT_FS = 44100


def capture(seconds: float, fs: int = DEFAULT_FS) -> np.ndarray:
    audio = sd.rec(round(seconds * fs), samplerate=fs, channels=1, dtype="float32")
    sd.wait()
    return audio[:, 0].astype(np.float64)


def plot_waveform_and_spectrogram(x: np.ndarray, fs: float, window_length: int, out: Path) -> None:
    f, t, sxx = signal.spectrogram(
        x, fs=fs, window="hann", nperseg=window_length, noverlap=window_length // 2
    )
    fig, (ax_wave, ax_spec) = plt.subplots(2, 1, figsize=(9, 6))
    ax_wave.plot(np.arange(x.size) / fs, x, linewidth=0.5)
    ax_wave.set_xlabel("time (s)")
    ax_wave.set_ylabel("amplitude")
    ax_spec.pcolormesh(t, f, 10.0 * np.log10(sxx + 1e-12), shading="auto")
    ax_spec.set_ylim(0, 1500)
    ax_spec.set_xlabel("time (s)")
    ax_spec.set_ylabel("frequency (Hz)")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser(description="Record a string and plot its spectrogram")
    parser.add_argument("name", help="label for the recording, e.g. E2")
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--window", type=int, default=4096, help="samples per spectrogram frame")
    args = parser.parse_args()

    x = capture(args.seconds)
    Path("recordings").mkdir(exist_ok=True)
    np.save(Path("recordings") / f"{args.name}.npy", x)
    plot_waveform_and_spectrogram(x, DEFAULT_FS, args.window, Path("plots") / f"{args.name}.png")


if __name__ == "__main__":
    main()
