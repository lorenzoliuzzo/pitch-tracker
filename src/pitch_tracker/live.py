import argparse
import queue
from collections import deque

import numpy as np
import sounddevice as sd

from pitch_tracker.estimators import METHODS, make_estimator
from pitch_tracker.notes import Note, nearest_note

FS = 44100
WINDOW = 4096
HOP_SECONDS = 0.08
GATE_RMS = 0.01
SMOOTHING = 5


class Tuner:
    def __init__(
        self,
        method: str = "autocorr",
        fs: float = FS,
        window: int = WINDOW,
        gate_rms: float = GATE_RMS,
        smoothing: int = SMOOTHING,
    ) -> None:
        self.estimate_frequency = make_estimator(method, fs)
        self.buffer = np.zeros(window)
        self.filled = 0
        self.gate_rms = gate_rms
        self.recent: deque[float] = deque(maxlen=smoothing)

    def push(self, samples: np.ndarray) -> None:
        n = min(samples.size, self.buffer.size)
        self.buffer = np.roll(self.buffer, -n)
        self.buffer[-n:] = samples[-n:]
        self.filled = min(self.filled + n, self.buffer.size)

    def reading(self) -> Note | None:
        if self.filled < self.buffer.size:
            return None
        if float(np.sqrt(np.mean(self.buffer**2))) < self.gate_rms:
            self.recent.clear()
            return None
        frequency = self.estimate_frequency(self.buffer)
        if frequency is None:
            return None
        self.recent.append(frequency)
        return nearest_note(float(np.median(self.recent)))


def needle(cents: float | None, half_width: int = 20) -> str:
    cells = ["-"] * (2 * half_width + 1)
    cells[half_width] = "|"
    if cents is not None:
        position = round(max(-50.0, min(50.0, cents)) / 50.0 * half_width)
        cells[half_width + position] = "O" if position == 0 else "o"
    return "".join(cells)


def format_reading(note: Note | None) -> str:
    if note is None:
        return f"{'--':<4}{'':>8}  {needle(None)}  listening"
    return f"{note.name:<4}{note.cents:+7.1f}c  {needle(note.cents)}  {note.frequency:7.2f} Hz"


def run(method: str, device: int | str | None) -> None:
    tuner = Tuner(method)
    blocks: queue.Queue[np.ndarray] = queue.Queue()

    def callback(indata: np.ndarray, frames: int, time: object, status: sd.CallbackFlags) -> None:
        blocks.put(indata[:, 0].astype(np.float64))

    print(f"Listening with the {method} estimator. Pluck a string. Ctrl+C to stop.")
    with sd.InputStream(
        samplerate=FS, channels=1, dtype="float32", device=device, callback=callback
    ):
        try:
            while True:
                try:
                    tuner.push(blocks.get(timeout=HOP_SECONDS))
                except queue.Empty:
                    pass
                while not blocks.empty():
                    tuner.push(blocks.get_nowait())
                print("\r" + format_reading(tuner.reading()) + "   ", end="", flush=True)
        except KeyboardInterrupt:
            print()


def main() -> None:
    parser = argparse.ArgumentParser(description="A live guitar tuner")
    parser.add_argument("--method", choices=METHODS, default="autocorr")
    parser.add_argument("--device", help="input device index or name (see --list-devices)")
    parser.add_argument("--list-devices", action="store_true")
    args = parser.parse_args()
    if args.list_devices:
        print(sd.query_devices())
        return
    device = int(args.device) if args.device and args.device.isdigit() else args.device
    run(args.method, device)


if __name__ == "__main__":
    main()
