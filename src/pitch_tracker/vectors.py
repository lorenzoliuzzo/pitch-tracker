import json
import sys
from pathlib import Path
from typing import Any

from pitch_tracker.estimators import F_MAX, F_MIN, METHODS, make_estimator
from pitch_tracker.notes import GUITAR_OPEN_STRINGS_HZ, cents_between, nearest_note
from pitch_tracker.synth import harmonic_tone

FS = 44100.0
N = 4096
PROFILES = {"pure": [1.0], "weak-fundamental": [0.2, 1.0, 0.8]}
OFFSETS_CENTS = (-30.0, 0.0, 30.0)
NOTE_FREQUENCIES_HZ = (20.0, 82.41, 108.5, 110.0, 196.0, 246.94, 329.63, 440.0, 445.0, 1000.0)
DEFAULT_PATH = Path("tests/data/vectors.json")


def build_vectors() -> dict[str, Any]:
    tones = []
    cases = [
        (name, f, off, prof)
        for prof in PROFILES
        for name, f in GUITAR_OPEN_STRINGS_HZ.items()
        for off in OFFSETS_CENTS
    ]
    cases.append(("silence", 110.0, 0.0, "silence"))
    for name, f, offset, profile in cases:
        f0 = f * 2.0 ** (offset / 1200.0)
        amplitudes = [0.0] if profile == "silence" else PROFILES[profile]
        x = harmonic_tone(f0, amplitudes, 1.0, FS)[:N]
        expected = {m: make_estimator(m, FS)(x) for m in METHODS}
        tones.append(
            {
                "name": f"{name} {offset:+.0f}c {profile}",
                "f0": f0,
                "amplitudes": amplitudes,
                "expected": expected,
            }
        )
    notes = []
    for f in NOTE_FREQUENCIES_HZ:
        note = nearest_note(f)
        notes.append(
            {
                "f": f,
                "name": note.name,
                "frequency": note.frequency,
                "cents": note.cents,
                "cents_to_a4": cents_between(f, 440.0),
            }
        )
    return {
        "version": 1,
        "fs": FS,
        "n": N,
        "f_min": F_MIN,
        "f_max": F_MAX,
        "tones": tones,
        "notes": notes,
    }


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(build_vectors(), indent=1) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
