import math
from dataclasses import dataclass

A4_HZ = 440.0
NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
GUITAR_OPEN_STRINGS_HZ = {
    "E2": 82.41,
    "A2": 110.00,
    "D3": 146.83,
    "G3": 196.00,
    "B3": 246.94,
    "E4": 329.63,
}


@dataclass(frozen=True)
class Note:
    name: str
    frequency: float
    cents: float


def cents_between(f: float, f_ref: float) -> float:
    return 1200.0 * math.log2(f / f_ref)


def nearest_note(f: float, a4: float = A4_HZ) -> Note:
    semitones_from_a4 = round(12.0 * math.log2(f / a4))
    midi = 69 + semitones_from_a4
    name = f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"
    frequency = a4 * 2.0 ** (semitones_from_a4 / 12.0)
    return Note(name=name, frequency=frequency, cents=cents_between(f, frequency))
