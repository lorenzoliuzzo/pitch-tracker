from pitch_tracker.estimators import METHODS, make_estimator
from pitch_tracker.notes import GUITAR_OPEN_STRINGS_HZ, cents_between
from pitch_tracker.synth import harmonic_tone

FS = 44100.0
N = 4096

ESTIMATORS = {method: make_estimator(method, FS) for method in METHODS}

# Amplitudes of the first harmonics. The second is the strongest, as on a low plucked string.
WEAK_FUNDAMENTAL = [0.2, 1.0, 0.8]


def error_table(
    amplitudes: list[float], detune_cents: float = 20.0, noise: float = 0.0
) -> dict[str, dict[str, float | None]]:
    table: dict[str, dict[str, float | None]] = {}
    for name, f in GUITAR_OPEN_STRINGS_HZ.items():
        f_true = f * 2.0 ** (detune_cents / 1200.0)
        x = harmonic_tone(f_true, amplitudes, 1.0, FS, noise=noise)[:N]
        row: dict[str, float | None] = {}
        for label, estimate in ESTIMATORS.items():
            f_hat = estimate(x)
            row[label] = None if f_hat is None else cents_between(f_hat, f_true)
        table[name] = row
    return table


def format_table(table: dict[str, dict[str, float | None]]) -> str:
    labels = list(ESTIMATORS)
    lines = ["string " + " ".join(f"{label:>10}" for label in labels)]
    for name, row in table.items():
        cells = ["      none" if row[k] is None else f"{row[k]:+10.1f}" for k in labels]
        lines.append(f"{name:<6} " + " ".join(cells))
    return "\n".join(lines)


def main() -> None:
    print("Error in cents, strings detuned +20 cents, N=4096 at 44.1 kHz\n")
    print("Weak fundamental (harmonics 0.2, 1.0, 0.8):")
    print(format_table(error_table(WEAK_FUNDAMENTAL)))
    print("\nPure tone (no harmonics):")
    print(format_table(error_table([1.0])))


if __name__ == "__main__":
    main()
