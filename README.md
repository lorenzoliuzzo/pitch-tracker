# pitch-tracker

A guitar tuner built from first principles: it takes a block of audio samples and returns one
frequency, accurate to a few cents, plus the nearest note and how many cents off it is.

It is the hands-on project of a digital signal processing course. The theory lives in the course
notes (see [Theory](#theory)); this repository is the code that puts it to work, and it is written
to be read.

**Status:** the estimators, the note arithmetic, the recording tool and the live tuner are written
and tested on synthetic signals. The live tuner has been started against this machine's default
input, but not yet checked by ear against a real guitar.

## Quickstart

You need Python 3.12 and [uv](https://docs.astral.sh/uv/). A microphone is only needed to record.

```bash
uv sync                          # create the environment and install dependencies
uv run pytest                    # 147 tests, no microphone needed
uv run pitch-demo                # compare the three estimators on synthetic strings
uv run pitch-record E2           # record 4 s from the microphone, plot its spectrogram
uv run pitch-tuner               # the live tuner: pluck a string, read the note and cents
```

`pitch-record` saves the samples to `recordings/E2.npy` and a waveform plus spectrogram to
`plots/E2.png`. Record all six open strings and look at the plots before trusting any estimator:
on the low strings the strongest line is often not the fundamental.

## The live tuner

```
$ uv run pitch-tuner
Listening with the autocorr estimator. Pluck a string. Ctrl+C to stop.
A2    -23.8c  ----------o---------|--------------------   110.00 Hz
```

The needle is centred on the note: `o` to the left means flat, to the right sharp, `O` in the
middle means in tune (the scale is plus or minus 50 cents; the frequency shown is the note's target, not what you played). The tuner keeps the latest 4096
samples, estimates every 80 ms, and shows the median of the last five estimates so one bad frame
does not move the needle. Input quieter than an RMS of 0.01 is ignored, so room noise does not
produce a confident wrong note.

Options: `--method peak|hps|autocorr` (default `autocorr`), `--list-devices` to see the inputs,
and `--device N` to pick one. If the display stays on "listening" while you play, check that the
microphone is not muted and try another device.

## Using it

Every estimator is a pure function: samples in, frequency in hertz out. There is no hidden state
and no audio device involved.

```python
from pitch_tracker.autocorr import autocorr_frequency
from pitch_tracker.hps import hps_frequency
from pitch_tracker.notes import nearest_note
from pitch_tracker.spectrum import peak_frequency
from pitch_tracker.synth import harmonic_tone

fs = 44100.0
x = harmonic_tone(82.41, [0.2, 1.0, 0.8], duration=1.0, fs=fs)[:4096]  # a low E, weak fundamental

peak_frequency(x, fs, 70, 700)  # 164.99 Hz  (an octave too high)
hps_frequency(x, fs, 70, 700)  #  82.5  Hz
autocorr_frequency(x, fs, 70, 700)  #  82.4  Hz   (None if no pitch is found)

nearest_note(82.5)  # Note(name='E2', frequency=82.41, cents=+1.96)
```

All three take the same arguments: the samples, the sampling rate, and the range of frequencies
to search (`f_min`, `f_max`). A guitar lives between about 70 Hz and 700 Hz for the fundamental.

## How it works

A plucked string vibrates at a fundamental frequency `f0` and at its harmonics `2 f0, 3 f0, ...`.
The note you hear is `f0`, but the loudest line in the spectrum can be a harmonic. Getting `f0`
right, and getting it to within a few cents, are two separate problems.

**Accuracy: a DFT bin is far too coarse.** With 4096 samples at 44.1 kHz the bins are 10.8 Hz
apart. At the low E (82.41 Hz) that is about 212 cents, two semitones, while a tuner wants a few.
Waiting longer for a finer bin makes the display slow, so `spectrum.py` interpolates instead: it
fits a parabola through the log magnitudes of the peak bin and its two neighbours, and reads the
peak off the vertex. A Hann window is applied first so the peak has a smooth, nearly parabolic top.

**Correctness: three ways to find `f0`.**

| Estimator | Idea | Strength | Weakness |
|---|---|---|---|
| `peak_frequency` (`spectrum.py`) | Largest peak in the windowed spectrum, then interpolate | Simple, accurate on a pure tone | Reads the strongest harmonic, often an octave too high |
| `hps_frequency` (`hps.py`) | Harmonic product spectrum: multiply the spectrum by copies of itself squeezed by 2 and 3, so only the true `f0` has energy at all of `k, 2k, 3k` | Finds a weak fundamental | Needs harmonics: on a pure tone it can land an octave or more too low |
| `autocorr_frequency` (`autocorr.py`) | The signal repeats every period, so it matches itself shifted by `fs / f0` samples; take the first peak of the autocorrelation | Finds a weak fundamental, no harmonics needed | Broad peak on a pure low tone, so a little less exact there |

One detail worth knowing: `hps_frequency` uses the product only to choose the *bin*. Interpolating
on the product itself is biased (a product of peaks is not a parabola; it gave a 35 cent error),
so it refines on the strongest harmonic of the plain spectrum and divides by that harmonic's index.

## What to expect

`uv run pitch-demo` prints the error in cents for each estimator on the six open strings, detuned
by +20 cents, with 4096 samples at 44.1 kHz.

A string with a weak fundamental (harmonic amplitudes 0.2, 1.0, 0.8):

| String | `peak` | `hps` | `autocorr` |
|---|---:|---:|---:|
| E2 | +1200.3 | +0.3 | +0.3 |
| A2 | +1198.7 | -1.3 | +0.2 |
| D3 | +1199.3 | -0.7 | +0.0 |
| G3 | +1199.4 | -0.6 | +0.0 |
| B3 | +1200.4 | +0.4 | +0.0 |
| E4 | +1199.9 | -0.1 | +0.1 |

`peak` is exactly one octave (1200 cents) off on every string, and the other two are within about
a cent. On a *pure* sine the picture reverses: `peak` is within 4 cents, `autocorr` within 7, and
`hps` fails on most strings because there are no harmonics to multiply. A real guitar string
sounds like the first table, which is why a tuner needs more than the largest peak.

These are clean synthetic signals. Real recordings add noise, a slowly decaying amplitude and a
pluck transient, so measure your own strings (`pitch-record`) before quoting an accuracy.

## Repository layout

```
src/pitch_tracker/
  notes.py      cents_between, nearest_note, the six open-string frequencies
  spectrum.py   windowed DFT, parabolic peak interpolation, peak_frequency
  hps.py        harmonic product spectrum, hps_frequency
  autocorr.py   FFT autocorrelation, autocorr_frequency
  synth.py      harmonic_tone: test signals with chosen harmonics, noise and decay
  demo.py       the accuracy table above (pitch-demo)
  estimators.py one place that turns a method name into an estimator function
  record.py     microphone capture and spectrogram plot (pitch-record)
  live.py       the live tuner: ring buffer, gate, smoothing, needle (pitch-tuner)
tests/          one file per module; synthetic signals only
```

`record.py` and the `run` function of `live.py` are the only code that touches the microphone.
The tuner's logic (`Tuner`, `needle`, `format_reading`) and every estimator are pure, which is why
the test suite needs no audio hardware.

## Development

```bash
uv run pytest            # the tests
uv run ruff check .      # lint
uv run ruff format .     # format
```

The tests assert the behaviour above rather than only checking that code runs: the octave error of
`peak` on a weak fundamental, the failure of `hps` on a pure tone, the shrinking of the
autocorrelation peak on a finite record, and a stress test that guards an out-of-bounds read at the
top of the spectrum.

## Theory

The notes behind each step are in the course site. In the course "Digital Signal Processing":

- *Spectrum Representation and Fourier Series*: harmonics and line spectra.
- *The Discrete Fourier Transform* and *The FFT and Windowing*: bins, the picket-fence effect,
  leakage and the Hann window.
- *Pitch Estimation from the Spectrum*: cents, parabolic interpolation, the harmonic product
  spectrum and the autocorrelation, with the resolution budget for the low strings.

The site also has a step-by-step guide, *How to build a guitar tuner*, that walks through this
repository.

## Roadmap

- Check the live tuner by ear against a real guitar and a reference app, on all six strings.
- Decimation: lowering the sampling rate to get finer bins for the same record length.
- Measure the three estimators on real recordings of all six strings and record the errors.
