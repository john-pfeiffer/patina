"""Synthesised test signals for the analysis tests (no fixtures on disk).

Not a test module in itself; the `test_analysis_*` modules import it."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt

from patine.analysis.decode import ANALYSIS_SR, DecodedAudio, resample_mono
from patine.audio import to_mono
from patine.theory import QUALITY_INTERVALS

ChordSpec = tuple[int, str]  # (root pitch class, quality)

PROGRESSION: tuple[ChordSpec, ...] = (
    (0, "maj"), (9, "min"), (5, "maj"), (7, "maj"),
    (0, "7"), (2, "m7"), (0, "maj7"), (7, "sus4"),
)


def midi_to_hz(midi: float, detune_cents: float = 0.0) -> float:
    return 440.0 * 2.0 ** ((midi - 69) / 12.0) * 2.0 ** (detune_cents / 1200.0)


def tone(f0: float, t: np.ndarray, *, n_harm: int = 5, amp: float = 1.0) -> np.ndarray:
    out = np.zeros_like(t)
    for h in range(1, n_harm + 1):
        out += (amp / h) * np.sin(2.0 * np.pi * f0 * h * t)
    return out


def chord_signal(
    chords: tuple[ChordSpec, ...] = PROGRESSION,
    *,
    sr: int = 44100,
    bpm: float = 120.0,
    beats_per_chord: float = 4.0,
    bass: bool = True,
    bass_interval: int = 0,
    clicks: bool = True,
    melody: bool = False,
    detune_cents: float = 0.0,
    noise: float = 0.002,
    stereo: bool = True,
    chord_octave: int = 3,
    seed: int = 0,
) -> tuple[np.ndarray, float]:
    """Sustained chords (bass an octave below the triad, on `bass_interval`
    semitones above the root so inversions can be built; five harmonics each),
    a short click on every beat, optional 8th-note melody with non-chord tones
    in the upper register, mild noise. Returns ((channels, n) float32, duration)."""
    rng = np.random.default_rng(seed)
    beat = 60.0 / bpm
    chord_s = beat * beats_per_chord
    total = chord_s * len(chords)
    n = int(round(total * sr))
    t = np.arange(n) / sr
    y = np.zeros(n)
    base_midi = 12 * (chord_octave + 1)
    for ci, (root, quality) in enumerate(chords):
        a = int(round(ci * chord_s * sr))
        b = min(n, int(round((ci + 1) * chord_s * sr)))
        seg_t = t[a:b] - t[a]
        env = np.minimum(1.0, seg_t / 0.01) * np.minimum(1.0, (seg_t[-1] - seg_t) / 0.01 + 1e-3)
        seg = np.zeros_like(seg_t)
        for iv in QUALITY_INTERVALS[quality]:
            seg += tone(midi_to_hz(base_midi + root + iv, detune_cents), seg_t, amp=0.18)
        if bass:
            seg += tone(midi_to_hz(base_midi - 12 + root + bass_interval, detune_cents), seg_t, amp=0.25, n_harm=4)
        if melody:
            # 8th notes: 9th, 11th, 13th and chromatic neighbours above the chord
            pattern = [14, 17, 21, 16, 18, 14, 13, 19]
            step = beat / 2.0
            k = 0
            pos = 0.0
            while pos < seg_t[-1]:
                i0 = int(pos * sr)
                i1 = min(seg.size, int((pos + step) * sr))
                mt = seg_t[i0:i1] - seg_t[i0]
                menv = np.minimum(1.0, mt / 0.005) * np.exp(-mt / (step * 0.8))
                midi = base_midi + 24 + root + pattern[k % len(pattern)]
                seg[i0:i1] += tone(midi_to_hz(midi, detune_cents), mt, amp=0.16, n_harm=3) * menv
                k += 1
                pos += step
        y[a:b] += seg * env
    if clicks:
        n_beats = int(round(total / beat))
        click = np.exp(-np.arange(300) / 50.0) * rng.standard_normal(300)
        for k in range(n_beats):
            i = int(round(k * beat * sr))
            m = min(300, n - i)
            if m > 0:
                y[i : i + m] += 0.5 * click[:m]
    y += noise * rng.standard_normal(n)
    y /= max(1e-9, np.abs(y).max()) * 1.25
    if stereo:
        data = np.stack([y, 0.92 * y + noise * rng.standard_normal(n)])
    else:
        data = y[None, :]
    return data.astype(np.float32), total


def decoded(y: np.ndarray, sr: int) -> DecodedAudio:
    """Wrap an array as the pipeline's decoded-audio input (no file needed)."""
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 1:
        y = y[None, :]
    mono = to_mono(y)
    return DecodedAudio(y=y, sr=sr, mono=mono, mono_a=resample_mono(mono, sr, ANALYSIS_SR), sr_a=ANALYSIS_SR)


def sine(freq: float, duration: float, sr: int, amp: float = 0.1) -> np.ndarray:
    t = np.arange(int(duration * sr)) / sr
    return (amp * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)


def noise(duration: float, sr: int, amp: float = 0.1, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (amp * rng.standard_normal(int(duration * sr))).astype(np.float32)


def filtered_noise(duration: float, sr: int, cutoff: float, kind: str, amp: float = 0.1, seed: int = 0) -> np.ndarray:
    sos = butter(8, cutoff, btype=kind, fs=sr, output="sos")
    return sosfilt(sos, noise(duration, sr, amp, seed)).astype(np.float32)


def clicks(duration: float, sr: int, period_s: float, *, decay_s: float = 0.03, amp: float = 0.5) -> tuple[np.ndarray, list[float]]:
    """Exponentially decaying noise bursts every `period_s`. Returns (signal, onset times)."""
    rng = np.random.default_rng(1)
    n = int(duration * sr)
    y = np.zeros(n, dtype=np.float32)
    onsets = []
    k = int(decay_s * sr * 6)
    pos = 0.0
    while pos < duration:
        i = int(pos * sr)
        m = min(k, n - i)
        if m <= 0:
            break
        burst = np.exp(-np.arange(m) / (decay_s * sr)) * rng.standard_normal(m)
        y[i : i + m] += amp * burst.astype(np.float32)
        onsets.append(pos)
        pos += period_s
    return y, onsets
