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


# --- melodic, sparse and noisy material (recall tests) -----------------------------------

def plucked_note(midi: float, t: np.ndarray, *, n_harm: int = 5, decay_s: float = 0.8, amp: float = 0.3) -> np.ndarray:
    """A plucked tone: 5 ms attack, exponential decay, `n_harm` harmonics at 1/h."""
    env = np.minimum(1.0, t / 0.005) * np.exp(-t / decay_s)
    return tone(midi_to_hz(midi), t, n_harm=n_harm, amp=amp) * env


def melody_signal(
    notes_midi: list[int],
    *,
    sr: int = 22050,
    bpm: float = 100.0,
    note_beats: float = 1.0,
    decay_s: float = 0.8,
    amp: float = 0.3,
) -> tuple[np.ndarray, float]:
    """Monophonic plucked melody, one note per `note_beats`; (mono float32, duration)."""
    beat = 60.0 / bpm
    note_s = beat * note_beats
    total = note_s * len(notes_midi)
    n = int(round(total * sr))
    y = np.zeros(n)
    for i, midi in enumerate(notes_midi):
        a = int(round(i * note_s * sr))
        b = min(n, int(round((i + 1) * note_s * sr)) + int(decay_s * sr))
        t = np.arange(b - a) / sr
        y[a:b] += plucked_note(midi, t, decay_s=decay_s, amp=amp)
    y /= max(1e-9, np.abs(y).max()) * 1.25
    return y.astype(np.float32), total


def arpeggio_signal(
    root: int,
    quality: str,
    *,
    sr: int = 22050,
    bpm: float = 120.0,
    bars: int = 2,
    division: int = 8,
    octave: int = 4,
    ring_s: float = 0.6,
    amp: float = 0.3,
) -> tuple[np.ndarray, float]:
    """Root-3rd-5th-octave cycle of plucked notes at `division`ths; `ring_s` is
    how long each note rings (0.15 is a dry pluck, 0.6 overlaps the next notes)."""
    beat = 60.0 / bpm
    step = beat * 4.0 / division
    total = beat * 4.0 * bars
    n = int(round(total * sr))
    y = np.zeros(n)
    ivs = list(QUALITY_INTERVALS[quality][:3]) + [12]
    base = 12 * (octave + 1) + root
    k = 0
    pos = 0.0
    while pos < total - 1e-9:
        a = int(round(pos * sr))
        b = min(n, a + int((step + ring_s) * sr))
        t = np.arange(b - a) / sr
        y[a:b] += plucked_note(base + ivs[k % len(ivs)], t, decay_s=ring_s / 2.5, amp=amp)
        k += 1
        pos += step
    y /= max(1e-9, np.abs(y).max()) * 1.25
    return y.astype(np.float32), total


def oneshot_note(midi: int = 55, *, sr: int = 22050, duration: float = 1.5, decay_s: float = 0.5) -> np.ndarray:
    """A single plucked note (G3 by default) that decays into silence."""
    t = np.arange(int(duration * sr)) / sr
    y = plucked_note(midi, t, decay_s=decay_s, amp=0.5)
    return (y / max(1e-9, np.abs(y).max()) * 0.8).astype(np.float32)


def chord_hit(root: int, quality: str, *, sr: int = 22050, duration: float = 0.8, decay_s: float = 0.4, octave: int = 3) -> np.ndarray:
    """One short chord stab with a bass note an octave below, decaying into silence."""
    t = np.arange(int(duration * sr)) / sr
    base = 12 * (octave + 1) + root
    y = np.zeros_like(t)
    for iv in QUALITY_INTERVALS[quality]:
        y += plucked_note(base + iv, t, decay_s=decay_s, amp=0.2)
    y += plucked_note(base - 12, t, n_harm=4, decay_s=decay_s, amp=0.25)
    return (y / max(1e-9, np.abs(y).max()) * 0.8).astype(np.float32)


def drum_loop(duration: float, sr: int = 22050, *, bpm: float = 120.0, seed: int = 3) -> np.ndarray:
    """Kick (60→45 Hz sweep) on every beat, snare (band-passed noise) on 2 and 4,
    hats (high-passed noise) on 8ths."""
    rng = np.random.default_rng(seed)
    beat = 60.0 / bpm
    n = int(duration * sr)
    y = np.zeros(n)
    kick_t = np.arange(int(0.12 * sr)) / sr
    kick = np.sin(2 * np.pi * (60.0 * kick_t - 0.5 * 125.0 * kick_t**2)) * np.exp(-kick_t / 0.05)
    snare_sos = butter(4, [150.0, 4000.0], btype="band", fs=sr, output="sos")
    hat_sos = butter(4, 6000.0, btype="high", fs=sr, output="sos")
    k = 0
    pos = 0.0
    while pos < duration:
        i = int(pos * sr)
        m = min(kick.size, n - i)
        if m > 0:
            y[i:i + m] += 0.9 * kick[:m]
        if k % 4 in (1, 3):
            sn = sosfilt(snare_sos, rng.standard_normal(int(0.15 * sr))) * np.exp(-np.arange(int(0.15 * sr)) / (0.04 * sr))
            m = min(sn.size, n - i)
            if m > 0:
                y[i:i + m] += 0.7 * sn[:m]
        for half in (0.0, 0.5):
            j = int((pos + half * beat) * sr)
            hat = sosfilt(hat_sos, rng.standard_normal(int(0.03 * sr))) * np.exp(-np.arange(int(0.03 * sr)) / (0.008 * sr))
            m = min(hat.size, n - j)
            if m > 0:
                y[j:j + m] += 0.35 * hat[:m]
        k += 1
        pos += beat
    return (y / max(1e-9, np.abs(y).max())).astype(np.float32)


def pad_under_drums(
    chords: tuple[ChordSpec, ...],
    *,
    sr: int = 22050,
    bpm: float = 120.0,
    beats_per_chord: float = 4.0,
    drum_db: float = 6.0,
) -> tuple[np.ndarray, float]:
    """Sustained chords with a drum loop `drum_db` louder (RMS) on top; mono."""
    pad, total = chord_signal(chords, sr=sr, bpm=bpm, beats_per_chord=beats_per_chord, clicks=False, stereo=False)
    pad = pad[0].astype(np.float64)
    drums = drum_loop(total, sr, bpm=bpm).astype(np.float64)[: pad.size]
    drums = np.pad(drums, (0, pad.size - drums.size))
    pad_rms = np.sqrt(np.mean(pad**2)) + 1e-9
    drums_rms = np.sqrt(np.mean(drums**2)) + 1e-9
    drums *= pad_rms / drums_rms * 10.0 ** (drum_db / 20.0)
    y = pad + drums
    y /= max(1e-9, np.abs(y).max()) * 1.25
    return y.astype(np.float32), total


def noise_loop(duration: float = 4.0, sr: int = 22050, *, lowpass_hz: float | None = None) -> np.ndarray:
    """White (or low-passed) noise with clicks every half second: no chord here."""
    base = noise(duration, sr, 0.2) if lowpass_hz is None else filtered_noise(duration, sr, lowpass_hz, "low", 0.2)
    hits, _ = clicks(duration, sr, 0.5)
    y = base + hits
    return (y / max(1e-9, np.abs(y).max()) * 0.9).astype(np.float32)
