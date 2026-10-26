"""Chroma features for chord, key and activity estimation (handoff §4.5–4.7).

One constant-Q transform (C1–B7, 36 bins per octave folded to semitones so
adjacent pitch classes barely leak into each other, hop 2048 at 22050 Hz, so
about 93 ms per frame) feeds three chroma views:

* ``weighted`` — **bass-weighted** chroma. The chord register (C3–B4) and the
  bass octaves count most and the top octaves little, so a melody line leaning
  on a 9th up top cannot flip the chord estimate (§4.5), while a bass on the
  third (C/E) still reads as C major rather than E minor.
* ``bass`` — the lowest octaves only (C1–B3), used to report ``bass_note``
  separately so inversions are visible.
* ``treble`` — the upper octaves only (C4–B7), used by the busyness score (§4.7).

CQT magnitudes are kept linear (`COMPRESSION` = 1; compressing them before
folding boosts leakage until every frame looks flat), but the folded chord
chroma (`weighted`, `treble`) is square-rooted (`CHROMA_COMPRESSION`): a bass
playing the root otherwise dominates the vector so much that a 7th or a
sus tone can no longer be told apart from a plain triad. `bass` and `energy`
stay linear. All arrays are float32.

Harmonic chroma (`HPSS`). Drums, clicks, reverb wash and noise put a broadband
floor under every pitch class, and a floor of a third of the chord's level
already makes a frame read as "no chord". The chord views (`weighted`,
`bass`, `energy`) are therefore folded from the *harmonic* part of the CQT:
a median filter along time keeps what sustains (a 3-frame kernel keeps
anything two frames long, so 8th notes up to about 160 bpm survive) and a
median filter along frequency removes what is spread across the spectrum.
`treble` and `cqt` stay on the full mix: busyness needs the melody's short
notes and the register centroid describes the whole sound. Playback and the
tonal measurements never see the separated signal."""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

ANALYSIS_SR = 22050
CHROMA_HOP = 2048               # 92.9 ms at 22050 Hz; divisible by 2**(N_OCTAVES-1)
N_OCTAVES = 7
BINS_PER_OCTAVE = 12                             # semitone bins kept per octave
CQT_BINS_PER_OCTAVE = 36                         # CQT resolution before folding
N_BINS = N_OCTAVES * BINS_PER_OCTAVE
FMIN_HZ = float(librosa.note_to_hz("C1"))      # 32.70 Hz
COMPRESSION = 1.0                                # CQT magnitude exponent (before folding)
CHROMA_COMPRESSION = 0.5                         # exponent on the folded chord chroma

# Per-octave weights, index 0 = C1–B3 ... 6 = C7–B7.
OCTAVE_WEIGHTS = np.array([0.5, 0.8, 1.0, 1.0, 0.6, 0.35, 0.2], dtype=np.float32)
BASS_WEIGHTS = np.array([1.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
TREBLE_WEIGHTS = np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0], dtype=np.float32)

_MIN_SIGNAL_S = 4.0  # pad shorter inputs so the low CQT octaves have room

# Harmonic–percussive separation of the CQT magnitude before folding.
HPSS = True
HPSS_TIME_KERNEL = 3      # frames (~280 ms): tones shorter than two frames are not harmonic
HPSS_FREQ_KERNEL = 31     # sub-bins (0.86 octave at 36/oct): broadband hits are percussive
HPSS_POWER = 2.0          # soft masks


@dataclass
class ChromaFeatures:
    """Frame-wise features over one stretch of audio. Frame `i` is centred at
    ``times[i]`` and covers ``[times[i] - hop_s/2, times[i] + hop_s/2)``."""

    sr: int
    hop: int
    times: np.ndarray        # (T,) frame centres, seconds
    cqt: np.ndarray          # (N_BINS, T) compressed magnitude
    freqs: np.ndarray        # (N_BINS,) bin centre frequencies, Hz
    weighted: np.ndarray     # (12, T) bass-weighted chroma
    bass: np.ndarray         # (12, T) low-register chroma
    treble: np.ndarray       # (12, T) upper-register chroma
    energy: np.ndarray       # (T,) sum of the weighted chroma
    duration: float          # seconds of real audio covered

    @property
    def hop_s(self) -> float:
        return float(self.hop) / float(self.sr)

    @property
    def n_frames(self) -> int:
        return int(self.times.size)

    def frame_start_s(self, i: int) -> float:
        return max(0.0, (i - 0.5) * self.hop_s)

    def frame_end_s(self, i: int) -> float:
        """End of frame `i` (exclusive), clipped to the audio."""
        return min(self.duration, (i + 0.5) * self.hop_s)

    def frame_bounds(self, start_s: float, end_s: float) -> tuple[int, int]:
        """Index range [a, b) of frames whose centre falls inside [start_s, end_s)."""
        a = int(np.searchsorted(self.times, start_s, side="left"))
        b = int(np.searchsorted(self.times, end_s, side="left"))
        return a, max(a, b)

    def slice_frames(self, a: int, b: int) -> ChromaFeatures:
        a = max(0, a)
        b = min(self.n_frames, b)
        return ChromaFeatures(
            sr=self.sr,
            hop=self.hop,
            times=self.times[a:b],
            cqt=self.cqt[:, a:b],
            freqs=self.freqs,
            weighted=self.weighted[:, a:b],
            bass=self.bass[:, a:b],
            treble=self.treble[:, a:b],
            energy=self.energy[a:b],
            duration=self.duration,
        )

    def slice_seconds(self, start_s: float, end_s: float) -> ChromaFeatures | None:
        """Frames inside [start_s, end_s); None when no frame centre falls inside."""
        a, b = self.frame_bounds(start_s, end_s)
        if b <= a:
            return None
        return self.slice_frames(a, b)

    def mean_weighted(self) -> np.ndarray:
        return _mean_cols(self.weighted)

    def mean_bass(self) -> np.ndarray:
        return _mean_cols(self.bass)


def _mean_cols(x: np.ndarray) -> np.ndarray:
    if x.shape[1] == 0:
        return np.zeros(x.shape[0], dtype=np.float32)
    return x.mean(axis=1).astype(np.float32)


def fold_octaves(cqt: np.ndarray, octave_weights: np.ndarray) -> np.ndarray:
    """Collapse an (N_BINS, T) CQT into (12, T) chroma with one weight per octave."""
    t = cqt.shape[1]
    stacked = cqt.reshape(N_OCTAVES, BINS_PER_OCTAVE, t)
    return np.einsum("o,opt->pt", octave_weights.astype(np.float32), stacked).astype(np.float32)


def compute_chroma(
    y: np.ndarray,
    sr: int = ANALYSIS_SR,
    *,
    tuning_cents: float = 0.0,
    hop: int = CHROMA_HOP,
    hpss: bool = HPSS,
) -> ChromaFeatures:
    """CQT-based chroma over mono `y` at `sr`, compensated for `tuning_cents` (§4.2)
    so a detuned file still lands on the right pitch classes. With `hpss` the
    chord views come from the harmonic part of the CQT (module docstring)."""
    y = np.ascontiguousarray(y, dtype=np.float32)
    duration = float(y.size) / float(sr)
    min_len = int(_MIN_SIGNAL_S * sr)
    if y.size < min_len:
        y = np.pad(y, (0, min_len - y.size))
    tuning = float(np.clip(tuning_cents, -50.0, 50.0)) / 100.0
    sub = CQT_BINS_PER_OCTAVE // BINS_PER_OCTAVE
    c = librosa.cqt(
        y,
        sr=sr,
        hop_length=hop,
        fmin=FMIN_HZ * 2.0 ** (-(sub // 2) / CQT_BINS_PER_OCTAVE),   # centre sub-bin on the semitone
        n_bins=N_OCTAVES * CQT_BINS_PER_OCTAVE,
        bins_per_octave=CQT_BINS_PER_OCTAVE,
        tuning=tuning * sub,
    )
    mag_sub = np.abs(c).astype(np.float32)                           # (N_BINS*sub, T)
    harm_sub = harmonic_part(mag_sub) if hpss else mag_sub

    def fold_sub(m: np.ndarray) -> np.ndarray:
        m = m.reshape(N_BINS, sub, m.shape[1]).sum(axis=1)          # fold sub-bins per semitone
        if COMPRESSION != 1.0:
            m = np.power(m, COMPRESSION, dtype=np.float32)
        return m

    mag = fold_sub(mag_sub)
    harm = fold_sub(harm_sub) if hpss else mag
    n_frames = int(np.floor(duration * sr / hop)) + 1
    n_frames = max(1, min(n_frames, mag.shape[1]))
    mag = mag[:, :n_frames]
    harm = harm[:, :n_frames]
    times = librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=hop)
    freqs = librosa.cqt_frequencies(N_BINS, fmin=FMIN_HZ, bins_per_octave=BINS_PER_OCTAVE, tuning=tuning)
    weighted = fold_octaves(harm, OCTAVE_WEIGHTS)
    energy = weighted.sum(axis=0).astype(np.float32)
    return ChromaFeatures(
        sr=int(sr),
        hop=int(hop),
        times=times.astype(np.float64),
        cqt=mag,
        freqs=np.asarray(freqs, dtype=np.float64),
        weighted=compress(weighted),
        bass=fold_octaves(harm, BASS_WEIGHTS),
        treble=compress(fold_octaves(mag, TREBLE_WEIGHTS)),
        energy=energy,
        duration=duration,
    )


def harmonic_part(mag: np.ndarray) -> np.ndarray:
    """Harmonic component of a (bins, T) magnitude spectrogram (see module
    docstring). Too few frames for the time kernel: returned unchanged."""
    if mag.shape[1] < HPSS_TIME_KERNEL or mag.shape[0] < HPSS_FREQ_KERNEL:
        return mag
    harm, _ = librosa.decompose.hpss(
        mag, kernel_size=(HPSS_TIME_KERNEL, HPSS_FREQ_KERNEL), power=HPSS_POWER, margin=1.0
    )
    return np.asarray(harm, dtype=np.float32)


def compress(chroma: np.ndarray) -> np.ndarray:
    """Power-law compression of a folded chroma (see module docstring)."""
    if CHROMA_COMPRESSION == 1.0:
        return chroma
    return np.power(np.maximum(chroma, 0.0), CHROMA_COMPRESSION, dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-9) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < eps or nb < eps:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
