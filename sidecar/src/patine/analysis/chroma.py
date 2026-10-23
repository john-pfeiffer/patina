"""Chroma features for chord, key and activity estimation (handoff §4.5–4.7).

One constant-Q transform (C1–B7, 12 bins per octave, hop 2048 at 22050 Hz, so
about 93 ms per frame) feeds three chroma views:

* ``weighted`` — **bass-weighted** chroma. The low octaves count more, so a
  melody line leaning on a 9th up top cannot flip the chord estimate (§4.5).
* ``bass`` — the lowest octaves only (C1–B3), used to report ``bass_note``
  separately so inversions are visible.
* ``treble`` — the upper octaves only (C4–B7), used by the busyness score (§4.7).

Magnitudes are compressed with a power law so one loud partial does not swamp
the softer chord tones. All arrays are float32."""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

ANALYSIS_SR = 22050
CHROMA_HOP = 2048               # 92.9 ms at 22050 Hz; divisible by 2**(N_OCTAVES-1)
N_OCTAVES = 7
BINS_PER_OCTAVE = 12
N_BINS = N_OCTAVES * BINS_PER_OCTAVE
FMIN_HZ = float(librosa.note_to_hz("C1"))      # 32.70 Hz
COMPRESSION = 0.5                                # magnitude exponent

# Per-octave weights, index 0 = C1–B3 ... 6 = C7–B7.
OCTAVE_WEIGHTS = np.array([0.6, 1.0, 1.0, 0.8, 0.5, 0.3, 0.15], dtype=np.float32)
BASS_WEIGHTS = np.array([1.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
TREBLE_WEIGHTS = np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0], dtype=np.float32)

_MIN_SIGNAL_S = 1.0  # pad shorter inputs so the low CQT filters have room


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
) -> ChromaFeatures:
    """CQT-based chroma over mono `y` at `sr`, compensated for `tuning_cents` (§4.2)
    so a detuned file still lands on the right pitch classes."""
    y = np.ascontiguousarray(y, dtype=np.float32)
    duration = float(y.size) / float(sr)
    min_len = int(_MIN_SIGNAL_S * sr)
    if y.size < min_len:
        y = np.pad(y, (0, min_len - y.size))
    tuning = float(np.clip(tuning_cents, -50.0, 50.0)) / 100.0
    c = librosa.cqt(
        y,
        sr=sr,
        hop_length=hop,
        fmin=FMIN_HZ,
        n_bins=N_BINS,
        bins_per_octave=BINS_PER_OCTAVE,
        tuning=tuning,
    )
    mag = np.abs(c).astype(np.float32)
    mag = np.power(mag, COMPRESSION, dtype=np.float32)
    n_frames = int(np.floor(duration * sr / hop)) + 1
    n_frames = max(1, min(n_frames, mag.shape[1]))
    mag = mag[:, :n_frames]
    times = librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=hop)
    freqs = librosa.cqt_frequencies(N_BINS, fmin=FMIN_HZ, bins_per_octave=BINS_PER_OCTAVE, tuning=tuning)
    weighted = fold_octaves(mag, OCTAVE_WEIGHTS)
    return ChromaFeatures(
        sr=int(sr),
        hop=int(hop),
        times=times.astype(np.float64),
        cqt=mag,
        freqs=np.asarray(freqs, dtype=np.float64),
        weighted=weighted,
        bass=fold_octaves(mag, BASS_WEIGHTS),
        treble=fold_octaves(mag, TREBLE_WEIGHTS),
        energy=weighted.sum(axis=0).astype(np.float32),
        duration=duration,
    )


def cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-9) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < eps or nb < eps:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
