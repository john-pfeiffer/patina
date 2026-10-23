"""§4.3 Tempo & beats.

Beat tracking over the whole file gives beat positions, a BPM estimate and a
confidence. Trackers are reliable about *where* the pulse is and unreliable
about *which level*, so the BPM is a suggestion; the temporal layer (§7.1)
has ½×/2× buttons and a declared-length path that override it.

Backends: `librosa` (always available) and `beat_this` (optional `ml` extra),
selected through `AnalysisOptions.beat_tracker`. Downbeats are best-effort:
`beat_this` reports them; the librosa path picks the 4/4 phase whose beats
carry the most onset energy and harmonic change, and reports nothing when no
phase stands out."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import librosa
import numpy as np

log = logging.getLogger(__name__)

ONSET_HOP = 512


@dataclass
class TempoResult:
    bpm: float
    confidence: float             # 0..1
    beats: list[float]            # seconds
    downbeats: list[float] = field(default_factory=list)
    backend: str = "librosa"
    onset_envelope: np.ndarray | None = None
    hop_length: int = ONSET_HOP


def onset_envelope(y: np.ndarray, sr: int, hop_length: int = ONSET_HOP) -> np.ndarray:
    y = np.ascontiguousarray(y, dtype=np.float32)
    if y.size < 4096:
        y = np.pad(y, (0, 4096 - y.size))
    return librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length).astype(np.float32)


def _refine_beat_times(beat_frames: np.ndarray, env: np.ndarray, sr: int, hop: int) -> np.ndarray:
    """Sub-frame beat times by parabolic interpolation of the onset envelope."""
    times = []
    n = env.size
    for f in beat_frames.astype(int):
        off = 0.0
        if 0 < f < n - 1:
            left, mid, right = float(env[f - 1]), float(env[f]), float(env[f + 1])
            denom = left - 2.0 * mid + right
            if denom < 0.0 and mid >= left and mid >= right:
                off = float(np.clip(0.5 * (left - right) / denom, -0.5, 0.5))
        times.append((f + off) * hop / sr)
    return np.asarray(times, dtype=np.float64)


def bpm_from_beats(beats: np.ndarray, fallback: float) -> float:
    """BPM from the mean of the plausible inter-beat intervals (drops the
    doubled/skipped ones) so frame quantisation averages out."""
    if beats.size < 2:
        return float(fallback)
    ibi = np.diff(beats)
    ibi = ibi[ibi > 1e-3]
    if ibi.size == 0:
        return float(fallback)
    med = float(np.median(ibi))
    keep = ibi[(ibi > 0.75 * med) & (ibi < 1.33 * med)]
    mean_ibi = float(keep.mean()) if keep.size else med
    return 60.0 / mean_ibi if mean_ibi > 0 else float(fallback)


def tempo_confidence(env: np.ndarray, beats: np.ndarray, bpm: float, sr: int, hop: int) -> float:
    """0..1, mostly how periodic the onset envelope is at the beat period
    (normalised autocorrelation: clicks ≈ 0.9, noise ≈ 0.1, a pad with no
    transients ≈ 0), with a small term for how much onset energy sits on the
    tracked beats. The beat tracker always returns *something*, so this is
    what tells the UI how much to trust it (§7.1)."""
    if env.size < 8 or bpm <= 0 or beats.size < 2:
        return 0.0
    x = env.astype(np.float64) - float(env.mean())
    if float(np.abs(x).max()) <= 0.0:
        return 0.0
    ac = librosa.autocorrelate(x)
    lag = int(round(60.0 / bpm * sr / hop))
    if lag < 1 or lag >= ac.size - 1 or ac[0] <= 0:
        return 0.0
    periodicity = float(np.clip(ac[lag - 1 : lag + 2].max() / ac[0], 0.0, 1.0))
    frames = np.clip(np.round(beats * sr / hop).astype(int), 0, env.size - 1)
    at_beats = float(np.mean(env[frames]))
    med = float(np.median(env))
    hi = float(np.percentile(env, 95))
    alignment = float(np.clip((at_beats - med) / max(hi - med, 1e-9), 0.0, 1.0))
    return float(np.clip(0.9 * periodicity + 0.1 * alignment, 0.0, 1.0))


def _track_librosa(y: np.ndarray, sr: int, env: np.ndarray, hop: int) -> TempoResult:
    tempo_est, beat_frames = librosa.beat.beat_track(
        onset_envelope=env, sr=sr, hop_length=hop, units="frames", trim=False
    )
    tempo_f = float(np.atleast_1d(tempo_est)[0])
    beats = _refine_beat_times(np.asarray(beat_frames), env, sr, hop)
    bpm = bpm_from_beats(beats, tempo_f)
    conf = tempo_confidence(env, beats, bpm, sr, hop)
    return TempoResult(
        bpm=bpm, confidence=conf, beats=[float(b) for b in beats], backend="librosa",
        onset_envelope=env, hop_length=hop,
    )


def _track_beat_this(y: np.ndarray, sr: int, env: np.ndarray, hop: int) -> TempoResult:
    """Optional backend (`ml` extra). Raises ImportError when not installed."""
    from beat_this.inference import Audio2Beats  # type: ignore[import-not-found]

    model = Audio2Beats(checkpoint_path="final0", device="cpu", dbn=False)
    beats_arr, downbeats_arr = model(np.asarray(y, dtype=np.float32), sr)
    beats = np.asarray(beats_arr, dtype=np.float64)
    downbeats = [float(d) for d in np.asarray(downbeats_arr, dtype=np.float64)]
    if beats.size < 2:
        raise RuntimeError("beat_this returned fewer than two beats")
    bpm = bpm_from_beats(beats, 120.0)
    conf = tempo_confidence(env, beats, bpm, sr, hop)
    return TempoResult(
        bpm=bpm, confidence=conf, beats=[float(b) for b in beats], downbeats=downbeats,
        backend="beat_this", onset_envelope=env, hop_length=hop,
    )


_BACKENDS: dict[str, Callable[[np.ndarray, int, np.ndarray, int], TempoResult]] = {
    "librosa": _track_librosa,
    "beat_this": _track_beat_this,
}


def track_beats(
    y: np.ndarray,
    sr: int,
    *,
    backend: str = "auto",
    env: np.ndarray | None = None,
    hop_length: int = ONSET_HOP,
) -> TempoResult:
    """Beat positions, BPM and confidence for mono `y` at `sr` (§4.3).
    `backend`: 'auto' (beat_this when importable, else librosa), 'librosa', 'beat_this'."""
    y = np.ascontiguousarray(y, dtype=np.float32)
    if env is None:
        env = onset_envelope(y, sr, hop_length)
    order = ["beat_this", "librosa"] if backend == "auto" else [backend, "librosa"]
    last_error: Exception | None = None
    for name in order:
        fn = _BACKENDS.get(name)
        if fn is None:
            log.warning("unknown beat tracker %r, using librosa", name)
            continue
        try:
            return fn(y, sr, env, hop_length)
        except ImportError as e:
            last_error = e
            if backend != "auto":
                log.warning("beat tracker %r unavailable (%s); falling back to librosa", name, e)
            else:
                log.debug("beat tracker %r unavailable: %s", name, e)
        except Exception as e:  # noqa: BLE001 - never let an optional backend sink the analysis
            last_error = e
            log.warning("beat tracker %r failed (%s); falling back to librosa", name, e)
    raise RuntimeError(f"no beat tracker succeeded: {last_error}")


def estimate_downbeats(
    beats: list[float],
    env: np.ndarray,
    sr: int,
    hop: int,
    *,
    chroma: np.ndarray | None = None,
    frame_times: np.ndarray | None = None,
    beats_per_bar: int = 4,
    min_margin: float = 0.12,
) -> list[float]:
    """Best-effort 4/4 downbeats for the librosa path. Scores each phase by the
    onset energy and the harmonic change (chroma novelty) on its beats and
    returns that phase, or [] when no phase wins by `min_margin`."""
    b = np.asarray(beats, dtype=np.float64)
    if b.size < 2 * beats_per_bar or env.size == 0:
        return []
    frames = np.clip(np.round(b * sr / hop).astype(int), 0, env.size - 1)
    env_n = env.astype(np.float64)
    env_n = env_n / max(float(env_n.max()), 1e-9)
    score = env_n[frames]
    if chroma is not None and frame_times is not None and chroma.shape[1] > 2:
        idx = np.clip(np.searchsorted(frame_times, b), 1, chroma.shape[1] - 1)
        x = chroma / (np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-9)
        novelty = 1.0 - np.sum(x[:, idx] * x[:, idx - 1], axis=0)
        score = score + np.clip(novelty, 0.0, 1.0)
    phases = np.array([score[p::beats_per_bar].mean() for p in range(beats_per_bar)])
    order = np.argsort(phases)[::-1]
    best, second = phases[order[0]], phases[order[1]]
    if best <= 0 or (best - second) / best < min_margin:
        return []
    return [float(t) for t in b[int(order[0]) :: beats_per_bar]]
