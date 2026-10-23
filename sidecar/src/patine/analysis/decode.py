"""§4.1 Decode & normalise.

Any common format is decoded to float32 at the file's native sample rate with
stereo preserved (`audio.load_audio`). A mono mix is derived for analysis only,
and a copy of that mono mix is resampled to `ANALYSIS_SR` for the chroma, beat
and tuning work, which does not need more bandwidth than 11 kHz. The tonal
metrics (§4.8) are measured at the native rate on the untouched channels."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soxr

from ..audio import load_audio, to_mono

log = logging.getLogger(__name__)

ANALYSIS_SR = 22050


@dataclass
class DecodedAudio:
    y: np.ndarray        # (channels, frames) float32 at the native rate
    sr: int
    mono: np.ndarray     # (frames,) float32 at the native rate
    mono_a: np.ndarray   # (frames_a,) float32 at `sr_a`
    sr_a: int = ANALYSIS_SR

    @property
    def channels(self) -> int:
        return int(self.y.shape[0])

    @property
    def duration(self) -> float:
        return float(self.y.shape[1]) / float(self.sr)


def resample_mono(mono: np.ndarray, sr: int, target_sr: int = ANALYSIS_SR) -> np.ndarray:
    """Resample a 1-D float32 signal with soxr (HQ). Identity when rates match."""
    mono = np.ascontiguousarray(mono, dtype=np.float32)
    if int(sr) == int(target_sr) or mono.size == 0:
        return mono
    out = soxr.resample(mono, int(sr), int(target_sr), quality="HQ")
    return np.ascontiguousarray(out, dtype=np.float32)


def decode(path: str | Path) -> DecodedAudio:
    """Decode `path` (§4.1). Raises whatever `audio.load_audio` raises."""
    y, sr = load_audio(path)
    y = np.ascontiguousarray(y, dtype=np.float32)
    if y.ndim == 1:
        y = y[None, :]
    if not np.all(np.isfinite(y)):
        log.warning("%s: non-finite samples replaced with 0", Path(path).name)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    mono = to_mono(y)
    mono_a = resample_mono(mono, sr, ANALYSIS_SR)
    return DecodedAudio(y=y, sr=int(sr), mono=mono, mono_a=mono_a, sr_a=ANALYSIS_SR)
