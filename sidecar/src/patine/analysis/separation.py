"""§4.5 optional source separation, behind `AnalysisOptions.use_source_separation`.

When the flag is on and `demucs` (optional `ml` extra) is importable, chords
are estimated from the accompaniment stem (everything but vocals). Analysis
only: playback always uses the untouched full mix. Any failure logs a warning
and the pipeline continues on the full mix; demucs is never a hard dependency."""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


def accompaniment_stem(y: np.ndarray, sr: int, *, model: str = "htdemucs") -> np.ndarray | None:
    """Mono accompaniment (no vocals) at `sr` for mono/stereo `y` (channels, frames),
    or None when demucs is unavailable or fails."""
    try:
        import torch  # type: ignore[import-not-found]
        from demucs.api import Separator  # type: ignore[import-not-found]
    except ImportError as e:
        log.warning("source separation requested but demucs is not installed (%s); using the full mix", e)
        return None
    try:
        from .decode import resample_mono

        wav = np.asarray(y, dtype=np.float32)
        if wav.ndim == 1:
            wav = wav[None, :]
        if wav.shape[0] == 1:
            wav = np.repeat(wav, 2, axis=0)
        separator = Separator(model=model, device="cpu", progress=False)
        _origin, stems = separator.separate_tensor(torch.from_numpy(wav[:2]), sr)
        acc = None
        for name, stem in stems.items():
            if name == "vocals":
                continue
            acc = stem if acc is None else acc + stem
        if acc is None:
            raise RuntimeError("demucs returned no non-vocal stems")
        mono = acc.mean(dim=0).cpu().numpy().astype(np.float32)
        return resample_mono(mono, int(separator.samplerate), sr)
    except Exception as e:  # noqa: BLE001 - optional path, never fatal
        log.warning("source separation failed (%s); using the full mix", e)
        return None
