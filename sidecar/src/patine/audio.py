"""Decode, write and hash audio. Shared by analysis, conform and export.

Arrays are float32 shaped (channels, frames). Stereo is preserved; callers that
want mono derive it with `to_mono` for analysis only (handoff §4.1)."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

AUDIO_EXTENSIONS = {".wav", ".aif", ".aiff", ".flac", ".ogg", ".oga", ".mp3", ".m4a", ".aac", ".wma", ".opus", ".caf"}


def file_hash(path: str | Path) -> str:
    h = hashlib.blake2b(digest_size=16)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """Any common format → float32 (channels, frames) at the file's native rate.
    libsndfile first; ffmpeg (if on PATH) for anything it cannot open."""
    path = Path(path)
    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return np.ascontiguousarray(data.T), int(sr)
    except Exception as e:  # noqa: BLE001 - fall through to ffmpeg
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError(f"cannot decode {path.name}: {e}; ffmpeg not found for fallback") from e
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "decoded.wav"
        cmd = [ffmpeg, "-v", "error", "-y", "-i", str(path), "-c:a", "pcm_f32le", str(out)]
        subprocess.run(cmd, check=True, capture_output=True)
        data, sr = sf.read(str(out), dtype="float32", always_2d=True)
        return np.ascontiguousarray(data.T), int(sr)


def probe(path: str | Path) -> tuple[float, int, int]:
    """(duration_s, sample_rate, channels) without decoding when possible."""
    try:
        info = sf.info(str(path))
        return float(info.frames) / float(info.samplerate), int(info.samplerate), int(info.channels)
    except Exception:  # noqa: BLE001
        y, sr = load_audio(path)
        return y.shape[1] / sr, sr, y.shape[0]


def to_mono(y: np.ndarray) -> np.ndarray:
    if y.ndim == 1:
        return y.astype(np.float32, copy=False)
    return y.mean(axis=0).astype(np.float32, copy=False)


def slice_seconds(y: np.ndarray, sr: int, start_s: float, end_s: float) -> np.ndarray:
    a = max(0, int(round(start_s * sr)))
    b = min(y.shape[-1], int(round(end_s * sr)))
    return y[..., a:b]


def write_wav(path: str | Path, y: np.ndarray, sr: int, subtype: str = "FLOAT") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = y.T if y.ndim == 2 else y
    sf.write(str(path), data, sr, subtype=subtype)


def is_audio_file(path: str | Path) -> bool:
    return Path(path).suffix.lower() in AUDIO_EXTENSIONS


def is_hidden(path: str | Path) -> bool:
    """Dot files and macOS AppleDouble `._x.wav` sidecars are never audio to scan."""
    name = Path(path).name
    return name.startswith(".") or name.startswith("._")


def iter_audio_files(folder: str | Path):
    """Audio files under `folder`, recursively, skipping hidden files and
    directories; sorted so scans are reproducible."""
    root = Path(folder)
    found: list[Path] = []
    for p in root.rglob("*"):
        if any(is_hidden(part) for part in p.relative_to(root).parts):
            continue
        if p.is_file() and is_audio_file(p):
            found.append(p)
    return sorted(found)
