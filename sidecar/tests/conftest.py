import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


@pytest.fixture
def tmp_data_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def app_client(tmp_data_dir):
    """FastAPI test client over a fresh data dir with the worker running."""
    from fastapi.testclient import TestClient

    from patine.paths import DataPaths
    from patine.server import AppState, create_app

    os.environ.pop("PATINE_DATA_DIR", None)
    state = AppState(DataPaths(tmp_data_dir), "test-token", workers=1)
    app = create_app(state)
    with TestClient(app, headers={"X-Patine-Token": "test-token"}) as client:
        client.state_obj = state  # type: ignore[attr-defined]
        yield client
    state.close()


def synth_chords(path: Path, chords=((0, "maj"), (5, "maj"), (7, "maj"), (0, "maj")), bpm=120.0,
                 beats_per_chord=4, sr=22050, detune_cents=0.0, stereo=True, seed=0):
    """Sustained chord pads with a click on every beat. Returns (path, duration)."""
    from patine.theory import QUALITY_INTERVALS

    rng = np.random.default_rng(seed)
    beat = 60.0 / bpm
    total = beat * beats_per_chord * len(chords)
    n = int(total * sr)
    t = np.arange(n) / sr
    y = np.zeros(n, dtype=np.float64)
    for ci, (root, quality) in enumerate(chords):
        a = int(ci * beats_per_chord * beat * sr)
        b = int((ci + 1) * beats_per_chord * beat * sr)
        seg_t = t[a:b] - t[a]
        env = np.minimum(1.0, seg_t / 0.02) * np.minimum(1.0, (seg_t[-1] - seg_t) / 0.02 + 1e-3)
        for iv in QUALITY_INTERVALS[quality]:
            midi = 48 + root + iv
            f0 = 440.0 * 2 ** ((midi - 69) / 12) * 2 ** (detune_cents / 1200)
            for h in range(1, 5):
                y[a:b] += (0.25 / h) * np.sin(2 * np.pi * f0 * h * seg_t) * env
    n_beats = int(round(total / beat))
    for k in range(n_beats):
        i = int(k * beat * sr)
        click = np.exp(-np.arange(400) / 60.0) * rng.standard_normal(400)
        seg = click[: max(0, min(400, n - i))]
        y[i:i + len(seg)] += 0.8 * seg
    y += 0.002 * rng.standard_normal(n)
    y /= max(1e-9, np.abs(y).max()) * 1.2
    data = np.stack([y, y * 0.9 + 0.01 * rng.standard_normal(n)], axis=1) if stereo else y
    sf.write(str(path), data.astype(np.float32), sr, subtype="FLOAT")
    return path, total
