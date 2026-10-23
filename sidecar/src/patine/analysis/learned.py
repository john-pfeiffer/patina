"""Learned root+maj/min chord backend (handoff §4.5, open question 1).

The handoff asks for a learned model trained on real music for the
root + major/minor segmentation, with template refinement for the extended
qualities. The weights are not bundled: they are fetched on first run into
`<data dir>/models/` (ARCHITECTURE.md) and loaded here through ONNX Runtime,
which is part of the optional `ml` extra.

Until a permissively licensed model has been chosen and exported, this class
is an integration point: constructing it raises `ImportError` (no
onnxruntime) or `FileNotFoundError` (no weights), and `chords.get_backend`
falls back to the template backend. The expected model contract is below so
an export can be dropped in without touching the pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .chords import N_STATES, ChromaFeatures

log = logging.getLogger(__name__)

MODEL_FILENAME = "chord_rootmajmin.onnx"


class LearnedChordBackend:
    """ONNX model mapping bass-weighted chroma to 25 chord-state logits.

    Model contract: input ``chroma`` float32 (1, 12, T); output float32
    (1, 25, T) logits or probabilities ordered like `chords.state_index`
    (0..11 major roots C..B, 12..23 minor roots, 24 no-chord)."""

    name = "learned"

    def __init__(self, models_dir: str | Path | None) -> None:
        try:
            import onnxruntime as ort  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "learned chord backend needs onnxruntime; install the 'ml' extra (pip install 'patine[ml]')"
            ) from e
        if models_dir is None:
            raise FileNotFoundError("learned chord backend: no models directory configured")
        path = Path(models_dir) / MODEL_FILENAME
        if not path.is_file():
            raise FileNotFoundError(f"learned chord backend: model weights not found at {path}")
        self.path = path
        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self._input = self.session.get_inputs()[0].name
        log.info("learned chord model loaded from %s", path)

    def observation_probabilities(self, feats: ChromaFeatures) -> np.ndarray:
        x = feats.weighted.astype(np.float32)[None, :, :]
        (out,) = self.session.run(None, {self._input: x})
        logits = np.asarray(out, dtype=np.float64)[0]
        if logits.shape[0] != N_STATES:
            raise ValueError(f"learned chord model returned {logits.shape[0]} states, expected {N_STATES}")
        if np.all(logits >= 0.0) and np.allclose(logits.sum(axis=0), 1.0, atol=1e-3):
            return logits
        logits = logits - logits.max(axis=0, keepdims=True)
        p = np.exp(logits)
        return p / p.sum(axis=0, keepdims=True)
