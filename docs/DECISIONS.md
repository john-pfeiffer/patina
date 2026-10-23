# Decisions and open questions

The handoff (§11) asks for these to be raised, not silently decided. Each
entry states what the scaffold does today and the recommendation. Nothing
here is final until confirmed.

## 1. Chord model

**Recommendation: `crema` (Brian McFee, ISC licence) for root + maj/min, with
Patine's own bass-weighted template refinement for the extended qualities.**

Surveyed:

| Model | Licence | Notes |
|---|---|---|
| madmom CNN chord / DBN beat | BSD code, **CC BY-NC-SA models** | Research-only models; PyPI release (0.16.1, 2018) is sdist-only and does not build cleanly on current Python/numpy. Not usable for a shipped product. |
| crema `chord` | **ISC** (code and bundled weights) | Trained on real music (Isophonics/Billboard-style data). Weights ship inside the 5.9 MB wheel. Depends on TensorFlow/Keras, which is the problem: ~500 MB and awkward on Windows/Apple Silicon. |
| Chordino (Vamp) | GPL | Template/NNLS, no learned prior; licence is a question for distribution. |
| Essentia ChordsDetection | AGPL | Template-based; AGPL is a no for a closed desktop app. |
| BTC / research transformers | Research licences | Strong but not redistributable. |

Plan: convert crema's chord model to ONNX once (a one-off script in `scripts/`,
run on a dev machine with TensorFlow) and ship inference through
`onnxruntime` (~15 MB) — no TensorFlow at runtime. The ONNX file is small
(single-digit MB) and is fetched on first run into `<data dir>/models/`. Until
that is done, the pipeline runs its template + Viterbi estimator on
bass-weighted chroma (`analysis/chords.py`), which handles sustained material
well and melody-on-harmony less well; `analysis/learned.py` is the hook the
ONNX backend plugs into. **Decision needed:** approve crema (ISC) + ONNX.

## 2. Stretch engine binding

**Answer: yes, a maintained binding exists.** `python-stretch` (PyPI, MIT,
v0.3.1) wraps Signalsmith Stretch via pybind11 and ships wheels for macOS
(x86_64 + arm64), Windows (win32 + amd64) and Linux, including `cp312-abi3`
wheels that cover every Python ≥ 3.12. The sidecar uses it
(`temporal/stretch.py`). No C extension of our own, no Rubber Band (GPL)
needed. The binding processes blocks at a fixed `timeFactor`; warp maps are
rendered interval by interval with context padding and joined at beat
boundaries (which are transients, so hard joins are in character).

## 3. Render cache on anchor change

**Doing: full re-render of all members, shortest first.** The cache key is
content-addressed over everything the render depends on (sample hash, bounds,
overrides, temporal settings, session bpm, tonal settings, anchor fingerprint,
transpose, `RENDER_VERSION`), so "incremental" later just means splitting the
key into a temporal key and a tonal key and caching the intermediate. The
queue is priority-ordered by region length so the progression becomes playable
again fast. Profile at 20+ regions before adding the intermediate cache.

## 4. Loudness window

**Doing: integrated LUFS over the region** (BS.1770 via pyloudnorm), both for
the measurement and for the Loudness module's match. Regions shorter than the
400 ms gating block are measured on a padded copy. Revisit for hold-mode
regions with long decays if they match badly; short-term max would be the
alternative, exposed as a set-level choice rather than a silent switch.

## 5. Noise floor source

**Doing: harvest the anchor's own floor** — its quietest contiguous ≥200 ms,
looped with a crossfade, levelled to the anchor's `noise_floor_db` — and fall
back to a synthesised floor shaped by the anchor's `noise_spectrum` only when
the anchor has no quiet stretch. The module never subtracts.

## 6. Sidecar packaging

**Recommendation: PyInstaller one-file bundle as a Tauri `externalBin`,
without torch.** The spec (`sidecar/patine-sidecar.spec`) collects librosa,
scipy, numba/llvmlite, soxr, python-stretch, pyloudnorm and excludes
torch/demucs/beat_this/onnxruntime. Expected size: 120–180 MB per platform
(numpy + scipy + numba dominate), which keeps the installer sane. Model-backed
features (ONNX chord model, `beat_this`, demucs) install their runtime into
`<data dir>` on first use via a bootstrap step, so the base installer never
carries torch. Model loading stays fast because weights live on local disk
after the first run. A first-run environment bootstrap for *everything* was
rejected: it makes the first launch depend on the network and on a working
Python toolchain on the user's machine.

Code signing and notarisation are not configured in CI (no certificates in
the repository); the workflow produces unsigned dev artefacts named
`Patine-dev-<sha7>-<OS>`.

## Also decided while scaffolding

- **Tauri over Electron.** The sidecar is a single external binary, which Tauri
  handles natively (`externalBin`), so the "easier under Electron" escape
  hatch was not needed.
- **Chord parsing lives in the sidecar only** (`POST /theory/parse`), so the
  UI and the ranking never disagree about what `bVII` means.
- **Minor-key roman numerals** follow the pop-harmony reading: `VI`, `VII`,
  `bVI`, `bVII` all mean the natural-minor degrees (Ab, Bb in C minor);
  `vii°` is the leading-tone chord (B°). Tests pin this.
- **Typing a session bpm pins it**; otherwise session tempo follows the anchor.
- **Inversions are tracked, shown, never matched** (§10) — `bass_note` is on
  every region and slot but strictness ignores it.
- **Python ≥ 3.12** for the sidecar (librosa 1.0 requires it; python-stretch
  ships abi3 wheels from 3.12).
