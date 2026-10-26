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
torch/demucs/beat_this/onnxruntime. Measured on Linux x86_64 with the
current dependency set: a 169 MB one-file executable that answers `/health`
about 4 s after launch (one-file bundles unpack to a temp dir on start; the
one-dir form would start faster at the cost of a messier bundle). numpy,
scipy and numba dominate the size; it keeps the installer sane. Model-backed
features (ONNX chord model, `beat_this`, demucs) install their runtime into
`<data dir>` on first use via a bootstrap step, so the base installer never
carries torch. Model loading stays fast because weights live on local disk
after the first run. A first-run environment bootstrap for *everything* was
rejected: it makes the first launch depend on the network and on a working
Python toolchain on the user's machine.

Code signing and notarisation are not configured in CI (no certificates in
the repository); the workflow produces ad-hoc-signed dev artefacts named
`Patine-dev-<sha7>-<OS>`.

Two macOS facts learned the hard way: (1) Tauri signs the sidecar with the
hardened runtime, which enforces library validation, so the PyInstaller
bundle's Python framework (signed by another team) is refused at `dlopen`
unless the app ships entitlements — `Entitlements.plist` grants
`disable-library-validation`, `allow-unsigned-executable-memory` and
`allow-jit` (numba) and is applied to every signed binary; (2) uv picks the
newest managed CPython it can (a 3.15 beta on the CI runner) unless pinned,
so `sidecar/.python-version` pins 3.13. CI now launches the bundled sidecar
after `tauri build` and calls `/health` on all three platforms.

## Also decided while scaffolding

- **Tauri over Electron.** The sidecar is a single external binary, which Tauri
  handles natively (`externalBin`), so the "easier under Electron" escape
  hatch was not needed.
- **Chord parsing lives in the sidecar only** (`POST /theory/parse`), so the
  UI and the ranking never disagree about what `bVII` means.
- **Minor-key roman numerals** follow the pop-harmony reading: `VI`, `VII`,
  `bVI`, `bVII` all mean the natural-minor degrees (Ab, Bb in C minor);
  `vii°` is the leading-tone chord (B°). Tests pin this.
- **Tuning target is the anchor's tuning, not A440, when a set has an anchor.**
  The handoff says "correct `tuning_cents` to A440", but the anchor is identity
  through the tonal layer, so a 15-cent-sharp anchor would stay sharp while
  every member went to A440 and the set would read as two tunings. Members are
  resampled to the anchor's tuning instead; without an anchor (previews on an
  empty set) the target is A440. **Raise if A440 was meant literally** — then
  the anchor would also need correcting, which breaks its identity.
- **Beat 0 of every render is the snap point.** Render sample 0 is where
  playback starts (first onset unless Keep pre-roll); retrigger restarts there
  and kit beat markers count from it. Nudge moves the play start without
  moving the grid.
- **Hold fills the slot.** The handoff says hold "plays through for the slot's
  length"; a region shorter than the slot restarts from its snap point (hard
  cut, declicked) until the slot ends rather than trailing silence, and a
  longer one is cut at the slot end. Time-stretching a region to the slot
  length is the other reading; it can be added as a per-slot `fit` choice.
- **Region boundaries snap to the beat grid.** Chord-change boundaries from
  the frame-wise estimator land within a frame or two of the true change;
  they now move onto a tracked beat when one is within about a third of a
  beat period (onsets otherwise), so a region's beat 0 is a beat of the
  source and it plays in phase with the progression's grid.
- **Typing a session bpm pins it**; otherwise session tempo follows the anchor.
- **Inversions are tracked, shown, never matched** (§10) — `bass_note` is on
  every region and slot but strictness ignores it.
- **Python ≥ 3.12** for the sidecar (librosa 1.0 requires it; python-stretch
  ships abi3 wheels from 3.12).
- **The transport clock needs `worker-src blob:` in the CSP.** Tone.js ticks its
  Transport from a Web Worker built from a `blob:` URL and only falls back to
  `setTimeout` when the constructor throws. A webview that refuses the worker
  asynchronously (the Tauri CSP did, on the packaged macOS build) leaves a
  transport whose position advances while nothing scheduled on it fires:
  solo audition and export worked, slot playback was silent, and the position
  ran past the loop end (tracker #35). Both policies now allow blob workers,
  the engine probes a blob worker at startup and watches for ticks after
  Play, switching Tone's `clockSource` to `timeout` when they never come
  (`apps/desktop/src/audio/clock.ts`), and Preferences › Audio shows the clock
  state with three test buttons (tone direct, tone through the bus, clicks on
  the clock) so "no sound" can be bisected in one click. The engine is also
  rendered offline in headless Chromium by `pnpm test:browser`.
- **The engine keeps playing a stale render.** `ensure_render` already returned
  the last good URL while a new key renders; the UI dropped it unless the
  status was `ready`. It now plays whatever URL the status carries for the
  slot's region (`renderPathFor`), as §2 of the handoff asks.
- **Output device selection is Chromium-only.** `AudioContext.setSinkId` and
  `audiooutput` enumeration exist on WebView2, not on WKWebView, so on macOS
  Preferences › Audio reports the system output and says where to change it
  rather than offering a picker that does nothing.
- **Chord recall: harmonic chroma, a structure gate, a one-beat observation
  window, a whole-span fallback and the `5` quality.** Samples came back with
  no regions for three reasons: a broadband floor (drums, noise, reverb) of a
  third of the chord's level made the old flatness gate call every frame
  "no chord"; a melody or arpeggio was decoded one chord per note and then
  merged away; and a file whose frames all decoded as no-chord was simply
  dropped. The chord chroma is now folded from the harmonic part of the CQT
  (`librosa.decompose.hpss`, 3-frame × 31-sub-bin kernels; the full mix still
  feeds busyness and the register centroid), the no-chord state scores
  `0.45 + 0.55 × max(silence, no_structure)` where `no_structure` is low when
  the best maj/min template stands out from the median fit by 0.16 or more
  (noise, drums and rumble stay below 0.14), the HMM sees chroma averaged over
  about one beat (3–9 frames) with stickiness of two beats, an extended
  quality is only allowed when each tone it adds stands 1.4× above the floor
  of the unused pitch classes (a four-note template otherwise wins under
  drums by covering more floor), and a file that still decodes as nothing
  gets one region over its longest tonal span from the floor-subtracted mean
  chroma, capped at confidence 0.5 so it ranks behind decoded regions. A bare
  root or root+fifth is reported as quality `5` instead of an invented third;
  `C5` matches any chord on C at `function`/`two_tones` strictness, parses
  from typed progressions and renders as `I5`. `scripts/eval_chords.py`
  measures the share of files with at least one region on a real folder,
  before (`--legacy`) and after.
- **Reanalysis keeps ids.** `ANALYSIS_VERSION` is 2; samples analysed by an
  older version are re-queued at startup and on re-ingest. A re-run matches
  new regions to old ones by temporal overlap and updates them in place, so
  set membership, the anchor and slot assignments survive; any region the
  user has edited becomes `source: user` and is never re-cut. Whole-file
  onsets are now stored on the sample (schema v2) so a region can be
  re-measured and its transient edited without re-running the file.
- **Library roots scan; they never delete.** A root is a folder (a Splice
  library) the sidecar walks recursively, skipping dot files and folders and
  macOS `._` sidecars. New files are indexed and analysed in the background at
  low priority; files whose size or mtime changed are re-analysed; files that
  vanished are marked `missing` and un-marked when they return. Removing a
  root keeps its samples as loose files unless asked to delete them, and
  re-adding the folder re-adopts them. Every root is rescanned when the
  sidecar starts (`scan_on_startup`). Analysis can be paused from
  Preferences: the worker parks analysis jobs and keeps rendering and
  exporting. Duplicate detection stays by path; the hash is stored for the
  render cache, and two copies of one file are two samples on purpose.
