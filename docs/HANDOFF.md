# Patine — v1 Handoff

*Élan Vital Studios · chord-sampling playground · desktop app*

---

## 0. What this is

Patine is a local-first desktop playground for building chord progressions out of samples. Drop in a library of audio — one-shot chord hits, loops, or stretches of real recordings that happen to sit on a chord — and Patine finds the harmonically stable regions inside each file, tags them with chord, key, tempo and a set of tonal measurements, and lets you assign them to the slots of a progression. The result plays back as hard-cut chops, one region per chord, retriggered on a grid or held through.

The reason it exists is the conform layer. Regions pulled from different records don't sound like one instrument; they sound like a splice. Patine measures what makes two recordings read as *different recordings* — tuning drift, level, spectral tilt, low-end floor, noise floor, stereo width, density — and conforms every region in the set toward a chosen anchor sample. It never touches what makes them *different music*: harmonic content, instrumentation, melody, voicing, pre-roll. The name is the principle — patina is the surface that makes objects from different places look like they aged together.

**The one rule that governs every design decision below: conform the recording, not the music.**

Smoothness in Patine means *tonal* smoothness. It does not mean crossfades, blur, or hiding seams. Cuts are hard. Melodic collisions between regions are the user's problem, solved by choosing samples, and Patine's job there is to make choosing easy.

---

## 1. Build workflow

Same as Folie, Refonte and Sillage:

- **Fresh scaffold.** Do not template from the plugin repos or any prior EVS web app.
- **Branches:** `main` (releases) and `develop` (integration). All feature work branches from `develop`.
- **Fast dev builds.** A single command (`pnpm dev` or equivalent) that launches the desktop shell with hot reload on the UI and auto-starts the analysis sidecar from source. Testing a change must never require a packaged build.
- **Release builds** via the shell's bundler, producing signed Mac and PC installers. Model weights and any large runtime dependencies are fetched on first run into the app data directory, not bundled.
- **Notion tracker:** create a database titled **"Patine — Dev Tracker"** with the standard EVS status flow: `Backlog → To do → In progress → Fixed in develop → Shipped in release`. Item types: Feature / Bug / Idea. Claude Code reads and updates it via the Notion MCP.

---

## 2. Architecture

**Form factor:** desktop app, Mac and PC.

**Shell:** Tauri (Rust) hosting a React + TypeScript UI. Tauri is preferred over Electron for binary size and memory; if the Python sidecar packaging turns out to be materially easier under Electron, that is an acceptable swap — raise it rather than silently choosing.

**Engine:** a Python sidecar, spawned by the shell on launch, exposing a local HTTP/JSON API (FastAPI or equivalent on a random localhost port handed to the UI). The sidecar owns:

- the SQLite library database
- the analysis pipeline (section 4)
- the conform renderer (sections 6–7)
- the render cache
- export

**Playback:** the UI plays audio. The sidecar renders every conformed region to a WAV in the cache; the UI loads those into Web Audio and schedules them with a lookahead clock (Tone.js Transport is fine). This is the central simplification of the whole app:

> **Offline conform, online play.** No real-time DSP beyond gain, declick ramps and the shared bus. Anything heavy renders to disk in the background. The UI never waits on DSP; it plays the last good render and shows a "stale" marker until the new one lands.

**Render cache:** content-addressed. Cache key = hash(source file, region bounds, temporal settings, tonal settings, anchor fingerprint). Changing the anchor invalidates every member's tonal render; the renderer processes them in playback order, shortest first, so the progression becomes playable again fast.

**Database:** SQLite, owned by the sidecar. Schema in section 3. No cloud, no sync, no accounts in v1.

---

## 3. Data model

```
Sample
  id, path, hash, duration, sample_rate, channels
  added_at, tuning_cents, detected_bpm, bpm_confidence
  key_estimate, key_confidence
  whole-file tonal metrics (see Region — same fields, measured over the file)

Region                              -- the usable unit
  id, sample_id, start_s, end_s
  chord_root, chord_quality, bass_note, chord_confidence
  key_context
  beats[]                           -- absolute beat positions in seconds
  first_onset_s                     -- transient snap point
  declared_length_beats (nullable)  -- user-declared, overrides detection
  busyness, register_centroid_hz
  loudness_lufs, tilt_bands[10], low_rolloff_hz, bass_mono_hz, sub_ratio
  noise_floor_db, noise_spectrum[10], width_bands[3], crest_db, wetness
  user_overrides (json)             -- pre-roll, nudge, warp markers, bypasses

Set
  id, name, anchor_region_id, session_bpm, bpm_pinned
  tonal_settings (json)             -- module toggles + amounts
  temporal_settings (json)
  members[] -> Region

Progression
  id, set_id, key, slots[]
Slot
  index, chord_symbol, roman, length_beats
  region_id (nullable), pattern (json)
Pattern
  mode: hold | retrigger
  division: 1/4 | 1/8 | 1/16        -- retrigger only
  steps[]: { gate, velocity, probability, offset_beats }
```

---

## 4. Analysis pipeline

Runs once per sample on ingest, in a background worker, with a progress indicator in the library pane. Re-runs only on explicit request. Every output is stored; nothing is computed at playback time.

**4.1 Decode & normalise.** Any common format → float32, native sample rate kept, stereo preserved. A mono mix is derived for analysis only.

**4.2 Tuning offset.** Estimate deviation from A440 in cents over the whole file (pitch-class histogram against a shifted reference; `librosa.estimate_tuning` is the floor, improve if it's noisy on dense material). Stored as `tuning_cents`.

**4.3 Tempo & beats.** Beat tracking on the whole file, producing beat positions and a BPM estimate with confidence. Beat trackers are reliable about *where* the pulse is and unreliable about *which level* — treat the BPM as a suggestion, never the authority (see 7.1). Downbeat estimation if the tracker supports it.

**4.4 Onsets.** Transient detection over the whole file. For each region, `first_onset_s` = first onset at or after the region's harmonic start.

**4.5 Chord estimation.** Frame-wise, over the whole file, then segmented into harmonically stable regions. Requirements:

- Must handle melody on top of harmony. Plain chroma template matching is fooled by a line leaning on a 9th; use a learned chord model trained on real music for root + major/minor segmentation, then refine quality (7, m7, maj7, dim, sus2, sus4) by template matching on **bass-weighted chroma** within the segment.
- Report `bass_note` separately so inversions are visible.
- Segments shorter than one beat are merged into neighbours. Each region carries `chord_confidence`.
- **Optional, behind a flag:** run a source separator on the sample and estimate chords from the accompaniment stem. Analysis only — playback always uses the untouched full mix. Off by default in v1; it is slow and the model is large.

**4.6 Key.** Whole-file key estimate plus a `key_context` per region (the key the region sits in, which can differ from the file key in modulating material).

**4.7 Busyness & register.** Per region: a melodic-activity score (rate of pitch-class change in the upper register, roughly) and a harmonic register centroid in Hz. These are browsing aids so the user can pair a busy region with a sparse one. They constrain nothing.

**4.8 Tonal metrics.** Per region and per whole file:

| Metric | How |
|---|---|
| `loudness_lufs` | ITU-R BS.1770 integrated over the region (`pyloudnorm`) |
| `tilt_bands[10]` | Long-term average spectrum (Welch), collapsed to 10 octave bands 31.5 Hz–16 kHz, dB |
| `low_rolloff_hz` | Frequency below which energy drops >12 dB relative to the 100–300 Hz band |
| `bass_mono_hz` | Highest frequency below which L/R correlation stays >0.9 |
| `sub_ratio` | Energy below 60 Hz relative to total |
| `noise_floor_db`, `noise_spectrum[10]` | Level and octave-band shape of the quietest 5% of frames |
| `width_bands[3]` | Side/Mid energy ratio in <200 Hz, 200 Hz–3 kHz, >3 kHz |
| `crest_db` | Peak-to-RMS |
| `wetness` | Late-energy ratio after onsets (decay slope). Informational unless Space is enabled |

---

## 5. Library & selection

**5.1 Library pane.** Every sample with its regions expanded underneath. Per region: chord, bass note, key context, beats, busyness, register, and the tonal metrics as compact readouts. Audition any region solo or *in context* — i.e. inserted into the current progression at the hovered slot, played against the rest of the set.

**5.2 Candidates for a slot.** For a slot with chord X, candidate regions are ranked by:

```
score = transposition_cost + timbral_distance_from_anchor + (1 - chord_confidence)
```

where `transposition_cost` is 0 for an exact match, small for ±1–2 semitones, and large beyond that.

**5.3 Strictness slider** (three positions):

- **Exact** — same root and quality.
- **Same function** — plus relative major/minor and common substitutions sharing the function in `key_context`.
- **Shares two tones** — any chord sharing ≥2 pitch classes with the target.

Looser strictness surfaces substitutions *before* transpositions, because a full-mix region shifted 5 semitones is a vocal that's wrong, while an Ab region in an Fm slot is often smoother than a repitched Fm.

**5.4 Transposition policy.** Exact first. ±2 semitones acceptable. Beyond that, substitution is preferred and transposition is available but ranked last and labelled with the semitone count. Repitch mode is a set-level choice: **Plain** (resample — changes length, compensated by the stretch stage; the classic sampler sound) or **Elastic** (pitch shift at constant length, formant-preserving where the engine supports it).

**5.5 Reroll.** The EVS Randomize convention, adapted: reassigns every unlocked slot from its candidate pool, respecting strictness. Locked slots are untouched. A Reroll Amount knob (0–100%) controls the fraction of slots that change.

---

## 6. Tonal conform layer

Everything here is applied per region, offline, against the **anchor** — one region in the set that all others are conformed toward. The anchor is identity through this layer: its render must be bit-identical to its source (after the temporal layer). This is also the test that the matching works.

**Anchor rules:** defaults to the first region added to the set; re-selectable at any time from the set pane. Switching recomputes every member's deltas and triggers re-render. Every member shows a **delta readout** — how far it moved on each module — which is how the user spots the one sample fighting the set.

Every module has a global toggle and a per-region bypass. Modules with an amount have it exposed.

| Module | Default | Amount | Behaviour |
|---|---|---|---|
| Tuning | **On** | — | Correct `tuning_cents` to A440 by resampling. Global toggle; per-region bypass for the one you want wobbly. Runs first; the length change is absorbed by the stretch stage. |
| Loudness | On | — | Match integrated LUFS to anchor. |
| Tilt | On | 40% | Octave-band difference curve (region → anchor), capped ±6 dB/band, scaled by amount, applied as a linear-phase FIR. Coarse on purpose — match the tilt and the room, not the fingerprint. |
| Low-end floor | On | — | If region has more sub than anchor: high-pass at anchor's `low_rolloff_hz`. Mono below anchor's `bass_mono_hz`. **Never adds low end** — only aligns the floor downward. |
| Noise floor | On | — | If region's floor is below anchor's: bed a floor under it. The floor is **harvested from the anchor** — its quietest contiguous ≥200 ms, looped with crossfade, levelled to `noise_floor_db`. Never subtracts or cleans. |
| Width | On | 100% | Scale Side relative to Mid per band to match `width_bands`. Capped at 1.5× widening. |
| Density | On, light | — | Not per region: a shared bus compressor on the set output. 2:1, ~30 ms attack, auto release, targeting 2–3 dB GR. |
| Space | **Off** | — | Opt-in. When on: a shared algorithmic reverb send applied **only to regions drier than the anchor**, send level proportional to the `wetness` gap. Wetter-than-anchor regions are never dried. Dry-against-wet is a feature; this module exists for when it isn't. |

Processing order per region: Tuning → (temporal layer) → Loudness → Tilt → Low-end → Noise → Width. Density and Space are bus-level in the UI's playback graph, not rendered into the region files.

---

## 7. Temporal layer

Applies to every region **including the anchor**. Session tempo defaults to the anchor's tempo and follows anchor changes unless pinned.

**7.1 Tempo.** Two paths, and the first is primary:

- **Declared length.** The user states the region is N beats or N bars. Tempo is derived from region length. Exact by construction, no octave ambiguity. Any region the user has trimmed to a musical length should go this way.
- **Detected.** The tracker's BPM with confidence, shown with **½× / 2×** buttons and a feel selector (quarter / 8th / half) for octave errors.

**7.2 Transient snap.** Default on: region playback starts at `first_onset_s`, so a sample never starts before it starts. Per-region **Keep pre-roll** toggle for swells, breaths and pick scrapes that are the character. Per-region **Nudge** in ms. The analyzer cannot tell silence from a pickup; the user can.

**7.3 Warp markers.** A region is not a BPM; it is a list of beat positions, each inter-beat interval with its own stretch ratio. Three tiers, falling back in order:

1. **Manual markers** — user adds, drags, removes. Always win.
2. **Auto beats** — every tracked beat becomes a marker, so drift in off-click material is absorbed interval by interval.
3. **Linear grid** from declared length — for pad-like or sparse material the tracker can't grab.

**7.4 Warp strength.** 0–100%, default **85%**, per-region override. At 100% every beat lands exactly on the grid. At 0% the region is stretched by its average ratio only and the inside breathes. The push and pull of a loose performance is often why it felt good; full quantize is the digital box.

**7.5 Stretch method.** Elastic by default (constant pitch). Repitch optional (pitch follows tempo — the tape-speed vibe). When warp is non-uniform, repitch produces beat-to-beat pitch wobble: force elastic unless the user explicitly opts into wobble via a labelled toggle. Show the **stretch ratio per region** so extreme asks are obvious.

**Engine:** Signalsmith Stretch (MIT) — the same engine as Refonte — called from the Python sidecar via a thin binding. See open question 2.

**7.6 Cuts.** Hard. A 1–2 ms declick ramp at region start and end. No crossfades, no overlap, no tail handling. Retrigger restarts the region from its snap point.

---

## 8. Progression & playback

**8.1 Progression editor.** Enter chords as symbols (`Cm Fm G`) or as Roman numerals against a key (`i iv V` in C minor). Each slot has a length in beats (default 4). Import from a spreadsheet/CSV with columns `chords, roman, key, bpm` — the format of the existing EVS chord progression database — so studied progressions are a preset bank.

**8.2 Per-slot pattern.**

- **Hold** — the region plays through for the slot's length (stretched to fit).
- **Retrigger** — restart the region from its snap point every 1/4, 1/8 or 1/16, with per-step gate, velocity, probability and a beat offset into the region. Same step model as Refonte's sequencer so the two feel related.

**8.3 Transport.** Session BPM, loop, play/stop, position. Playback graph: region players → per-region gain → set bus (Density comp, Space send if on) → output.

**8.4 Export.**

- Full progression render (WAV, session rate).
- Per-slot stems.
- **Kit export:** every conformed region as a WAV plus `patine.json` carrying chord, key, tempo, beat markers and the conform deltas — drop into Refonte or any sampler.
- MIDI file of the progression (chord notes, slot lengths).

---

## 9. UI

Pure-function, same as the plugins: no heavy styling, readouts are readouts, knobs look like knobs. Resizable at launch.

Four panes: **Library** (samples → regions, analysis readouts, audition), **Set** (anchor selector, member list with delta readouts, module toggles/amounts, temporal settings), **Progression** (slots, patterns, candidate drawer per slot with strictness slider and Reroll), **Transport**.

Analysis and render progress are always visible. A region whose render is stale is marked, never silent.

---

## 10. v1 scope fence

In v1: everything above.

Deferred to v2 or later:

- **Centroid matching** (conform toward the set's average rather than an anchor). Anchor only in v1.
- **Derive-from-one mode** — polyphonic re-pitching of a single region's internal notes to produce the whole progression from one sample.
- **Experimental timbre tier** — spectral envelope swap (keep region's harmonic content, impose anchor's cepstral envelope), neural timbre transfer.
- **Source separation for analysis** stays behind a flag; not a supported path until the model footprint is acceptable.
- **Per-region crest/density matching** (v1 uses the bus comp only).
- Live MIDI input, plugin build, cloud sync, multi-set projects, slash-chord/inversion enforcement in strictness (bass note is tracked and shown, not matched).

---

## 11. Open questions for Claude Code

Raise these rather than silently deciding.

1. **Chord model.** Which learned chord model, and its license. Some strong ones are research-only. Recommend a permissively licensed model for root + maj/min with template refinement for extended qualities, and report the weights' size so first-run download is reasonable.
2. **Stretch engine binding.** Does Signalsmith Stretch have a maintained Python binding, or does the sidecar need a thin C extension? Fallback is Rubber Band (GPL, fine for personal use, a licensing question for distribution).
3. **Render cache on anchor change.** Full re-render of all members (simple, slow on big sets) vs. incremental (only modules whose anchor-derived parameters changed). Recommend starting full, profiling, then going incremental if sets of 20+ regions feel sluggish.
4. **Loudness window.** Integrated LUFS over the region vs. short-term. Recommend integrated; revisit if hold-mode regions with long decays match badly.
5. **Noise floor source.** Harvesting the anchor's own floor vs. synthesising a shaped floor from `noise_spectrum`. Recommend harvest; it carries the actual texture. Fall back to synthesis when the anchor has no quiet frames.
6. **Sidecar packaging.** PyInstaller-style bundling of a Python runtime with audio/ML dependencies for both platforms, vs. a first-run environment bootstrap. Which keeps the installer sane and which keeps model loading fast.

---

*Handoff prepared for Claude Code. Scaffold fresh, set up main/develop with the dev-build command, create the Notion tracker, then build in this order: shell + sidecar plumbing + SQLite → ingest + analysis pipeline with library readouts → progression editor + raw playback with hard cuts + temporal layer → tonal conform modules one at a time (Loudness, Tuning, Tilt, Low-end, Noise, Width, Density, Space), each A/B-able against raw → selection ranking, strictness, Reroll → export → polish. The milestone that proves the app is the end of the tonal modules: a progression built from four unrelated records that sounds like one recording that doesn't exist.*
