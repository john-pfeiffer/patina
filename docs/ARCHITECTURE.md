# Patine architecture

Read `HANDOFF.md` first; this file says where each part of it lives.

```
patina/
  apps/desktop/            Tauri 2 shell (Rust) + React/TypeScript UI (Vite)
    src/                   UI — panes, playback engine, API client
    src-tauri/             Rust shell: window, sidecar lifecycle, `sidecar_info` command
  sidecar/                 Python engine, package `patine` (uv-managed)
    src/patine/
      cli.py               `patine-sidecar` entry point, startup handshake
      server.py            FastAPI app, token middleware, SSE, static renders
      api/                 routers: library, sets, progressions, theory, renders, export, jobs
      db.py  models.py     SQLite schema + Pydantic models (handoff §3)
      theory.py            chords, keys, roman numerals, strictness (§5.3, §8.1)
      selection.py         candidate ranking + Reroll (§5.2–5.5)
      jobs.py events.py    background worker, progress events
      analysis/            §4 pipeline
      temporal/            §7 tempo, warp, stretch (Signalsmith via python-stretch)
      conform/             §6 modules + renderer + render cache
      export/              §8.4
    tests/
  scripts/                 dev.mjs (one-command dev), build-sidecar.mjs (PyInstaller)
  docs/                    HANDOFF.md, API.md, DECISIONS.md, this file
```

## Process model

The shell spawns the sidecar on launch and kills it on exit. The sidecar binds
`127.0.0.1:0`, prints one line

    PATINE_READY {"url":"http://127.0.0.1:PORT","token":"…"}

and the shell exposes that to the UI via `invoke("sidecar_info")`. In debug
builds the shell runs `uv run --project <repo>/sidecar patine-sidecar --dev`
(fixed port 47621, token `dev`, auto-reload) so `pnpm dev` is the whole
workflow. In release builds it runs the PyInstaller binary bundled as a Tauri
`externalBin`. `PATINE_SIDECAR_URL`/`PATINE_SIDECAR_TOKEN` make the shell attach
to an already-running sidecar instead of spawning one.

## Offline conform, online play

The sidecar renders each conformed region to a float32 WAV in the render cache.
The UI loads those into Web Audio (Tone.js) and schedules them with a lookahead
clock. The UI never waits on DSP: it plays the last good render and marks a
region *stale* until the new one lands (`render.updated` SSE events).

Render cache key = blake2b of (sample hash, region bounds, user overrides,
temporal settings, session bpm, tonal settings, anchor fingerprint, transpose,
`RENDER_VERSION`). The anchor fingerprint is a hash of the anchor's metrics,
tuning and identity, so switching anchor invalidates every member. The render
queue processes members in playback order, shortest first.

## Data

SQLite at `<data dir>/library.sqlite3` (WAL). Renders at `<data dir>/renders/`.
Model weights at `<data dir>/models/` (fetched on first use, never bundled).
Data dir: macOS `~/Library/Application Support/Patine`, Windows `%APPDATA%\Patine`,
Linux `~/.local/share/patine`; `--dev` uses `<repo>/.patine-dev`.

## Contracts between layers

- `analysis.pipeline.analyze_file(path, progress) -> AnalysisResult`
- `analysis.pipeline.measure_region(y, sr, start, end, ctx) -> RegionResult`
- `temporal.grid.derive_region_tempo(...) -> RegionTempo`
- `temporal.warp.build_warp_plan(...) -> WarpPlan`
- `temporal.stretch.stretch_plan(y, sr, plan, method, transpose) -> ndarray`
- `conform.renderer.cache_key(req)`, `compute_deltas(req)`, `render_region(req, out)`
- `export.exporter.export_progression(job) -> [Path]`

The API layer (`api/*`) is the only code that touches both the database and
these modules. Analysis/conform/export modules never import the database.
