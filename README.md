# Patine

*Élan Vital Studios — chord-sampling playground, desktop app.*

Patine builds chord progressions out of samples. Drop in audio, it finds the
harmonically stable regions, tags them with chord, key, tempo and tonal
measurements, and lets you assign them to the slots of a progression. The
conform layer then makes regions pulled from different records read as one
recording — tuning, level, tilt, low-end floor, noise floor, width — without
touching what makes them different music.

**Conform the recording, not the music.** Cuts are hard.

Read `docs/HANDOFF.md` for the full spec, `docs/ARCHITECTURE.md` for where
things live, `docs/API.md` for the sidecar contract and `docs/DECISIONS.md`
for the open questions and what was decided.

## Develop

Prerequisites: Node 22 + pnpm, Rust (rustup), [uv](https://docs.astral.sh/uv/),
and the Tauri platform prerequisites for your OS. ffmpeg on PATH extends the
decodable formats.

```sh
pnpm bootstrap  # uv sync + pnpm install
pnpm dev        # desktop shell with UI hot reload; the shell runs the sidecar from source
pnpm test       # sidecar pytest + UI vitest
pnpm lint       # ruff + tsc
```

Useful variants:

```sh
pnpm dev:sidecar   # just the engine on http://127.0.0.1:47621 (token "dev")
pnpm dev:ui        # just the UI in a browser on http://localhost:1420 (talks to the dev sidecar)
```

## Release build

```sh
pnpm build      # PyInstaller sidecar → Tauri bundle (dmg / nsis / appimage)
```

CI builds unsigned dev artefacts for macOS, Windows and Linux on every push
to `develop` and `main`.

## Layout

```
apps/desktop   Tauri 2 shell (Rust) + React/TypeScript UI
sidecar        Python engine: SQLite library, analysis, conform renderer, render cache, export
docs           handoff, architecture, API, decisions
scripts        dev.mjs, build-sidecar.mjs
```
