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

## Branches and releases

- `main` is the default branch and the released, known-good state. Until v1
  it stays at the initial commit; nothing merges into it before then.
- `develop` is where all work lands. Feature work (`feature/*`, `fix/*`,
  `chore/*`, and Claude Code's `claude/*` branches) branches **from
  `develop`** and comes back by squash-merged pull request **into `develop`**
  with CI green. Two habits, because `main` is the default: set the base to
  `develop` when opening a PR (the `Release source` check fails a PR into
  `main` from anywhere else), and start Claude Code sessions from `develop`.
- Test builds are the CI artefacts `Patine-dev-<sha7>-<OS>` from every push
  to `develop`. A tracker item merged to `develop` is "Fixed in develop" with
  that build id.
- Releases: open a PR `develop → main` (merge commit), then tag the merge
  commit `vX.Y.Z` on `main`. The `Release` workflow builds the installers and
  publishes a GitHub Release; a tag with a suffix (`v1.1.0-beta.1`) publishes
  a pre-release. Hotfixes branch from `main` as `hotfix/*`, PR into `main`,
  then PR `main → develop`. Rulesets enforcing this are in
  `.github/rulesets/`.
- Versions: `pnpm version:bump 1.0.0` updates every file that declares the
  version (CI and the release workflow check they agree); commit that on
  `develop` before the release PR.

```sh
pnpm build      # local: PyInstaller sidecar → Tauri bundle (dmg / nsis / appimage)
```

## Layout

```
apps/desktop   Tauri 2 shell (Rust) + React/TypeScript UI
sidecar        Python engine: SQLite library, analysis, conform renderer, render cache, export
docs           handoff, architecture, API, decisions
scripts        dev.mjs, build-sidecar.mjs
```
