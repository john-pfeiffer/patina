# Patine sidecar API (v1)

Local HTTP/JSON on `127.0.0.1`, port chosen at launch. Every request except
`GET /health` carries `X-Patine-Token: <token>`. The shell gets the URL and
token from the sidecar's startup line and hands them to the UI through the
Tauri command `sidecar_info`. In a plain browser (UI dev without Tauri) the
UI falls back to `http://127.0.0.1:47621` with token `dev`, which is what
`patine-sidecar --dev` listens on.

Models are the Pydantic classes in `sidecar/src/patine/models.py`; JSON field
names are identical. Times are seconds, absolute inside the source file unless
stated. Pitch classes are note names with sharps (`C#`), qualities are
`maj min 7 m7 maj7 dim sus2 sus4` (analysis) plus `aug dim7 m7b5 6 m6` (typed).

## Meta

| Method | Path | Body → Response |
|---|---|---|
| GET | `/health` | → `SidecarInfo` (no token needed) |
| GET | `/events` | Server-Sent Events, see below |

SSE events (`event:` name, `data:` JSON):

- `sample.updated` → `{sample_id, analysis_status, analysis_progress, message}`
- `sample.removed` → `{sample_id}`
- `region.updated` → `{region_id, sample_id}`
- `set.updated` → `{set_id}` (anchor, settings, membership or bpm changed; renders may be stale)
- `render.updated` → `RenderStatus`
- `job.updated` → `Job`
- `ping` every 15 s

## Library (§5.1)

| Method | Path | Body → Response |
|---|---|---|
| POST | `/samples` | `IngestRequest {paths}` → `{samples: Sample[], jobs: Job[]}`. Directories are walked recursively; known paths return the existing sample. Analysis is queued. |
| GET | `/samples` | → `Sample[]` with `regions` filled |
| GET | `/samples/{id}` | → `Sample` |
| DELETE | `/samples/{id}` | → 204 |
| POST | `/samples/{id}/reanalyze` | → `Job` (§4: re-runs only on explicit request) |
| GET | `/samples/{id}/peaks?bins=800` | → `{bins, duration, peaks: number[]}` mono abs-peak per bin, 0..1 |
| GET | `/samples/{id}/audio` | → `audio/wav` whole source file |
| POST | `/samples/{id}/regions` | `RegionCreate` → `Region` (user region; measured immediately) |
| GET | `/regions/{id}` | → `Region` |
| PATCH | `/regions/{id}` | `RegionPatch` → `Region`. Changing bounds re-measures; any change invalidates that region's renders (clients will see `render.updated` with `stale`). |
| DELETE | `/regions/{id}` | → 204 |
| GET | `/regions/{id}/audio` | → `audio/wav` raw region slice (solo audition, A/B against conformed) |

## Sets (§6–7)

| Method | Path | Body → Response |
|---|---|---|
| POST | `/sets` | `SetCreate` → `Set` |
| GET | `/sets` | → `Set[]` |
| GET | `/sets/{id}` | → `Set` with `members[].region` |
| PATCH | `/sets/{id}` | `SetPatch` → `Set`. Changing `anchor_region_id`, `tonal_settings`, `temporal_settings` or `session_bpm` invalidates renders. Session bpm follows the anchor's tempo unless `bpm_pinned`. |
| DELETE | `/sets/{id}` | → 204 |
| POST | `/sets/{id}/members` | `MemberAdd {region_id}` → `Set`. The first member becomes the anchor when none is set. |
| DELETE | `/sets/{id}/members/{region_id}` | → `Set` |
| GET | `/sets/{id}/deltas` | → `{[region_id]: ConformDeltas}` for every member, computed without rendering |
| GET | `/sets/{id}/renders` | → `RenderStatus[]`, one per member at transpose 0; missing renders are queued as a side effect (shortest first, §2) |
| POST | `/sets/{id}/renders/preview` | `{region_id, transpose_semitones?}` → `RenderStatus` for a region that is not (yet) a member: in-context audition |

## Progressions (§8)

| Method | Path | Body → Response |
|---|---|---|
| POST | `/progressions` | `ProgressionCreate {set_id, name, key?, text?, default_length_beats}` → `Progression`. `text` is `Cm Fm G` or `i iv V` (roman needs `key`); `Fm:2` sets beats. |
| GET | `/progressions?set_id=` | → `Progression[]` |
| GET | `/progressions/{id}` | → `Progression` |
| PATCH | `/progressions/{id}` | `ProgressionPatch` → `Progression`. `text` replaces the slots, keeping region assignments on slots whose chord did not change. |
| DELETE | `/progressions/{id}` | → 204 |
| GET | `/progressions/{id}/renders` | → `{slot_id: RenderStatus}` for every assigned slot at that slot's transposition; missing renders are queued |
| PATCH | `/slots/{id}` | `SlotPatch` → `Slot`. Assigning a `region_id` also adds that region to the progression's set (first member becomes the anchor), so the member list is always every region in use. |
| GET | `/slots/{id}/candidates?strictness=exact\|function\|two_tones&limit=50&unique_samples=false` | → `Candidate[]` ranked by §5.2 score; substitutions rank before transpositions. Each candidate carries `sample_id` and `used_by_slot_idx` (the other slots playing a region of the same sample); `unique_samples=true` hides those |
| POST | `/progressions/{id}/reroll` | `RerollRequest {amount, strictness, seed?, unique_samples?}` → `RerollResponse {progression, warnings}` (§5.5: unlocked slots only). With `unique_samples` no two slots get regions of the same source sample: kept slots claim their samples, changing slots are filled in order excluding claimed samples, and a slot with nothing left falls back to the full pool with a warning |
| POST | `/progressions/import-csv` | `CsvImportRequest {set_id, csv_text}` → `Progression[]`; columns `chords, roman, key, bpm` (header row required, any order) |

## Theory

| Method | Path | Body → Response |
|---|---|---|
| POST | `/theory/parse` | `ParseRequest {text, key?}` → `ParseResponse`. Never 4xx for bad input: `ok=false` with `error`. |

## Renders & export

| Method | Path | Body → Response |
|---|---|---|
| GET | `/renders/{cache_key}.wav` | → `audio/wav` float32, session sample rate. Served only when `status == ready`. |
| POST | `/export` | `ExportRequest {progression_id, kind: full\|stems\|kit\|midi, out_dir, loops, overwrite?}` → `Job`; result `{files: string[], take: int\|null, prefix}`. Renders that are missing are produced first. Every export is a numbered take that never overwrites an earlier one: `<name>_tNN_full.wav`, `stems` one `<name>_tNN_slotNN_<chord>.wav` per slot, `midi` `<name>_tNN.mid`, and `kit` a self-contained folder `<name>_tNN_kit/` holding `NN_<chord>.wav` per slot plus `patine.json`. Full, stems and midi also write `<name>_tNN.json`, a record of the take (chords, regions, sample paths, bounds, transposition, set settings, deltas); the kit carries the same record inside its `patine.json`. `overwrite: true` restores the fixed `<name>_full.wav` names. |

## Jobs

| Method | Path | Body → Response |
|---|---|---|
| GET | `/jobs?status=` | → `Job[]` newest first |
| GET | `/jobs/{id}` | → `Job` |
| POST | `/jobs/{id}/cancel` | → `Job` |

Job kinds: `analyze` (payload `{sample_id}`), `render` (`{cache_key}`), `export`.

## Errors

JSON `{detail: string}` with 400 (bad input), 404, 409 (e.g. export with
unrendered slots when `wait=false`), 500. Theory parsing never errors.
