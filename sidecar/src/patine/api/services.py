"""Orchestration shared by the routers: ingest + analysis jobs, render
requests + queue, invalidation, export jobs. Routers stay thin."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .. import ANALYSIS_VERSION
from ..audio import file_hash, is_audio_file, is_hidden, iter_audio_files, probe
from ..db import now_iso
from ..jobs import JobContext
from ..models import LibraryRoot, Region, RenderStatus, Sample, Set, Settings, Slot
from ..repo import NotFound

if TYPE_CHECKING:
    from ..server import AppState

log = logging.getLogger(__name__)


# --- ingest & analysis ------------------------------------------------------------

def expand_paths(paths: list[str]) -> list[str]:
    out: list[str] = []
    for p in paths:
        path = Path(p).expanduser()
        if path.is_dir():
            for child in iter_audio_files(path):
                out.append(str(child.resolve()))
        elif path.is_file() and is_audio_file(path) and not is_hidden(path):
            out.append(str(path.resolve()))
    seen: set[str] = set()
    return [p for p in out if not (p in seen or seen.add(p))]


def ingest(state: AppState, paths: list[str]) -> tuple[list[Sample], list[Any]]:
    samples: list[Sample] = []
    jobs = []
    for p in expand_paths(paths):
        existing = state.repo.sample_by_path(p)
        if existing is not None:
            samples.append(existing)
            if _analysis_is_stale(existing):
                jobs.append(queue_analysis(state, existing.id, priority=7))
            continue
        try:
            duration, sr, ch = probe(p)
        except Exception as e:  # noqa: BLE001
            log.warning("skipping %s: %s", p, e)
            continue
        sample = state.repo.insert_sample(p, file_hash(p), duration, sr, ch)
        samples.append(sample)
        jobs.append(queue_analysis(state, sample.id))
    return samples, jobs


def _analysis_is_stale(sample: Sample) -> bool:
    return sample.analysis_status in ("done", "failed") and (
        sample.analysis_version is None or sample.analysis_version < ANALYSIS_VERSION
    )


def queue_analysis(state: AppState, sample_id: str, *, priority: int = 5):
    state.repo.update_sample(sample_id, analysis_status="pending", analysis_progress=0.0, analysis_error=None)
    publish_sample(state, sample_id, "queued")
    return state.worker.submit("analyze", {"sample_id": sample_id}, priority=priority, dedupe_key=sample_id)


def analysis_options(state: AppState):
    """Pipeline options from the stored settings (Preferences › Analysis)."""
    from ..analysis.pipeline import AnalysisOptions

    s = state.repo.get_settings()
    return AnalysisOptions(hpss=s.analysis_hpss, window_beats=s.analysis_window_beats, fallback=s.analysis_fallback)


def apply_settings(state: AppState) -> Settings:
    """Make the worker follow the stored settings (pause/resume analysis)."""
    s = state.repo.get_settings()
    if s.analysis_paused:
        state.worker.pause("analyze")
    else:
        state.worker.resume("analyze")
    return s


# --- library roots ----------------------------------------------------------------

def add_root(state: AppState, path: str, name: str | None = None) -> tuple[LibraryRoot, Any]:
    folder = Path(path).expanduser()
    if not folder.is_dir():
        raise ValueError(f"not a folder: {path}")
    resolved = str(folder.resolve())
    existing = state.repo.root_by_path(resolved)
    root = existing or state.repo.insert_root(resolved, name or folder.name or resolved)
    job = queue_scan(state, root.id)
    return root, job


def queue_scan(state: AppState, root_id: str, *, priority: int = 4):
    return state.worker.submit("scan", {"root_id": root_id}, priority=priority, dedupe_key=f"scan:{root_id}")


def scan_all_roots(state: AppState, *, priority: int = 9) -> list[Any]:
    return [queue_scan(state, r.id, priority=priority) for r in state.repo.list_roots()]


def publish_root(state: AppState, root_id: str, message: str = "", scanned: int | None = None, total: int | None = None) -> None:
    try:
        root = state.repo.get_root(root_id)
    except NotFound:
        return
    state.events.publish("root.updated", {
        "root_id": root.id, "status": root.status, "file_count": root.file_count, "error": root.error,
        "last_scan_at": root.last_scan_at, "message": message, "scanned": scanned, "total": total,
    })


def run_scan_job(state: AppState, job: JobContext) -> dict[str, Any]:
    """Walk a root: new files are added and queued for analysis, files whose
    size or mtime changed are re-queued, files that vanished are marked
    `missing` (and un-marked when they come back). Nothing is deleted."""
    root_id = job.payload["root_id"]
    root = state.repo.get_root(root_id)
    state.repo.update_root(root_id, status="scanning", error=None)
    publish_root(state, root_id, "scanning")
    folder = Path(root.path)
    added = changed = missing = unchanged = 0
    try:
        if not folder.is_dir():
            raise FileNotFoundError(f"folder not found: {root.path}")
        known = {s.path: s for s in state.repo.samples_for_root(root_id)}
        seen: set[str] = set()
        files = list(iter_audio_files(folder))
        for i, f in enumerate(files):
            if i % 25 == 0:
                job.progress(0.05 + 0.9 * i / max(1, len(files)), f"{i}/{len(files)} files")
                publish_root(state, root_id, f"{i}/{len(files)} files", i, len(files))
            path = str(f.resolve())
            seen.add(path)
            try:
                st = f.stat()
            except OSError:
                continue
            rel = str(f.relative_to(folder))
            sample = known.get(path) or state.repo.sample_by_path(path)
            if sample is None:
                try:
                    duration, sr, ch = probe(path)
                except Exception as e:  # noqa: BLE001 - one undecodable file is not an error for the scan
                    log.warning("scan %s: skipping %s: %s", root.name, rel, e)
                    continue
                state.repo.insert_sample(path, file_hash(path), duration, sr, ch,
                                         root_id=root_id, rel_path=rel, mtime=st.st_mtime, size=st.st_size)
                sample = state.repo.sample_by_path(path)
                assert sample is not None
                queue_analysis(state, sample.id, priority=8)
                added += 1
                continue
            fields: dict[str, Any] = {}
            if sample.root_id != root_id or sample.rel_path != rel:
                fields.update(root_id=root_id, rel_path=rel)
            file_changed = sample.mtime is None or sample.size is None or abs(sample.mtime - st.st_mtime) > 1e-6 or sample.size != st.st_size
            if file_changed:
                fields.update(mtime=st.st_mtime, size=st.st_size, hash=file_hash(path))
            if sample.analysis_status == "missing":
                fields["analysis_status"] = "done" if sample.analysis_version else "pending"
            if fields:
                state.repo.update_sample(sample.id, **fields)
            if file_changed and sample.mtime is not None:
                queue_analysis(state, sample.id, priority=8)
                changed += 1
            elif file_changed:
                queue_analysis(state, sample.id, priority=8)   # scanned before provenance existed: analyse at v2
                changed += 1
            elif _analysis_is_stale(sample):
                queue_analysis(state, sample.id, priority=9)
                unchanged += 1
            else:
                unchanged += 1
        for path, sample in known.items():
            if path not in seen and sample.analysis_status != "missing":
                state.repo.update_sample(sample.id, analysis_status="missing")
                publish_sample(state, sample.id, "missing")
                missing += 1
        state.repo.update_root(root_id, status="idle", file_count=len(seen), last_scan_at=now_iso())
    except Exception as e:
        state.repo.update_root(root_id, status="error", error=f"{type(e).__name__}: {e}")
        publish_root(state, root_id, "error")
        raise
    publish_root(state, root_id, "scanned")
    return {"added": added, "changed": changed, "missing": missing, "unchanged": unchanged, "files": len(seen)}


def requeue_stale_analyses(state: AppState) -> int:
    """Queue every sample analysed by an older pipeline, behind fresh work
    (priority 9). Called at startup; region ids survive the re-run
    (`Repo.replace_analysis_regions`)."""
    stale = state.repo.stale_sample_ids(ANALYSIS_VERSION)
    for sid in stale:
        queue_analysis(state, sid, priority=9)
    if stale:
        log.info("queued %d sample(s) for reanalysis (analysis v%d)", len(stale), ANALYSIS_VERSION)
    return len(stale)


def publish_sample(state: AppState, sample_id: str, message: str = "") -> None:
    try:
        s = state.repo.get_sample(sample_id)
    except NotFound:
        return
    state.events.publish(
        "sample.updated",
        {"sample_id": s.id, "analysis_status": s.analysis_status, "analysis_progress": s.analysis_progress, "message": message},
    )


def _region_fields(r: Any) -> dict[str, Any]:
    return {
        "start_s": r.start_s,
        "end_s": r.end_s,
        "chord_root": r.chord_root,
        "chord_quality": r.chord_quality,
        "bass_note": r.bass_note,
        "chord_confidence": r.chord_confidence,
        "key_context": r.key_context,
        "beats": list(r.beats),
        "first_onset_s": r.first_onset_s,
        "busyness": r.busyness,
        "register_centroid_hz": r.register_centroid_hz,
        "metrics": r.metrics.model_dump(),
    }


def run_analysis_job(state: AppState, job: JobContext) -> dict[str, Any]:
    from ..analysis.pipeline import analyze_file

    sample_id = job.payload["sample_id"]
    sample = state.repo.get_sample(sample_id)
    state.repo.update_sample(sample_id, analysis_status="running", analysis_progress=0.0)
    publish_sample(state, sample_id, "analysing")

    def progress(frac: float, msg: str) -> None:
        state.repo.update_sample(sample_id, analysis_progress=frac)
        publish_sample(state, sample_id, msg)
        job.progress(frac, msg)

    try:
        result = analyze_file(sample.path, progress=progress, options=analysis_options(state))
    except Exception as e:
        state.repo.update_sample(sample_id, analysis_status="failed", analysis_error=f"{type(e).__name__}: {e}")
        publish_sample(state, sample_id, "failed")
        raise
    state.repo.update_sample(
        sample_id,
        duration=result.duration,
        sample_rate=result.sample_rate,
        channels=result.channels,
        tuning_cents=result.tuning_cents,
        detected_bpm=result.detected_bpm,
        bpm_confidence=result.bpm_confidence,
        beats=list(result.beats),
        downbeats=list(result.downbeats),
        onsets=[float(o) for o in result.context.onsets],
        key_estimate=result.key_estimate,
        key_confidence=result.key_confidence,
        metrics=result.metrics.model_dump(),
        analysis_status="done",
        analysis_error=None,
        analysis_version=ANALYSIS_VERSION,
        analysis_progress=1.0,
    )
    regions = state.repo.replace_analysis_regions(sample_id, [_region_fields(r) for r in result.regions])
    publish_sample(state, sample_id, "done")
    # Kept ids may carry new bounds and metrics: their renders re-key lazily,
    # and the sets that hold them need to know.
    touched = {sid for r in regions for sid in state.repo.sets_containing(r.id)}
    for set_id in touched:
        invalidate_set(state, set_id)
    return {"regions": len(regions), "recovered": sum(1 for r in result.regions if r.recovered)}


def remeasure_region(state: AppState, region: Region, *, chord_root: str | None = None, chord_quality: str | None = None) -> Region:
    """Re-measure a region whose bounds changed (handoff §4: context from the
    sample's stored whole-file analysis, no file re-run)."""
    from ..analysis.pipeline import AnalysisContext, measure_region
    from ..audio import load_audio

    sample = state.repo.get_sample(region.sample_id)
    y, sr = load_audio(sample.path)
    ctx = AnalysisContext(
        sample_rate=sr,
        tuning_cents=sample.tuning_cents or 0.0,
        beats=list(sample.beats),
        downbeats=list(sample.downbeats),
        onsets=list(sample.onsets),
        key_estimate=sample.key_estimate,
    )
    res = measure_region(y, sr, region.start_s, region.end_s, ctx, chord_root=chord_root, chord_quality=chord_quality)
    fields = _region_fields(res)
    if chord_root is not None:
        fields["chord_root"] = chord_root
    if chord_quality is not None:
        fields["chord_quality"] = chord_quality
    return state.repo.update_region(region.id, **fields)


# --- renders -------------------------------------------------------------------------

def _anchor_profile(state: AppState, set_: Set):
    from ..conform.renderer import AnchorProfile

    if not set_.anchor_region_id:
        return None
    try:
        anchor = state.repo.get_region(set_.anchor_region_id)
    except NotFound:
        return None
    sample = state.repo.get_sample(anchor.sample_id)
    if anchor.metrics is None:
        return None
    return AnchorProfile(
        region_id=anchor.id,
        metrics=anchor.metrics,
        tuning_cents=sample.tuning_cents or 0.0,
        sample_path=sample.path,
        sample_hash=sample.hash,
        start_s=anchor.start_s,
        end_s=anchor.end_s,
    )


def effective_session_bpm(state: AppState, set_: Set) -> float:
    """§7: session tempo defaults to the anchor's tempo and follows anchor
    changes unless pinned."""
    if set_.bpm_pinned and set_.session_bpm:
        return set_.session_bpm
    if set_.anchor_region_id:
        try:
            from ..temporal.grid import derive_region_tempo

            anchor = state.repo.get_region(set_.anchor_region_id)
            sample = state.repo.get_sample(anchor.sample_id)
            tempo = derive_region_tempo(anchor, sample.detected_bpm, sample.bpm_confidence, sample.beats,
                                        transient_snap=set_.temporal_settings.transient_snap)
            if tempo.bpm > 0:
                if set_.session_bpm != tempo.bpm:
                    state.repo.update_set(set_.id, session_bpm=tempo.bpm)
                return tempo.bpm
        except Exception as e:  # noqa: BLE001
            log.warning("anchor tempo unavailable: %s", e)
    return set_.session_bpm or 120.0


def build_render_request(state: AppState, set_: Set, region: Region, transpose: int = 0):
    from ..conform.renderer import RenderRequest

    sample = state.repo.get_sample(region.sample_id)
    return RenderRequest(
        region=region,
        sample_path=sample.path,
        sample_hash=sample.hash,
        sample_tuning_cents=sample.tuning_cents or 0.0,
        sample_bpm=sample.detected_bpm,
        sample_bpm_confidence=sample.bpm_confidence,
        sample_beats=list(sample.beats),
        anchor=_anchor_profile(state, set_),
        tonal=set_.tonal_settings,
        temporal=set_.temporal_settings,
        session_bpm=effective_session_bpm(state, set_),
        transpose_semitones=transpose,
        is_anchor=(region.id == set_.anchor_region_id),
    )


def ensure_render(state: AppState, set_: Set, region: Region, transpose: int = 0, *, priority: int = 5) -> RenderStatus:
    """Return the render status for (set, region, transpose), queueing the
    render when the cache has no ready entry for the current key. While it is
    pending, the status carries `stale=True` and the URL of the last good
    render for this region in this set, so the UI keeps playing."""
    from ..conform.renderer import cache_key

    if region.metrics is None:
        return RenderStatus(cache_key="", region_id=region.id, set_id=set_.id, status="failed",
                            error="region not analysed yet")
    req = build_render_request(state, set_, region, transpose)
    key = cache_key(req)
    existing = state.repo.get_render(key)
    if existing is not None and existing.status == "ready" and Path(state.repo.render_path(key) or "").exists():
        return existing
    if existing is None or existing.status == "failed":
        path = str(state.paths.renders / f"{key}.wav")
        params = {"region_id": region.id, "set_id": set_.id, "transpose": transpose, "session_bpm": req.session_bpm}
        existing = state.repo.upsert_render(key, region.id, set_.id, path, "queued", params)
        # Shortest first (§2) so the progression becomes playable again fast.
        prio = priority + min(9, int(region.duration_s))
        state.worker.submit("render", {"cache_key": key, "region_id": region.id, "set_id": set_.id, "transpose": transpose},
                            priority=prio, dedupe_key=key)
        state.events.publish("render.updated", existing.model_dump())
    last = state.repo.latest_ready_render(set_.id, region.id)
    out = existing.model_copy()
    out.stale = True
    if last is not None:
        out.url = last.url
        out.deltas = out.deltas or last.deltas
    return out


def run_render_job(state: AppState, job: JobContext) -> dict[str, Any]:
    from ..conform.renderer import cache_key, render_region

    key = job.payload["cache_key"]
    region = state.repo.get_region(job.payload["region_id"])
    set_ = state.repo.get_set(job.payload["set_id"])
    transpose = int(job.payload.get("transpose", 0))
    req = build_render_request(state, set_, region, transpose)
    if cache_key(req) != key:
        # Settings moved on while this was queued; the newer key has its own job.
        state.repo.upsert_render(key, region.id, set_.id, "", "failed", {}, error="superseded")
        return {"superseded": True}
    path = state.paths.renders / f"{key}.wav"
    state.repo.upsert_render(key, region.id, set_.id, str(path), "rendering", job.payload)
    state.events.publish("render.updated", state.repo.get_render(key).model_dump())  # type: ignore[union-attr]
    tmp = path.with_suffix(".part.wav")
    try:
        deltas = render_region(req, str(tmp), progress=job.progress)
        os.replace(tmp, path)
    except Exception as e:
        tmp.unlink(missing_ok=True)
        st = state.repo.upsert_render(key, region.id, set_.id, str(path), "failed", job.payload, error=f"{type(e).__name__}: {e}")
        state.events.publish("render.updated", st.model_dump())
        raise
    st = state.repo.upsert_render(key, region.id, set_.id, str(path), "ready", job.payload, deltas=deltas.model_dump())
    state.events.publish("render.updated", st.model_dump())
    return {"cache_key": key}


def set_renders(state: AppState, set_: Set) -> list[RenderStatus]:
    members = sorted(set_.members, key=lambda m: (m.region.duration_s, m.position))
    return [ensure_render(state, set_, m.region) for m in members]


def progression_renders(state: AppState, prog_id: str) -> dict[str, RenderStatus]:
    prog = state.repo.get_progression(prog_id)
    set_ = state.repo.get_set(prog.set_id)
    out: dict[str, RenderStatus] = {}
    slots: list[Slot] = [s for s in prog.slots if s.region_id]
    slots.sort(key=lambda s: s.idx)
    for s in slots:
        try:
            region = state.repo.get_region(s.region_id)  # type: ignore[arg-type]
        except NotFound:
            continue
        out[s.id] = ensure_render(state, set_, region, s.transpose_semitones)
    return out


def set_deltas(state: AppState, set_: Set) -> dict[str, Any]:
    from ..conform.renderer import compute_deltas

    out: dict[str, Any] = {}
    for m in set_.members:
        if m.region.metrics is None:
            continue
        try:
            out[m.region.id] = compute_deltas(build_render_request(state, set_, m.region)).model_dump()
        except Exception as e:  # noqa: BLE001
            log.warning("deltas for %s: %s", m.region.id, e)
    return out


def invalidate_set(state: AppState, set_id: str) -> None:
    """Settings or membership changed: tell clients; renders are re-keyed lazily."""
    state.events.publish("set.updated", {"set_id": set_id})


def invalidate_region(state: AppState, region_id: str) -> None:
    for set_id in state.repo.sets_containing(region_id):
        invalidate_set(state, set_id)


# --- export ---------------------------------------------------------------------------

def run_export_job(state: AppState, job: JobContext) -> dict[str, Any]:
    from ..conform.renderer import cache_key, render_region
    from ..export.exporter import ExportJob, export_take

    prog = state.repo.get_progression(job.payload["progression_id"])
    set_ = state.repo.get_set(prog.set_id)
    kind = job.payload["kind"]
    out_dir = Path(job.payload["out_dir"]).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    regions: dict[str, Region] = {}
    renders: dict[str, str] = {}
    deltas: dict[str, dict] = {}
    sample_paths: dict[str, str] = {}
    assigned = [s for s in prog.slots if s.region_id]
    for i, slot in enumerate(assigned):
        region = state.repo.get_region(slot.region_id)  # type: ignore[arg-type]
        regions[region.id] = region
        if region.sample_id not in sample_paths:
            try:
                sample_paths[region.sample_id] = state.repo.get_sample(region.sample_id).path
            except Exception:  # noqa: BLE001 - the record is informational
                pass
        if kind == "midi":
            continue
        job.progress(0.1 + 0.6 * i / max(1, len(assigned)), f"rendering slot {slot.idx + 1}")
        req = build_render_request(state, set_, region, slot.transpose_semitones)
        key = cache_key(req)
        path = state.paths.renders / f"{key}.wav"
        st = state.repo.get_render(key)
        if st is None or st.status != "ready" or not path.exists():
            d = render_region(req, str(path))
            st = state.repo.upsert_render(key, region.id, set_.id, str(path), "ready", {"export": True}, deltas=d.model_dump())
            state.events.publish("render.updated", st.model_dump())
        # Slots may share a region with different transpositions; key per slot.
        renders[slot.id] = str(path)
        if st.deltas:
            deltas[slot.id] = st.deltas.model_dump()
    job.progress(0.75, "writing files")
    ej = ExportJob(
        kind=kind, progression=prog, set_=set_, regions=regions, renders=renders, deltas=deltas,
        out_dir=out_dir, loops=int(job.payload.get("loops", 1)),
        sample_rate=int(job.payload.get("sample_rate", 48000)),
        sample_paths=sample_paths, overwrite=bool(job.payload.get("overwrite", False)),
    )
    result = export_take(ej)
    return {"files": [str(f) for f in result.files], "take": result.take, "prefix": result.prefix}


def register_job_handlers(state: AppState) -> None:
    state.worker.register("scan", lambda job: run_scan_job(state, job))
    state.worker.register("analyze", lambda job: run_analysis_job(state, job))
    state.worker.register("render", lambda job: run_render_job(state, job))
    state.worker.register("export", lambda job: run_export_job(state, job))
