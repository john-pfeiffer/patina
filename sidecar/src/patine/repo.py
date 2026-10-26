"""Row ↔ model mapping and the CRUD the API layer needs. Nothing here knows
about audio; nothing in analysis/conform/export knows about this."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from .db import Database, dumps, loads, new_id, now_iso
from .models import (
    ConformDeltas,
    Pattern,
    Progression,
    Region,
    RenderStatus,
    Sample,
    Set,
    SetMember,
    Slot,
    TemporalSettings,
    TonalMetrics,
    TonalSettings,
    UserOverrides,
)


class NotFound(KeyError):
    pass


def _metrics(text: str | None) -> TonalMetrics | None:
    d = loads(text)
    return TonalMetrics.model_validate(d) if d else None


def row_to_region(r: sqlite3.Row) -> Region:
    return Region(
        id=r["id"],
        sample_id=r["sample_id"],
        position=r["position"],
        start_s=r["start_s"],
        end_s=r["end_s"],
        chord_root=r["chord_root"],
        chord_quality=r["chord_quality"],
        bass_note=r["bass_note"],
        chord_confidence=r["chord_confidence"],
        key_context=r["key_context"],
        beats=loads(r["beats"], []),
        first_onset_s=r["first_onset_s"],
        declared_length_beats=r["declared_length_beats"],
        busyness=r["busyness"],
        register_centroid_hz=r["register_centroid_hz"],
        metrics=_metrics(r["metrics"]),
        user_overrides=UserOverrides.model_validate(loads(r["user_overrides"], {})),
        source=r["source"],
    )


def row_to_sample(r: sqlite3.Row, regions: list[Region] | None = None) -> Sample:
    return Sample(
        id=r["id"],
        path=r["path"],
        hash=r["hash"],
        name=r["name"],
        duration=r["duration"],
        sample_rate=r["sample_rate"],
        channels=r["channels"],
        added_at=r["added_at"],
        tuning_cents=r["tuning_cents"],
        detected_bpm=r["detected_bpm"],
        bpm_confidence=r["bpm_confidence"],
        beats=loads(r["beats"], []),
        downbeats=loads(r["downbeats"], []),
        onsets=loads(r["onsets"], []) if "onsets" in r.keys() else [],
        key_estimate=r["key_estimate"],
        key_confidence=r["key_confidence"],
        metrics=_metrics(r["metrics"]),
        analysis_status=r["analysis_status"],
        analysis_error=r["analysis_error"],
        analysis_version=r["analysis_version"],
        analysis_progress=r["analysis_progress"],
        regions=regions or [],
    )


def row_to_slot(r: sqlite3.Row) -> Slot:
    return Slot(
        id=r["id"],
        progression_id=r["progression_id"],
        idx=r["idx"],
        chord_symbol=r["chord_symbol"],
        roman=r["roman"],
        chord_root=r["chord_root"],
        chord_quality=r["chord_quality"],
        bass_note=r["bass_note"],
        length_beats=r["length_beats"],
        region_id=r["region_id"],
        transpose_semitones=r["transpose_semitones"],
        locked=bool(r["locked"]),
        pattern=Pattern.model_validate(loads(r["pattern"], {})),
    )


def row_to_render(r: sqlite3.Row, *, stale: bool = False) -> RenderStatus:
    deltas = loads(r["deltas"])
    return RenderStatus(
        cache_key=r["cache_key"],
        region_id=r["region_id"],
        set_id=r["set_id"],
        status=r["status"],
        url=f"/renders/{r['cache_key']}.wav" if r["status"] == "ready" else None,
        error=r["error"],
        deltas=ConformDeltas.model_validate(deltas) if deltas else None,
        stale=stale,
    )


def _json_fields(fields: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in fields.items():
        if hasattr(v, "model_dump"):
            v = v.model_dump()
        if isinstance(v, list | dict):
            v = dumps(v)
        if isinstance(v, bool):
            v = int(v)
        out[k] = v
    return out


class Repo:
    def __init__(self, db: Database):
        self.db = db

    # --- samples ------------------------------------------------------------------

    def insert_sample(self, path: str, file_hash: str, duration: float, sample_rate: int, channels: int) -> Sample:
        sid = new_id()
        with self.db.tx() as conn:
            conn.execute(
                "INSERT INTO samples (id, path, hash, name, duration, sample_rate, channels, added_at) VALUES (?,?,?,?,?,?,?,?)",
                (sid, path, file_hash, Path(path).name, duration, sample_rate, channels, now_iso()),
            )
        return self.get_sample(sid)

    def sample_by_path(self, path: str) -> Sample | None:
        r = self.db.one("SELECT * FROM samples WHERE path = ?", (path,))
        return self.get_sample(r["id"]) if r else None

    def get_sample(self, sample_id: str) -> Sample:
        r = self.db.one("SELECT * FROM samples WHERE id = ?", (sample_id,))
        if r is None:
            raise NotFound(sample_id)
        return row_to_sample(r, self.regions_for_sample(sample_id))

    def list_samples(self) -> list[Sample]:
        rows = self.db.all("SELECT * FROM samples ORDER BY added_at, name")
        regions = self.db.all("SELECT * FROM regions ORDER BY sample_id, position, start_s")
        by_sample: dict[str, list[Region]] = {}
        for rr in regions:
            by_sample.setdefault(rr["sample_id"], []).append(row_to_region(rr))
        return [row_to_sample(r, by_sample.get(r["id"], [])) for r in rows]

    def update_sample(self, sample_id: str, **fields: Any) -> Sample:
        f = _json_fields(fields)
        cols = ", ".join(f"{k} = ?" for k in f)
        with self.db.tx() as conn:
            conn.execute(f"UPDATE samples SET {cols} WHERE id = ?", (*f.values(), sample_id))
        return self.get_sample(sample_id)

    def delete_sample(self, sample_id: str) -> None:
        with self.db.tx() as conn:
            conn.execute("DELETE FROM samples WHERE id = ?", (sample_id,))

    def stale_sample_ids(self, current_version: int) -> list[str]:
        """Samples analysed by an older pipeline (or never versioned) that are
        not already queued or running."""
        rows = self.db.all(
            "SELECT id FROM samples WHERE analysis_status IN ('done', 'failed') "
            "AND (analysis_version IS NULL OR analysis_version < ?) ORDER BY added_at",
            (current_version,),
        )
        return [r["id"] for r in rows]

    # --- regions -------------------------------------------------------------------

    def regions_for_sample(self, sample_id: str) -> list[Region]:
        rows = self.db.all("SELECT * FROM regions WHERE sample_id = ? ORDER BY position, start_s", (sample_id,))
        return [row_to_region(r) for r in rows]

    def get_region(self, region_id: str) -> Region:
        r = self.db.one("SELECT * FROM regions WHERE id = ?", (region_id,))
        if r is None:
            raise NotFound(region_id)
        return row_to_region(r)

    def insert_region(self, sample_id: str, fields: dict[str, Any], *, source: str = "analysis") -> Region:
        rid = new_id()
        f = _json_fields({k: v for k, v in fields.items() if k != "id"})
        f.setdefault("beats", "[]")
        f.setdefault("user_overrides", "{}")
        cols = ["id", "sample_id", "source", *f.keys()]
        marks = ",".join("?" for _ in cols)
        with self.db.tx() as conn:
            conn.execute(f"INSERT INTO regions ({', '.join(cols)}) VALUES ({marks})", (rid, sample_id, source, *f.values()))
        return self.get_region(rid)

    def replace_analysis_regions(
        self, sample_id: str, regions: list[dict[str, Any]], *, match_iou: float = 0.5, user_iou: float = 0.8
    ) -> list[Region]:
        """Replace a sample's analysis regions with a new analysis, keeping ids
        where the new and old regions overlap (temporal IoU >= `match_iou`,
        greedy by overlap) so set membership, the anchor and slot assignments
        survive a reanalysis; a matched region keeps its user overrides and
        declared length. Old regions nothing matches are deleted; new regions
        that cover a span a user region already owns (IoU >= `user_iou`) are
        not inserted. Returns the sample's analysis regions in order."""
        old = [r for r in self.regions_for_sample(sample_id) if r.source == "analysis"]
        users = [r for r in self.regions_for_sample(sample_id) if r.source == "user"]

        def iou(a0: float, a1: float, b0: float, b1: float) -> float:
            inter = max(0.0, min(a1, b1) - max(a0, b0))
            union = max(a1, b1) - min(a0, b0)
            return inter / union if union > 0 else 0.0

        pairs = sorted(
            ((iou(f["start_s"], f["end_s"], o.start_s, o.end_s), i, j) for i, f in enumerate(regions) for j, o in enumerate(old)),
            reverse=True,
        )
        matched_new: dict[int, int] = {}
        matched_old: set[int] = set()
        for score, i, j in pairs:
            if score < match_iou:
                break
            if i in matched_new or j in matched_old:
                continue
            matched_new[i] = j
            matched_old.add(j)
        keep_cols = ("user_overrides", "declared_length_beats")
        out: list[Region] = []
        with self.db.tx() as conn:
            for j, o in enumerate(old):
                if j not in matched_old:
                    conn.execute("DELETE FROM regions WHERE id = ?", (o.id,))
        for i, fields in enumerate(regions):
            fields = {k: v for k, v in fields.items() if k not in ("id", "source")}
            if i in matched_new:
                o = old[matched_new[i]]
                f = {k: v for k, v in fields.items() if k not in keep_cols}
                out.append(self.update_region(o.id, **f, position=i))
                continue
            if any(iou(fields["start_s"], fields["end_s"], u.start_s, u.end_s) >= user_iou for u in users):
                continue
            out.append(self.insert_region(sample_id, {**fields, "position": i}, source="analysis"))
        return out

    def update_region(self, region_id: str, **fields: Any) -> Region:
        f = _json_fields(fields)
        if f:
            cols = ", ".join(f"{k} = ?" for k in f)
            with self.db.tx() as conn:
                conn.execute(f"UPDATE regions SET {cols} WHERE id = ?", (*f.values(), region_id))
        return self.get_region(region_id)

    def delete_region(self, region_id: str) -> None:
        with self.db.tx() as conn:
            conn.execute("DELETE FROM regions WHERE id = ?", (region_id,))

    def regions_with_samples(self) -> list[tuple[Region, Sample]]:
        samples = {s.id: s for s in self.list_samples()}
        out: list[tuple[Region, Sample]] = []
        for s in samples.values():
            for r in s.regions:
                out.append((r, s))
        return out

    # --- sets --------------------------------------------------------------------------

    def insert_set(self, name: str) -> Set:
        sid = new_id()
        with self.db.tx() as conn:
            conn.execute(
                "INSERT INTO sets (id, name, tonal_settings, temporal_settings, created_at) VALUES (?,?,?,?,?)",
                (sid, name, dumps(TonalSettings().model_dump()), dumps(TemporalSettings().model_dump()), now_iso()),
            )
        return self.get_set(sid)

    def get_set(self, set_id: str) -> Set:
        r = self.db.one("SELECT * FROM sets WHERE id = ?", (set_id,))
        if r is None:
            raise NotFound(set_id)
        members = self.db.all(
            "SELECT m.position, r.* FROM set_members m JOIN regions r ON r.id = m.region_id WHERE m.set_id = ? ORDER BY m.position",
            (set_id,),
        )
        return Set(
            id=r["id"],
            name=r["name"],
            anchor_region_id=r["anchor_region_id"],
            session_bpm=r["session_bpm"],
            bpm_pinned=bool(r["bpm_pinned"]),
            tonal_settings=TonalSettings.model_validate(loads(r["tonal_settings"], {})),
            temporal_settings=TemporalSettings.model_validate(loads(r["temporal_settings"], {})),
            created_at=r["created_at"],
            members=[SetMember(region=row_to_region(m), position=m["position"]) for m in members],
        )

    def list_sets(self) -> list[Set]:
        return [self.get_set(r["id"]) for r in self.db.all("SELECT id FROM sets ORDER BY created_at")]

    def update_set(self, set_id: str, **fields: Any) -> Set:
        f = _json_fields(fields)
        if f:
            cols = ", ".join(f"{k} = ?" for k in f)
            with self.db.tx() as conn:
                conn.execute(f"UPDATE sets SET {cols} WHERE id = ?", (*f.values(), set_id))
        return self.get_set(set_id)

    def delete_set(self, set_id: str) -> None:
        with self.db.tx() as conn:
            conn.execute("DELETE FROM sets WHERE id = ?", (set_id,))

    def add_member(self, set_id: str, region_id: str) -> Set:
        with self.db.tx() as conn:
            pos = conn.execute("SELECT COALESCE(MAX(position), -1) + 1 FROM set_members WHERE set_id = ?", (set_id,)).fetchone()[0]
            conn.execute(
                "INSERT OR IGNORE INTO set_members (set_id, region_id, position) VALUES (?,?,?)",
                (set_id, region_id, pos),
            )
        return self.get_set(set_id)

    def remove_member(self, set_id: str, region_id: str) -> Set:
        with self.db.tx() as conn:
            conn.execute("DELETE FROM set_members WHERE set_id = ? AND region_id = ?", (set_id, region_id))
        return self.get_set(set_id)

    def sets_containing(self, region_id: str) -> list[str]:
        rows = self.db.all("SELECT set_id FROM set_members WHERE region_id = ?", (region_id,))
        return [r["set_id"] for r in rows]

    # --- progressions ------------------------------------------------------------------

    def insert_progression(self, set_id: str, name: str, key: str | None) -> Progression:
        pid = new_id()
        with self.db.tx() as conn:
            conn.execute(
                "INSERT INTO progressions (id, set_id, name, key, created_at) VALUES (?,?,?,?,?)",
                (pid, set_id, name, key, now_iso()),
            )
        return self.get_progression(pid)

    def get_progression(self, prog_id: str) -> Progression:
        r = self.db.one("SELECT * FROM progressions WHERE id = ?", (prog_id,))
        if r is None:
            raise NotFound(prog_id)
        return Progression(
            id=r["id"],
            set_id=r["set_id"],
            name=r["name"],
            key=r["key"],
            loop=bool(r["loop"]),
            created_at=r["created_at"],
            slots=self.slots(prog_id),
        )

    def list_progressions(self, set_id: str | None = None) -> list[Progression]:
        if set_id:
            rows = self.db.all("SELECT id FROM progressions WHERE set_id = ? ORDER BY created_at", (set_id,))
        else:
            rows = self.db.all("SELECT id FROM progressions ORDER BY created_at")
        return [self.get_progression(r["id"]) for r in rows]

    def update_progression(self, prog_id: str, **fields: Any) -> Progression:
        f = _json_fields(fields)
        if f:
            cols = ", ".join(f"{k} = ?" for k in f)
            with self.db.tx() as conn:
                conn.execute(f"UPDATE progressions SET {cols} WHERE id = ?", (*f.values(), prog_id))
        return self.get_progression(prog_id)

    def delete_progression(self, prog_id: str) -> None:
        with self.db.tx() as conn:
            conn.execute("DELETE FROM progressions WHERE id = ?", (prog_id,))

    def slots(self, prog_id: str) -> list[Slot]:
        rows = self.db.all("SELECT * FROM slots WHERE progression_id = ? ORDER BY idx", (prog_id,))
        return [row_to_slot(r) for r in rows]

    def get_slot(self, slot_id: str) -> Slot:
        r = self.db.one("SELECT * FROM slots WHERE id = ?", (slot_id,))
        if r is None:
            raise NotFound(slot_id)
        return row_to_slot(r)

    def replace_slots(self, prog_id: str, specs: list[dict[str, Any]]) -> list[Slot]:
        """Replace all slots. A slot whose (idx, chord_root, chord_quality)
        matches an old slot keeps its region assignment, lock, transpose and
        pattern."""
        old = {(s.idx, s.chord_root, s.chord_quality): s for s in self.slots(prog_id)}
        with self.db.tx() as conn:
            conn.execute("DELETE FROM slots WHERE progression_id = ?", (prog_id,))
            for i, spec in enumerate(specs):
                prev = old.get((i, spec.get("chord_root"), spec.get("chord_quality")))
                conn.execute(
                    "INSERT INTO slots (id, progression_id, idx, chord_symbol, roman, chord_root, chord_quality, bass_note, length_beats, region_id, transpose_semitones, locked, pattern) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        new_id(), prog_id, i, spec["chord_symbol"], spec.get("roman"), spec.get("chord_root"),
                        spec.get("chord_quality"), spec.get("bass_note"), spec.get("length_beats", 4.0),
                        prev.region_id if prev else None,
                        prev.transpose_semitones if prev else 0,
                        int(prev.locked) if prev else 0,
                        dumps(prev.pattern.model_dump() if prev else Pattern().model_dump()),
                    ),
                )
        return self.slots(prog_id)

    def update_slot(self, slot_id: str, **fields: Any) -> Slot:
        f = _json_fields(fields)
        if f:
            cols = ", ".join(f"{k} = ?" for k in f)
            with self.db.tx() as conn:
                conn.execute(f"UPDATE slots SET {cols} WHERE id = ?", (*f.values(), slot_id))
        return self.get_slot(slot_id)

    # --- renders --------------------------------------------------------------------------

    def get_render(self, cache_key: str) -> RenderStatus | None:
        r = self.db.one("SELECT * FROM renders WHERE cache_key = ?", (cache_key,))
        return row_to_render(r) if r else None

    def upsert_render(self, cache_key: str, region_id: str, set_id: str, path: str, status: str, params: dict[str, Any], *,
                      deltas: dict[str, Any] | None = None, error: str | None = None) -> RenderStatus:
        ts = now_iso()
        with self.db.tx() as conn:
            conn.execute(
                """INSERT INTO renders (cache_key, region_id, set_id, path, status, error, params, deltas, created_at, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(cache_key) DO UPDATE SET status = excluded.status, error = excluded.error,
                     deltas = COALESCE(excluded.deltas, renders.deltas), updated_at = excluded.updated_at""",
                (cache_key, region_id, set_id, path, status, error, dumps(params), dumps(deltas) if deltas else None, ts, ts),
            )
        out = self.get_render(cache_key)
        assert out is not None
        return out

    def latest_ready_render(self, set_id: str, region_id: str) -> RenderStatus | None:
        r = self.db.one(
            "SELECT * FROM renders WHERE set_id = ? AND region_id = ? AND status = 'ready' ORDER BY updated_at DESC LIMIT 1",
            (set_id, region_id),
        )
        return row_to_render(r) if r else None

    def render_path(self, cache_key: str) -> str | None:
        r = self.db.one("SELECT path, status FROM renders WHERE cache_key = ?", (cache_key,))
        return r["path"] if r and r["status"] == "ready" else None
