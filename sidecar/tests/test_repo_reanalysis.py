"""Reanalysis policy: ids survive a re-run where regions overlap, user
regions are never re-cut, stale analyses are re-queued."""

from __future__ import annotations

import pytest
from conftest import synth_chords

from patine import ANALYSIS_VERSION
from patine.db import Database
from patine.repo import Repo


def _region(start: float, end: float, root: str = "C", quality: str = "maj") -> dict:
    return {"start_s": start, "end_s": end, "chord_root": root, "chord_quality": quality, "chord_confidence": 0.9}


def test_replace_keeps_ids_by_overlap_and_references_survive(app_client):
    state = app_client.state_obj
    repo: Repo = state.repo
    sample = repo.insert_sample("/music/a.wav", "h", 8.0, 44100, 2)
    a, b = repo.replace_analysis_regions(sample.id, [_region(0.0, 2.0), _region(2.0, 4.0, "F")])
    repo.update_region(a.id, declared_length_beats=4.0, user_overrides={"nudge_ms": 5.0})

    st = app_client.post("/sets", json={"name": "set"}).json()
    app_client.post(f"/sets/{st['id']}/members", json={"region_id": a["id"] if isinstance(a, dict) else a.id})
    app_client.patch(f"/sets/{st['id']}", json={"anchor_region_id": a.id})
    prog = app_client.post("/progressions", json={"set_id": st["id"], "name": "p", "key": "C major", "text": "C F"}).json()
    slot = app_client.patch(f"/slots/{prog['slots'][0]['id']}", json={"region_id": a.id}).json()
    assert slot["region_id"] == a.id

    # A re-run: A moves a little (A'), B is gone, C is new.
    out = repo.replace_analysis_regions(sample.id, [_region(0.1, 2.0, "C", "min"), _region(4.0, 6.0, "G")])
    assert [r.id for r in out][0] == a.id
    assert out[0].start_s == 0.1 and out[0].chord_quality == "min"
    assert out[0].declared_length_beats == 4.0 and out[0].user_overrides.nudge_ms == 5.0
    assert out[0].position == 0 and out[1].position == 1
    ids = {r.id for r in repo.regions_for_sample(sample.id)}
    assert b.id not in ids and out[1].id in ids
    st2 = app_client.get(f"/sets/{st['id']}").json()
    assert st2["anchor_region_id"] == a.id
    assert [m["region"]["id"] for m in st2["members"]] == [a.id]
    prog2 = app_client.get(f"/progressions/{prog['id']}").json()
    assert prog2["slots"][0]["region_id"] == a.id


def test_user_regions_are_never_recut():
    repo = Repo(Database(":memory:"))
    sample = repo.insert_sample("/music/b.wav", "h", 8.0, 44100, 2)
    repo.replace_analysis_regions(sample.id, [_region(0.0, 2.0)])
    mine = repo.insert_region(sample.id, {**_region(2.0, 4.0, "F"), "position": 1}, source="user")
    out = repo.replace_analysis_regions(sample.id, [_region(0.0, 2.0, "C", "7"), _region(2.1, 4.0, "A", "min")])
    assert len(out) == 1 and out[0].chord_quality == "7"          # the span the user owns is not re-cut
    regions = repo.regions_for_sample(sample.id)
    assert {r.id for r in regions} == {out[0].id, mine.id}
    assert repo.get_region(mine.id).chord_root == "F"


def test_stale_samples_are_listed_and_requeued(app_client, tmp_data_dir):
    state = app_client.state_obj
    repo: Repo = state.repo
    path, _ = synth_chords(tmp_data_dir / "c.wav", chords=((0, "maj"),), beats_per_chord=4)
    res = app_client.post("/samples", json={"paths": [str(path)]}).json()
    sid = res["samples"][0]["id"]
    assert len(res["jobs"]) == 1
    # Pretend an older pipeline analysed it.
    repo.update_sample(sid, analysis_status="done", analysis_version=ANALYSIS_VERSION - 1)
    assert repo.stale_sample_ids(ANALYSIS_VERSION) == [sid]
    again = app_client.post("/samples", json={"paths": [str(path)]}).json()
    assert [s["id"] for s in again["samples"]] == [sid]
    assert len(again["jobs"]) == 1 and again["jobs"][0]["payload"]["sample_id"] == sid
    repo.update_sample(sid, analysis_status="done", analysis_version=ANALYSIS_VERSION)
    assert repo.stale_sample_ids(ANALYSIS_VERSION) == []
    assert app_client.post("/samples", json={"paths": [str(path)]}).json()["jobs"] == []


def test_patching_an_analysis_region_makes_it_the_users(app_client, tmp_data_dir):
    state = app_client.state_obj
    repo: Repo = state.repo
    path, _ = synth_chords(tmp_data_dir / "d.wav", chords=((0, "maj"),), beats_per_chord=4)
    sample = repo.insert_sample(str(path), "h", 2.0, 22050, 2)
    repo.update_sample(sample.id, analysis_status="done", analysis_version=ANALYSIS_VERSION, beats=[0.0, 0.5, 1.0, 1.5],
                       onsets=[0.0, 0.5, 1.0, 1.5])
    (region,) = repo.replace_analysis_regions(sample.id, [_region(0.0, 2.0)])
    assert region.source == "analysis"
    patched = app_client.patch(f"/regions/{region.id}", json={"start_s": 0.5}).json()
    assert patched["source"] == "user" and patched["start_s"] == 0.5
    assert patched["first_onset_s"] == 0.5          # stored onsets feed the re-measure
    # and a reanalysis now leaves it alone
    out = repo.replace_analysis_regions(sample.id, [_region(0.4, 2.0, "A", "min")])
    assert out == []
    assert repo.get_region(region.id).chord_root == "C"


def test_region_editor_endpoints(app_client, tmp_data_dir):
    """Windowed peaks and audio slices, snap_s round trip, bounds validation."""
    from patine.conform.renderer import RenderRequest, cache_key
    from patine.models import Region, UserOverrides
    from patine.temporal.grid import snap_point

    state = app_client.state_obj
    repo = state.repo
    path, total = synth_chords(tmp_data_dir / "e.wav", chords=((0, "maj"),), beats_per_chord=4)
    sample = repo.insert_sample(str(path), "h", total, 22050, 2)
    repo.update_sample(sample.id, analysis_status="done", analysis_version=ANALYSIS_VERSION, duration=total,
                       beats=[0.0, 0.5, 1.0, 1.5], onsets=[0.0, 0.5, 1.0, 1.5])
    (region,) = repo.replace_analysis_regions(sample.id, [_region(0.0, 2.0)])

    whole = app_client.get(f"/samples/{sample.id}/peaks", params={"bins": 100}).json()
    assert whole["bins"] == 100 and whole["start"] == 0.0 and whole["end"] == pytest.approx(total, abs=1e-3)
    win = app_client.get(f"/samples/{sample.id}/peaks", params={"bins": 50, "start": 0.5, "end": 1.0}).json()
    assert win["bins"] == 50 and win["start"] == 0.5 and win["end"] == 1.0 and len(win["peaks"]) == 50
    assert max(win["peaks"]) > 0.1

    audio = app_client.get(f"/samples/{sample.id}/audio", params={"start": 0.25, "end": 0.75})
    assert audio.status_code == 200 and audio.headers["content-type"].startswith("audio/wav")
    import io

    import soundfile as sf
    y, sr = sf.read(io.BytesIO(audio.content), dtype="float32", always_2d=True)
    assert sr == 22050 and abs(y.shape[0] - 0.5 * sr) <= 2
    assert app_client.get(f"/samples/{sample.id}/audio", params={"start": 1.0, "end": 0.5}).status_code == 400

    # snap_s is honoured by the snap point, changes the render key, must lie inside the region
    base = app_client.get(f"/regions/{region.id}").json()
    ov = {**base["user_overrides"], "snap_s": 1.0}
    patched = app_client.patch(f"/regions/{region.id}", json={"user_overrides": ov}).json()
    assert patched["user_overrides"]["snap_s"] == 1.0 and patched["source"] == "user"
    r_before = Region.model_validate(base)
    r_after = Region.model_validate(patched)
    assert snap_point(r_before) == 0.0 and snap_point(r_after) == 1.0
    assert snap_point(Region.model_validate({**patched, "user_overrides": {**ov, "keep_preroll": True}})) == 1.0
    from patine.models import TemporalSettings, TonalSettings

    req = lambda r: RenderRequest(  # noqa: E731
        region=r, sample_path=str(path), sample_hash="h", sample_tuning_cents=0.0, sample_bpm=120.0,
        sample_bpm_confidence=0.9, sample_beats=[0.0, 0.5, 1.0, 1.5], anchor=None, tonal=TonalSettings(),
        temporal=TemporalSettings(), session_bpm=120.0,
    )
    assert cache_key(req(r_before)) != cache_key(req(r_after))
    assert app_client.patch(f"/regions/{region.id}", json={"user_overrides": {**ov, "snap_s": 5.0}}).status_code == 400
    # bounds are validated against the file before anything is written
    assert app_client.patch(f"/regions/{region.id}", json={"start_s": total + 1.0, "end_s": total + 2.0}).status_code == 400
    assert app_client.get(f"/regions/{region.id}").json()["start_s"] == 0.0
    assert app_client.patch(f"/regions/{region.id}", json={"end_s": total + 5.0}).json()["end_s"] == pytest.approx(total, abs=1e-6)
    assert isinstance(UserOverrides().snap_s, type(None))
