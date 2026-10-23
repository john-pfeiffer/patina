"""End to end on synthetic audio: ingest → analysis → set → progression →
candidates → renders → export. Exercises the real pipeline and renderer."""

import time

import numpy as np
import soundfile as sf

from .conftest import synth_chords


def _wait(client, pred, timeout=120.0, every=0.2):
    t0 = time.time()
    while time.time() - t0 < timeout:
        v = pred()
        if v:
            return v
        time.sleep(every)
    raise AssertionError("timed out")


def test_full_flow(app_client, tmp_data_dir):
    a, _ = synth_chords(tmp_data_dir / "a.wav", chords=((0, "min"), (5, "min"), (7, "maj"), (8, "maj")), bpm=120.0, seed=1)
    b, _ = synth_chords(tmp_data_dir / "b.wav", chords=((5, "min"), (0, "min")), bpm=100.0, seed=2, detune_cents=15)

    r = app_client.post("/samples", json={"paths": [str(tmp_data_dir)]})
    assert r.status_code == 200, r.text
    assert len(r.json()["samples"]) == 2 and len(r.json()["jobs"]) == 2

    def analysed():
        samples = app_client.get("/samples").json()
        return samples if all(s["analysis_status"] in ("done", "failed") for s in samples) else None

    samples = _wait(app_client, analysed)
    for s in samples:
        assert s["analysis_status"] == "done", s["analysis_error"]
        assert s["detected_bpm"] and s["metrics"]["loudness_lufs"] is not None
        assert s["regions"], "analysis produced no regions"
    by_name = {s["name"]: s for s in samples}
    sa, sb = by_name["a.wav"], by_name["b.wav"]
    assert abs(sb["tuning_cents"] - 15) < 8, sb["tuning_cents"]
    roots_a = [rg["chord_root"] for rg in sa["regions"]]
    assert "C" in roots_a and "F" in roots_a, roots_a

    # Set: first member becomes the anchor; session bpm follows it.
    st = app_client.post("/sets", json={"name": "e2e"}).json()
    anchor = sa["regions"][0]
    st = app_client.post(f"/sets/{st['id']}/members", json={"region_id": anchor["id"]}).json()
    assert st["anchor_region_id"] == anchor["id"]
    assert st["session_bpm"] and 100 < st["session_bpm"] < 140, st["session_bpm"]
    for rg in sb["regions"][:2]:
        st = app_client.post(f"/sets/{st['id']}/members", json={"region_id": rg["id"]}).json()
    assert len(st["members"]) == 3

    deltas = app_client.get(f"/sets/{st['id']}/deltas").json()
    assert deltas[anchor["id"]]["loudness_db"] == 0.0
    member = sb["regions"][0]["id"]
    assert abs(deltas[member]["tuning_cents"]) > 5

    # Progression + candidates + assignment.
    p = app_client.post("/progressions", json={"set_id": st["id"], "key": "C minor", "text": "i iv V bVI"}).json()
    slot_i = p["slots"][0]
    cands = app_client.get(f"/slots/{slot_i['id']}/candidates?strictness=function").json()
    assert cands and cands[0]["label"] == "exact" and cands[0]["region"]["chord_root"] == "C"
    for slot in p["slots"]:
        cs = app_client.get(f"/slots/{slot['id']}/candidates?strictness=two_tones").json()
        if cs:
            app_client.patch(f"/slots/{slot['id']}", json={"region_id": cs[0]["region"]["id"],
                                                          "transpose_semitones": cs[0]["transpose_semitones"]})
    p = app_client.get(f"/progressions/{p['id']}").json()
    assert sum(1 for s in p["slots"] if s["region_id"]) >= 2

    # Renders: queued on first ask, ready after the worker runs.
    def ready():
        rs = app_client.get(f"/progressions/{p['id']}/renders").json()
        return rs if rs and all(x["status"] == "ready" for x in rs.values()) else None

    rs = _wait(app_client, ready)
    for slot_id, status in rs.items():
        assert status["url"].startswith("/renders/")
        wav = app_client.get(status["url"])
        assert wav.status_code == 200 and wav.headers["content-type"].startswith("audio/wav")
    # Anchor render = identity on the tonal layer.
    set_renders = app_client.get(f"/sets/{st['id']}/renders").json()
    anchor_status = next(x for x in set_renders if x["region_id"] == anchor["id"])
    _wait(app_client, lambda: app_client.get(f"/sets/{st['id']}/renders").json() and
          all(x["status"] == "ready" for x in app_client.get(f"/sets/{st['id']}/renders").json()))
    anchor_status = next(x for x in app_client.get(f"/sets/{st['id']}/renders").json() if x["region_id"] == anchor["id"])
    assert anchor_status["deltas"]["loudness_db"] == 0.0 and anchor_status["deltas"]["tuning_cents"] == 0.0

    # Changing the anchor invalidates members (new keys, stale markers).
    st2 = app_client.patch(f"/sets/{st['id']}", json={"anchor_region_id": member}).json()
    assert st2["anchor_region_id"] == member
    rs2 = app_client.get(f"/progressions/{p['id']}/renders").json()
    assert any(x["stale"] for x in rs2.values())

    # Reroll changes only unlocked slots.
    app_client.patch(f"/slots/{p['slots'][0]['id']}", json={"locked": True})
    before = {s["id"]: s["region_id"] for s in p["slots"]}
    p3 = app_client.post(f"/progressions/{p['id']}/reroll", json={"amount": 100, "strictness": "two_tones", "seed": 3}).json()
    assert p3["slots"][0]["region_id"] == before[p3["slots"][0]["id"]]

    # Export all four kinds.
    out = tmp_data_dir / "out"
    for kind in ("midi", "kit", "stems", "full"):
        job = app_client.post("/export", json={"progression_id": p["id"], "kind": kind, "out_dir": str(out / kind)}).json()
        job = _wait(app_client, lambda: (lambda j: j if j["status"] in ("done", "failed") else None)(app_client.get(f"/jobs/{job['id']}").json()))
        assert job["status"] == "done", job["error"]
        assert job["result"]["files"], kind
    full = [f for f in (out / "full").iterdir() if f.suffix == ".wav"]
    assert full
    y, sr = sf.read(str(full[0]))
    beats = sum(s["length_beats"] for s in p3["slots"])
    expected = beats * 60.0 / app_client.get(f"/sets/{st['id']}").json()["session_bpm"]
    assert abs(len(y) / sr - expected) < 0.05, (len(y) / sr, expected)
    assert np.abs(y).max() > 0.01
    assert (out / "kit" / "patine.json").exists()
    assert any(f.suffix == ".mid" for f in (out / "midi").iterdir())
