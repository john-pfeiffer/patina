"""Library roots: recursive scans, provenance, change detection, removal, settings and the analysis pause."""

from __future__ import annotations

import os
import time

from conftest import synth_chords

from patine.audio import iter_audio_files


def _wait(client, pred, timeout=30.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.05)
    return False


def _job_done(client, job_id):
    return client.get(f"/jobs/{job_id}").json()["status"] in ("done", "failed", "cancelled")


def _tree(folder):
    (folder / "Packs" / "Keys").mkdir(parents=True)
    (folder / "Packs" / ".hidden").mkdir(parents=True)
    (folder / "Loops").mkdir()
    synth_chords(folder / "Packs" / "Keys" / "pad_c.wav", chords=((0, "maj"),), beats_per_chord=2)
    synth_chords(folder / "Loops" / "loop_f.wav", chords=((5, "maj"),), beats_per_chord=2)
    (folder / "Loops" / "._loop_f.wav").write_bytes(b"\x00\x05\x16\x07" + b"\x00" * 64)   # AppleDouble sidecar
    (folder / ".DS_Store").write_bytes(b"\x00" * 16)
    synth_chords(folder / "Packs" / ".hidden" / "secret.wav", chords=((7, "maj"),), beats_per_chord=2)
    (folder / "notes.txt").write_text("not audio")


def test_iter_audio_files_skips_hidden(tmp_path):
    _tree(tmp_path)
    rel = [str(p.relative_to(tmp_path)) for p in iter_audio_files(tmp_path)]
    assert rel == [os.path.join("Loops", "loop_f.wav"), os.path.join("Packs", "Keys", "pad_c.wav")]


def test_add_root_scans_recursively_and_records_provenance(app_client, tmp_data_dir):
    lib = tmp_data_dir / "lib"
    _tree(lib)
    res = app_client.post("/roots", json={"path": str(lib)}).json()
    root, job = res["root"], res["job"]
    assert root["name"] == "lib" and job["kind"] == "scan"
    assert _wait(app_client, lambda: _job_done(app_client, job["id"]))
    result = app_client.get(f"/jobs/{job['id']}").json()["result"]
    assert result["added"] == 2 and result["files"] == 2
    root = app_client.get(f"/roots/{root['id']}").json()
    assert root["status"] == "idle" and root["file_count"] == 2 and root["last_scan_at"]
    samples = app_client.get("/samples").json()
    by_rel = {s["rel_path"]: s for s in samples}
    assert set(by_rel) == {os.path.join("Loops", "loop_f.wav"), os.path.join("Packs", "Keys", "pad_c.wav")}
    assert all(s["root_id"] == root["id"] and s["mtime"] and s["size"] for s in samples)
    # the same folder again is the same root (no duplicate), and files already known are not re-added
    again = app_client.post("/roots", json={"path": str(lib)}).json()
    assert again["root"]["id"] == root["id"]
    assert _wait(app_client, lambda: _job_done(app_client, again["job"]["id"]))
    assert app_client.get(f"/jobs/{again['job']['id']}").json()["result"]["added"] == 0
    assert len(app_client.get("/samples").json()) == 2
    # analysis of everything found runs in the background
    assert _wait(app_client, lambda: all(s["analysis_status"] == "done" for s in app_client.get("/samples").json()), 90.0)


def test_rescan_detects_new_changed_and_missing_files(app_client, tmp_data_dir):
    lib = tmp_data_dir / "lib"
    _tree(lib)
    res = app_client.post("/roots", json={"path": str(lib)}).json()
    assert _wait(app_client, lambda: _job_done(app_client, res["job"]["id"]))
    assert _wait(app_client, lambda: all(s["analysis_status"] == "done" for s in app_client.get("/samples").json()), 90.0)
    # add one, change one, delete one
    synth_chords(lib / "Loops" / "new_g.wav", chords=((7, "maj"),), beats_per_chord=2)
    changed = lib / "Packs" / "Keys" / "pad_c.wav"
    synth_chords(changed, chords=((9, "min"),), beats_per_chord=4)
    os.utime(changed, (time.time() + 5, time.time() + 5))
    (lib / "Loops" / "loop_f.wav").unlink()
    job = app_client.post(f"/roots/{res['root']['id']}/scan").json()
    assert _wait(app_client, lambda: _job_done(app_client, job["id"]))
    result = app_client.get(f"/jobs/{job['id']}").json()["result"]
    assert (result["added"], result["changed"], result["missing"]) == (1, 1, 1)
    samples = {s["rel_path"]: s for s in app_client.get("/samples").json()}
    assert samples[os.path.join("Loops", "loop_f.wav")]["analysis_status"] == "missing"
    assert samples[os.path.join("Loops", "new_g.wav")]["analysis_status"] in ("pending", "running", "done")
    assert _wait(app_client, lambda: app_client.get("/samples").json() and all(
        s["analysis_status"] in ("done", "missing") for s in app_client.get("/samples").json()), 90.0)
    pad = next(s for s in app_client.get("/samples").json() if s["rel_path"] == os.path.join("Packs", "Keys", "pad_c.wav"))
    assert pad["regions"] and pad["regions"][0]["chord_root"] == "A"      # re-analysed after the change
    # the deleted file comes back: no longer missing, not re-analysed
    synth_chords(lib / "Loops" / "loop_f.wav", chords=((5, "maj"),), beats_per_chord=2)
    job2 = app_client.post(f"/roots/{res['root']['id']}/scan").json()
    assert _wait(app_client, lambda: _job_done(app_client, job2["id"]))
    back = next(s for s in app_client.get("/samples").json() if s["rel_path"] == os.path.join("Loops", "loop_f.wav"))
    assert back["analysis_status"] != "missing"


def test_remove_root_keeps_or_deletes_samples(app_client, tmp_data_dir):
    lib = tmp_data_dir / "lib"
    _tree(lib)
    res = app_client.post("/roots", json={"path": str(lib)}).json()
    assert _wait(app_client, lambda: _job_done(app_client, res["job"]["id"]))
    assert app_client.delete(f"/roots/{res['root']['id']}").status_code == 204
    assert app_client.get("/roots").json() == []
    kept = app_client.get("/samples").json()
    assert len(kept) == 2 and all(s["root_id"] is None for s in kept)
    res2 = app_client.post("/roots", json={"path": str(lib), "name": "Splice"}).json()
    assert res2["root"]["name"] == "Splice"
    assert _wait(app_client, lambda: _job_done(app_client, res2["job"]["id"]))
    assert all(s["root_id"] == res2["root"]["id"] for s in app_client.get("/samples").json())   # re-adopted
    assert app_client.delete(f"/roots/{res2['root']['id']}", params={"delete_samples": "true"}).status_code == 204
    assert app_client.get("/samples").json() == []
    assert app_client.post("/roots", json={"path": str(lib / "nope")}).status_code == 400


def test_settings_and_analysis_pause(app_client, tmp_data_dir):
    defaults = app_client.get("/settings").json()
    assert defaults == {"analysis_paused": False, "analysis_hpss": True, "analysis_window_beats": 1.0,
                        "analysis_fallback": True, "scan_on_startup": True}
    paused = app_client.patch("/settings", json={"analysis_paused": True, "analysis_window_beats": 0.5}).json()
    assert paused["analysis_paused"] is True and paused["analysis_window_beats"] == 0.5
    state = app_client.state_obj
    assert state.worker.is_paused("analyze")
    path, _ = synth_chords(tmp_data_dir / "x.wav", chords=((0, "maj"),), beats_per_chord=2)
    res = app_client.post("/samples", json={"paths": [str(path)]}).json()
    job = res["jobs"][0]
    time.sleep(0.5)
    assert app_client.get(f"/jobs/{job['id']}").json()["status"] == "queued"     # parked, not run
    assert state.worker.parked_count >= 1 or app_client.get(f"/jobs/{job['id']}").json()["status"] == "queued"
    app_client.patch("/settings", json={"analysis_paused": False})
    assert not state.worker.is_paused("analyze")
    assert _wait(app_client, lambda: _job_done(app_client, job["id"]), 60.0)
    assert app_client.get(f"/jobs/{job['id']}").json()["status"] == "done"
    assert app_client.get("/settings").json()["analysis_paused"] is False
