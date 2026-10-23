"""API behaviour that needs no analysis: sets, progressions, theory, jobs."""


def test_health_and_token(app_client):
    r = app_client.get("/health")
    assert r.status_code == 200 and r.json()["name"] == "patine"
    r = app_client.get("/samples", headers={"X-Patine-Token": "wrong"})
    assert r.status_code == 401
    assert app_client.get("/samples").json() == []


def test_sets_and_progressions(app_client):
    s = app_client.post("/sets", json={"name": "A"}).json()
    assert s["tonal_settings"]["tilt"]["amount"] == 40 and s["temporal_settings"]["warp_strength"] == 85
    p = app_client.post("/progressions", json={"set_id": s["id"], "key": "C minor", "text": "i iv V:2 bVI"}).json()
    assert [(x["chord_symbol"], x["roman"], x["length_beats"]) for x in p["slots"]] == [
        ("Cm", "i", 4.0), ("Fm", "iv", 4.0), ("G", "V", 2.0), ("Ab", "bVI", 4.0)]
    slot = p["slots"][2]
    r = app_client.patch(f"/slots/{slot['id']}", json={"chord_symbol": "G7", "locked": True,
                                                        "pattern": {"mode": "retrigger", "division": "1/16", "steps": [{"gate": 0.5}]}})
    assert r.status_code == 200
    j = r.json()
    assert j["chord_quality"] == "7" and j["roman"] == "V7" and j["locked"] and j["pattern"]["division"] == "1/16"
    p2 = app_client.patch(f"/progressions/{p['id']}", json={"text": "i iv V7 bVI VII"}).json()
    assert len(p2["slots"]) == 5 and p2["slots"][2]["locked"] is True and p2["slots"][2]["pattern"]["mode"] == "retrigger"
    assert app_client.patch(f"/slots/{slot['id']}", json={"chord_symbol": "nope"}).status_code in (400, 404)
    assert app_client.get(f"/slots/{p2['slots'][0]['id']}/candidates").json() == []
    assert app_client.get(f"/progressions/{p['id']}/renders").json() == {}
    assert app_client.delete(f"/progressions/{p['id']}").status_code == 204
    assert app_client.get(f"/progressions/{p['id']}").status_code == 404


def test_csv_import(app_client):
    s = app_client.post("/sets", json={"name": "A"}).json()
    csv_text = "chords,roman,key,bpm\n\"Cm Fm G\",\"i iv V\",C minor,92\n,\"I V vi IV\",G major,120\n"
    r = app_client.post("/progressions/import-csv", json={"set_id": s["id"], "csv_text": csv_text})
    assert r.status_code == 200, r.text
    progs = r.json()
    assert len(progs) == 2
    assert [x["chord_symbol"] for x in progs[0]["slots"]] == ["Cm", "Fm", "G"]
    assert [x["chord_symbol"] for x in progs[1]["slots"]] == ["G", "D", "Em", "C"]
    assert progs[1]["key"] == "G major" and "120 bpm" in progs[1]["name"]


def test_theory_parse_never_errors(app_client):
    ok = app_client.post("/theory/parse", json={"text": "Cm Fm G", "key": "Cm"}).json()
    assert ok["ok"] and ok["chords"][1]["roman"] == "iv"
    bad = app_client.post("/theory/parse", json={"text": "Xm"}).json()
    assert bad["ok"] is False and "A–G" in bad["error"]


def test_jobs_listing_and_unknown(app_client):
    assert app_client.get("/jobs").json() == []
    assert app_client.get("/jobs/nope").status_code == 404
