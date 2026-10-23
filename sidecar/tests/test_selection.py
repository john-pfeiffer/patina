"""§5.2 ranking: exact < substitution < near transposition < far transposition."""

from patine.models import Region, Sample, TonalMetrics
from patine.selection import rank_candidates, reroll, timbral_distance
from patine.theory import parse_chord_symbol, parse_key


def _region(rid, root, quality, conf=0.9, metrics=None):
    return Region(id=rid, sample_id="s", start_s=0, end_s=2, chord_root=root, chord_quality=quality,
                  chord_confidence=conf, metrics=metrics or TonalMetrics(loudness_lufs=-18))


def _sample():
    return Sample(id="s", path="/x.wav", hash="h", name="x.wav", added_at="now")


def test_ranking_order_and_labels():
    pool = [
        (_region("exact", "F", "min"), _sample()),
        (_region("relative", "G#", "maj"), _sample()),
        (_region("near", "G", "min"), _sample()),
        (_region("far", "B", "min"), _sample()),
        (_region("wrong", "C", "7"), _sample()),
    ]
    cands = rank_candidates(parse_chord_symbol("Fm"), parse_key("F minor"), pool, None, "function")
    ids = [c.region.id for c in cands]
    assert ids[:3] == ["exact", "relative", "near"]
    assert ids[-1] == "far"
    assert "wrong" not in ids
    labels = {c.region.id: c.label for c in cands}
    assert labels["exact"] == "exact" and labels["relative"] == "relative" and labels["near"] == "-2 st"


def test_exact_strictness_still_offers_transpositions():
    pool = [(_region("near", "G", "min"), _sample()), (_region("sub", "G#", "maj"), _sample())]
    cands = rank_candidates(parse_chord_symbol("Fm"), parse_key("F minor"), pool, None, "exact")
    assert [c.region.id for c in cands] == ["near"]


def test_confidence_and_timbre_cost():
    far_tilt = TonalMetrics(loudness_lufs=-30, tilt_bands=[12, 10, 8, 6, 4, 2, 0, -2, -4, -6],
                            noise_floor_db=-40, crest_db=18, width_bands=[0.5, 0.5, 0.5], wetness=0.9)
    flat = TonalMetrics(loudness_lufs=-18, noise_floor_db=-70, crest_db=8, width_bands=[0, 0, 0], wetness=0.1)
    pool = [(_region("sure", "F", "min", conf=0.95, metrics=far_tilt), _sample()),
            (_region("close", "F", "min", conf=0.6, metrics=flat), _sample())]
    cands = rank_candidates(parse_chord_symbol("Fm"), None, pool, flat, "exact")
    assert cands[0].region.id == "close"
    assert timbral_distance(far_tilt, flat) > timbral_distance(flat, flat) == 0.0


def test_reroll_respects_locks_and_amount():
    pool = [(_region(f"r{i}", "F", "min"), _sample()) for i in range(6)]
    slots = [("a", parse_chord_symbol("Fm"), "r0", False), ("b", parse_chord_symbol("Fm"), "r1", True),
             ("c", parse_chord_symbol("Fm"), "r2", False)]
    changes = reroll(slots, None, pool, None, "exact", 100.0, seed=1)
    assert "b" not in changes and set(changes) == {"a", "c"}
    assert changes["a"][0] != "r0" and changes["c"][0] != "r2"
    assert reroll(slots, None, pool, None, "exact", 0.0, seed=1) == {}
    assert len(reroll(slots, None, pool, None, "exact", 50.0, seed=1)) == 1
