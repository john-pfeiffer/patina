"""§7.2–7.4 warp plans (temporal/warp.py)."""

import pytest

from patine.models import Region, UserOverrides, WarpMarker
from patine.temporal.grid import derive_region_tempo
from patine.temporal.warp import (
    beat_times_s,
    build_warp_plan,
    rebase_plan,
    scale_plan_source,
)


def declared_region(**kw) -> Region:
    base = dict(id="r", sample_id="s", start_s=0.0, end_s=2.0, declared_length_beats=4, first_onset_s=0.0)
    base.update(kw)
    return Region(**base)


def uneven_region() -> Region:
    return Region(id="u", sample_id="s", start_s=1.0, end_s=3.0, beats=[1.0, 1.52, 2.0, 2.47], first_onset_s=1.0)


def test_declared_four_beats_at_120_to_session_100():
    r = declared_region()
    t = derive_region_tempo(r, None, None, [])
    p = build_warp_plan(r, t, 100.0, 85.0, r.user_overrides)
    assert p.dst_len_s == pytest.approx(2.4)
    assert p.src_len_s == pytest.approx(2.0)
    assert p.average_ratio == pytest.approx(1.2)
    assert p.uniform
    assert len(p.segments) == 4
    for s in p.segments:
        assert s.ratio == pytest.approx(1.2)
        assert s.dst_len_s == pytest.approx(0.6)


def test_same_tempo_is_identity_plan():
    r = declared_region()
    t = derive_region_tempo(r, None, None, [])
    p = build_warp_plan(r, t, 120.0, 85.0, r.user_overrides)
    assert p.average_ratio == pytest.approx(1.0)
    assert p.uniform
    assert p.play_start_s == 0.0
    assert p.dst_len_s == pytest.approx(2.0)


def test_strength_100_puts_every_beat_on_grid():
    r = uneven_region()
    t = derive_region_tempo(r, 120.0, 0.9, [])
    p = build_warp_plan(r, t, 120.0, 100.0, r.user_overrides)
    assert not p.uniform
    for s in p.segments[:-1]:
        assert s.dst_len_s == pytest.approx(0.5)   # each tracked interval → exactly one session beat
    ratios = [s.ratio for s in p.segments]
    assert max(ratios) > 1.05 and min(ratios) < 0.98


def test_strength_0_is_average_ratio_only_with_same_total():
    r = uneven_region()
    t = derive_region_tempo(r, 120.0, 0.9, [])
    p100 = build_warp_plan(r, t, 120.0, 100.0, r.user_overrides)
    p0 = build_warp_plan(r, t, 120.0, 0.0, r.user_overrides)
    assert p0.uniform
    assert p0.dst_len_s == pytest.approx(p100.dst_len_s)
    for s in p0.segments:
        assert s.ratio == pytest.approx(p0.average_ratio)
    # the beat boundaries drift from the grid at 0 %
    boundaries0 = [sum(s.dst_len_s for s in p0.segments[:k]) for k in range(1, len(p0.segments))]
    assert any(abs(b - round(b / 0.5) * 0.5) > 0.005 for b in boundaries0)


def test_strength_blends_between():
    r = uneven_region()
    t = derive_region_tempo(r, 120.0, 0.9, [])
    p100 = build_warp_plan(r, t, 120.0, 100.0, r.user_overrides)
    p50 = build_warp_plan(r, t, 120.0, 50.0, r.user_overrides)
    for a, b in zip(p50.segments, p100.segments, strict=True):
        expected = p100.average_ratio + 0.5 * (b.ratio - p100.average_ratio)
        assert a.ratio == pytest.approx(expected, rel=1e-6)
    assert p50.dst_len_s == pytest.approx(p100.dst_len_s)


def test_manual_marker_pins_source_to_beat_and_wins_over_grid():
    r = uneven_region()
    t = derive_region_tempo(r, 120.0, 0.9, [])
    ov = UserOverrides(warp_markers=[WarpMarker(source_s=1.6, beat=1.0)])
    p = build_warp_plan(r, t, 120.0, 100.0, ov)
    first = p.segments[0]
    assert first.src_start_s == 1.0 and first.src_end_s == pytest.approx(1.6)
    assert first.dst_len_s == pytest.approx(0.5)
    # the tracked beat at 1.52 (beat 1) yielded to the marker; 2.0 (beat 2) is kept
    assert [round(s.src_end_s, 3) for s in p.segments][:3] == [1.6, 2.0, 2.47]
    # markers hold at any strength
    p0 = build_warp_plan(r, t, 120.0, 0.0, ov)
    assert sum(s.dst_len_s for s in p0.segments if s.src_end_s <= 1.6 + 1e-9) == pytest.approx(0.5)


def test_transient_snap_nudge_and_preroll():
    r = declared_region(first_onset_s=0.02)
    t = derive_region_tempo(r, None, None, [])
    p = build_warp_plan(r, t, 120.0, 85.0, r.user_overrides)
    assert p.play_start_s == pytest.approx(0.02)
    assert p.dst_len_s == pytest.approx(1.98)
    nudged = UserOverrides(nudge_ms=5.0)
    pn = build_warp_plan(r, t, 120.0, 85.0, nudged)
    assert pn.play_start_s == pytest.approx(0.025)
    assert pn.dst_len_s == pytest.approx(1.975)
    back = UserOverrides(nudge_ms=-10.0)
    pb = build_warp_plan(r, t, 120.0, 85.0, back)
    assert pb.play_start_s == pytest.approx(0.01)
    assert pb.dst_len_s == pytest.approx(1.99)
    # nudge cannot leave the region
    pc = build_warp_plan(r, t, 120.0, 85.0, UserOverrides(nudge_ms=-500.0))
    assert pc.play_start_s == 0.0
    keep = declared_region(first_onset_s=0.02, user_overrides=UserOverrides(keep_preroll=True))
    tk = derive_region_tempo(keep, None, None, [])
    pk = build_warp_plan(keep, tk, 120.0, 85.0, keep.user_overrides)
    assert pk.play_start_s == 0.0 and pk.dst_len_s == pytest.approx(2.0)


def test_rebase_and_scale_keep_destination():
    r = declared_region()
    t = derive_region_tempo(r, None, None, [])
    p = build_warp_plan(r, t, 100.0, 85.0, r.user_overrides)
    rb = rebase_plan(p, 0.5)
    assert rb.segments[0].src_start_s == pytest.approx(-0.5)
    assert rb.dst_len_s == pytest.approx(p.dst_len_s)
    sc = scale_plan_source(p, 1.1)
    assert sc.src_len_s == pytest.approx(2.2)
    assert sc.dst_len_s == pytest.approx(2.4)
    assert sc.average_ratio == pytest.approx(2.4 / 2.2)


def test_beat_times_at_session_tempo():
    r = uneven_region()
    t = derive_region_tempo(r, 120.0, 0.9, [])
    p = build_warp_plan(r, t, 120.0, 100.0, r.user_overrides)
    times = beat_times_s(p, t)
    assert [b for b, _ in times] == [0.0, 1.0, 2.0, 3.0]
    assert [round(ts, 6) for _, ts in times] == pytest.approx([0.0, 0.5, 1.0, 1.5])


def test_bad_session_bpm():
    r = declared_region()
    t = derive_region_tempo(r, None, None, [])
    with pytest.raises(ValueError):
        build_warp_plan(r, t, 0.0, 85.0, r.user_overrides)
