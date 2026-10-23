"""§7.1–7.3 tempo derivation and beat grid (temporal/grid.py)."""

import numpy as np
import pytest

from patine.models import Region, UserOverrides
from patine.temporal.grid import derive_region_tempo, snap_point


def region(**kw) -> Region:
    base = dict(id="r", sample_id="s", start_s=0.0, end_s=2.0)
    base.update(kw)
    return Region(**base)


def test_declared_length_wins_and_is_exact():
    r = region(declared_length_beats=4, first_onset_s=0.0)
    t = derive_region_tempo(r, sample_bpm=97.0, sample_bpm_confidence=0.9, sample_beats=[0.1, 0.7, 1.3])
    assert t.source == "declared"
    assert t.bpm == pytest.approx(120.0)
    assert t.length_beats == pytest.approx(4.0)
    assert t.confidence == 1.0
    assert t.beat_grid == pytest.approx([0.0, 0.5, 1.0, 1.5, 2.0])
    assert t.grid_step_beats == 1.0


def test_declared_bars_style_lengths():
    r = region(start_s=1.0, end_s=4.0, declared_length_beats=8)
    t = derive_region_tempo(r, None, None, [])
    assert t.bpm == pytest.approx(160.0)
    assert t.beat_grid[0] == 1.0
    assert len(t.beat_grid) == 9


def test_detected_uses_tracked_beats_inside_region():
    beats = [0.0, 0.52, 1.0, 1.47, 2.0, 2.5]
    r = region(start_s=0.0, end_s=2.0, beats=beats, first_onset_s=0.0)
    t = derive_region_tempo(r, 120.0, 0.8, [])
    assert t.source == "detected"
    assert t.beat_grid == pytest.approx([0.0, 0.52, 1.0, 1.47, 2.0])
    # bpm measured inside the region (median interval 0.5 s → 120)
    assert t.bpm == pytest.approx(120.0, rel=0.05)
    assert t.length_beats == pytest.approx(4.0, abs=0.1)
    assert t.confidence == pytest.approx(0.8)


def test_detected_falls_back_to_sample_beats_when_region_has_none():
    r = region(start_s=1.0, end_s=3.0)
    t = derive_region_tempo(r, 120.0, 0.5, [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
    assert t.source == "detected"
    assert t.beat_grid[0] == 1.0
    assert len(t.beat_grid) == 5


def test_tempo_multiplier_and_feel_change_step():
    beats = [0.0, 0.5, 1.0, 1.5, 2.0]
    r = region(beats=beats, user_overrides=UserOverrides(tempo_multiplier=2.0))
    t = derive_region_tempo(r, 120.0, 0.9, [])
    assert t.bpm == pytest.approx(240.0)
    assert t.grid_step_beats == 2.0
    assert t.length_beats == pytest.approx(8.0)
    r8 = region(beats=beats, user_overrides=UserOverrides(tempo_feel="eighth"))
    t8 = derive_region_tempo(r8, 120.0, 0.9, [])
    assert t8.bpm == pytest.approx(60.0)
    assert t8.grid_step_beats == 0.5
    assert t8.length_beats == pytest.approx(2.0)


def test_fallback_linear_grid_with_fewer_than_two_beats():
    r = region(beats=[0.3])
    t = derive_region_tempo(r, 100.0, 0.4, [])
    assert t.source == "fallback"
    assert t.bpm == pytest.approx(100.0)
    assert np.allclose(np.diff(t.beat_grid), 0.6)
    t_none = derive_region_tempo(region(), None, None, [])
    assert t_none.source == "fallback" and t_none.confidence == 0.0


def test_snap_point_and_keep_preroll():
    r = region(first_onset_s=0.08, declared_length_beats=4)
    assert snap_point(r) == pytest.approx(0.08)
    assert snap_point(r, transient_snap=False) == 0.0
    r.user_overrides.keep_preroll = True
    assert snap_point(r) == 0.0
    t = derive_region_tempo(region(first_onset_s=0.08, declared_length_beats=4), None, None, [])
    assert t.beat_grid[0] == pytest.approx(0.08)
    assert t.bpm == pytest.approx(120.0)           # tempo from the declared bounds, exact
    assert t.length_beats == pytest.approx(3.84)   # snapped span in beats


def test_skipped_tracked_beat_is_interpolated():
    r = region(beats=[0.0, 0.5, 1.5, 2.0], end_s=2.0)
    t = derive_region_tempo(r, 120.0, 0.9, [])
    assert t.beat_grid == pytest.approx([0.0, 0.5, 1.0, 1.5, 2.0])


def test_empty_region_raises():
    with pytest.raises(ValueError):
        derive_region_tempo(region(start_s=1.0, end_s=1.0), None, None, [])
