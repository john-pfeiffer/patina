"""§7.5–7.6 stretch engine wrapper (temporal/stretch.py)."""

import numpy as np
import pytest

from patine.models import Region, TemporalSettings, UserOverrides
from patine.temporal.grid import derive_region_tempo
from patine.temporal.stretch import declick, resample_ratio, resolve_method, stretch_plan
from patine.temporal.warp import build_warp_plan

SR = 48000


def tone(freq: float, seconds: float, channels: int = 2) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    y = np.stack([np.sin(2 * np.pi * freq * (k + 1) / (k + 1) * t) for k in range(channels)])
    return (0.5 * y).astype(np.float32)


def est_freq(x: np.ndarray, a: int, b: int) -> float:
    seg = x[a:b]
    zc = np.where(np.diff(np.sign(seg)) > 0)[0]
    return len(zc) / ((b - a) / SR)


def declared(session_bpm: float, strength: float = 85.0, **kw):
    base = dict(id="r", sample_id="s", start_s=0.0, end_s=2.0, declared_length_beats=4, first_onset_s=0.0)
    base.update(kw)
    r = Region(**base)
    t = derive_region_tempo(r, None, None, [])
    return build_warp_plan(r, t, session_bpm, strength, r.user_overrides)


def uneven(session_bpm: float, strength: float = 100.0):
    r = Region(id="u", sample_id="s", start_s=1.0, end_s=3.0, beats=[1.0, 1.52, 2.0, 2.47], first_onset_s=1.0)
    t = derive_region_tempo(r, 120.0, 0.9, [])
    return build_warp_plan(r, t, session_bpm, strength, r.user_overrides)


@pytest.mark.parametrize("bpm", [100.0, 140.0, 60.0, 123.4])
def test_elastic_output_length_is_exact(bpm):
    y = tone(220.0, 3.0)
    p = declared(bpm)
    out = stretch_plan(y, SR, p, method="elastic")
    assert out.shape == (2, int(round(p.dst_len_s * SR)))
    assert out.dtype == np.float32


def test_elastic_keeps_pitch_and_level():
    y = tone(220.0, 3.0)
    out = stretch_plan(y, SR, declared(100.0), method="elastic", declick_ms=0)
    assert est_freq(out[0], 10000, 100000) == pytest.approx(220.0, rel=0.02)
    assert np.sqrt(np.mean(out ** 2)) == pytest.approx(np.sqrt(np.mean(y[:, : 2 * SR] ** 2)), rel=0.1)


def test_non_uniform_plan_length_and_continuity():
    y = tone(220.0, 3.5)
    p = uneven(120.0)
    assert not p.uniform
    out = stretch_plan(y, SR, p, method="elastic")
    assert out.shape[1] == int(round(p.dst_len_s * SR))
    jumps = np.abs(np.diff(out[0]))
    # a sine at 220 Hz moves at most ~0.014/sample; a bad join would show as a step
    assert jumps.max() < 0.05
    assert np.sqrt(np.mean(out ** 2)) > 0.3


def test_repitch_changes_pitch_with_tempo_and_keeps_length():
    y = tone(220.0, 3.0)
    p = declared(100.0)
    out = stretch_plan(y, SR, p, method="repitch", declick_ms=0)
    assert out.shape[1] == int(round(2.4 * SR))
    assert est_freq(out[0], 10000, 100000) == pytest.approx(220.0 / 1.2, rel=0.02)


def test_elastic_transposition_constant_length():
    y = tone(220.0, 3.0)
    p = declared(120.0)
    out = stretch_plan(y, SR, p, method="elastic", transpose_semitones=3, declick_ms=0)
    assert out.shape[1] == 2 * SR
    assert est_freq(out[0], 10000, 90000) == pytest.approx(220.0 * 2 ** (3 / 12), rel=0.02)


def test_plain_repitch_length_compensation():
    """Plain transposition = resample (changes length) + stretch folded into the
    plan so the final length is still the planned one (§5.4)."""
    from patine.temporal.warp import scale_plan_source

    y = tone(220.0, 3.0)
    f = 2 ** (3 / 12)
    resampled = resample_ratio(y, SR, f)
    assert resampled.shape[1] == int(round(3 * SR / f))
    p = scale_plan_source(declared(100.0), 1.0 / f)
    out = stretch_plan(resampled, SR, p, method="elastic", declick_ms=0)
    assert out.shape[1] == int(round(2.4 * SR))
    assert est_freq(out[0], 10000, 100000) == pytest.approx(220.0 * f, rel=0.02)


def test_identity_plan_is_bit_exact_copy():
    rng = np.random.default_rng(3)
    y = (rng.standard_normal((2, 3 * SR)) * 0.2).astype(np.float32)
    out = stretch_plan(y, SR, declared(120.0), declick_ms=0)
    assert out.shape == (2, 2 * SR)
    assert np.array_equal(out, y[:, : 2 * SR])


def test_declick_ramps():
    y = np.ones((1, SR), dtype=np.float32)
    out = declick(y, SR, 1.5)
    r = int(round(0.0015 * SR))
    assert out[0, 0] == 0.0
    assert out[0, r - 1] < 1.0 and out[0, r] == 1.0
    assert out[0, -1] < 0.02 and out[0, -r - 1] == 1.0
    assert np.all(out[0, r:-r] == 1.0)


def test_resample_ratio_lengths_and_pitch():
    y = tone(220.0, 2.0)
    up = resample_ratio(y, SR, 2.0)
    assert up.shape[1] == SR
    assert est_freq(up[0], 1000, 40000) == pytest.approx(440.0, rel=0.02)
    tiny = resample_ratio(y, SR, 2 ** (10 / 1200))
    assert tiny.shape[1] == int(round(2 * SR / 2 ** (10 / 1200)))
    assert np.array_equal(resample_ratio(y, SR, 1.0), y)


def test_resolve_method_forces_elastic_on_non_uniform_plan():
    settings = TemporalSettings(stretch_method="repitch")
    assert resolve_method(settings, UserOverrides(), uneven(120.0)) == ("elastic", True)
    assert resolve_method(settings, UserOverrides(allow_wobble=True), uneven(120.0)) == ("repitch", False)
    assert resolve_method(settings, UserOverrides(), declared(100.0)) == ("repitch", False)
    assert resolve_method(TemporalSettings(), UserOverrides(stretch_method="repitch"), declared(100.0)) == ("repitch", False)
    assert resolve_method(TemporalSettings(), UserOverrides(), uneven(120.0)) == ("elastic", False)


def test_mono_input_and_short_region():
    y = tone(330.0, 1.0, channels=1)
    r = Region(id="s", sample_id="s", start_s=0.1, end_s=0.4, declared_length_beats=1)
    t = derive_region_tempo(r, None, None, [])
    p = build_warp_plan(r, t, 150.0, 85.0, r.user_overrides)
    out = stretch_plan(y, SR, p)
    assert out.shape == (1, int(round(p.dst_len_s * SR)))
