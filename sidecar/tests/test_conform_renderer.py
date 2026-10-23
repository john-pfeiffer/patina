"""Conform renderer: cache key, deltas, render_region and the anchor invariant (§2, §6)."""

import copy
from pathlib import Path

import numpy as np
import pytest
from conftest import synth_chords

from patine.audio import file_hash, load_audio
from patine.conform import filters as F
from patine.conform.loudness import measure_lufs
from patine.conform.renderer import (
    AnchorProfile,
    RenderRequest,
    cache_key,
    compute_deltas,
    render_region,
)
from patine.models import Region, TemporalSettings, TonalMetrics, TonalSettings, UserOverrides

SR = 48000


def measure(y: np.ndarray, sr: int, **over) -> TonalMetrics:
    m = TonalMetrics(
        loudness_lufs=measure_lufs(y, sr),
        tilt_bands=F.octave_band_energy_db(y, sr),
        low_rolloff_hz=60.0,
        bass_mono_hz=120.0,
        sub_ratio=0.01,
        noise_floor_db=F.noise_floor_db(y, sr),
        noise_spectrum=[0.0] * 10,
        width_bands=F.width_bands(y, sr),
        crest_db=12.0,
        wetness=0.2,
    )
    return m.model_copy(update=over)


@pytest.fixture(scope="module")
def sample(tmp_path_factory):
    d = tmp_path_factory.mktemp("conform")
    path, _ = synth_chords(d / "chords.wav", sr=SR)      # 4 chords × 4 beats at 120 bpm = 8 s
    y, sr = load_audio(path)
    return str(path), file_hash(path), y, sr


def make_region(sample, rid: str, start: float, end: float, **kw) -> Region:
    _, _, y, sr = sample
    metrics = kw.pop("metrics", None) or measure(y[:, int(start * sr):int(end * sr)], sr)
    return Region(id=rid, sample_id="s", start_s=start, end_s=end, declared_length_beats=4,
                  first_onset_s=start, metrics=metrics, **kw)


def profile(sample, region: Region, tuning: float = 0.0) -> AnchorProfile:
    path, h, _, _ = sample
    return AnchorProfile(region_id=region.id, metrics=region.metrics, tuning_cents=tuning,
                         sample_path=path, sample_hash=h, start_s=region.start_s, end_s=region.end_s)


def request(sample, region: Region, anchor: AnchorProfile | None, **kw) -> RenderRequest:
    path, h, _, _ = sample
    base = dict(region=region, sample_path=path, sample_hash=h, sample_tuning_cents=0.0,
                sample_bpm=120.0, sample_bpm_confidence=0.9, sample_beats=[], anchor=anchor,
                tonal=TonalSettings(), temporal=TemporalSettings(), session_bpm=120.0)
    base.update(kw)
    return RenderRequest(**base)


# --- anchor invariant ----------------------------------------------------------------------

def test_anchor_identity_bit_exact_apart_from_declick(sample, tmp_path):
    path, h, y, sr = sample
    r = make_region(sample, "a", 0.0, 2.0)
    req = request(sample, r, profile(sample, r), is_anchor=True, session_bpm=120.0)
    deltas = render_region(req, str(tmp_path / "anchor.wav"))
    out, osr = load_audio(tmp_path / "anchor.wav")
    assert osr == sr
    src = y[:, : 2 * sr]
    assert out.shape == src.shape
    ramp = int(round(0.0015 * sr))
    assert np.allclose(out[:, ramp:-ramp], src[:, ramp:-ramp], atol=1e-6)
    assert np.array_equal(out[:, ramp:-ramp], src[:, ramp:-ramp])
    assert np.all(np.abs(out[:, :ramp]) <= np.abs(src[:, :ramp]) + 1e-7)
    assert out[0, 0] == 0.0
    assert deltas.tuning_cents == 0.0 and deltas.loudness_db == 0.0
    assert deltas.tilt_db == [0.0] * 10 and deltas.width_ratio == [1.0] * 3
    assert deltas.lowend_hpf_hz is None and deltas.noise_floor_db is None
    assert deltas.stretch_ratio == pytest.approx(1.0)


def test_anchor_tonal_identity_even_when_profile_differs(sample, tmp_path):
    """Metrics in the profile never move the anchor: the tonal layer is identity for it."""
    path, h, y, sr = sample
    r = make_region(sample, "a", 0.0, 2.0)
    other = measure(y[:, : 2 * sr] * 0.1, sr)
    prof = AnchorProfile(region_id="a", metrics=other, tuning_cents=30.0, sample_path=path,
                         sample_hash=h, start_s=0.0, end_s=2.0)
    req = request(sample, r, prof, is_anchor=True, sample_tuning_cents=12.0)
    deltas = render_region(req, str(tmp_path / "anchor2.wav"))
    out, _ = load_audio(tmp_path / "anchor2.wav")
    ramp = int(round(0.0015 * sr))
    assert np.array_equal(out[:, ramp:-ramp], y[:, ramp: 2 * sr - ramp])
    assert deltas.tuning_cents == 0.0 and deltas.loudness_db == 0.0


def test_anchor_still_goes_through_temporal_layer(sample, tmp_path):
    _, _, y, sr = sample
    r = make_region(sample, "a", 0.0, 2.0)
    req = request(sample, r, profile(sample, r), is_anchor=True, session_bpm=100.0)
    deltas = render_region(req, str(tmp_path / "anchor_100.wav"))
    out, _ = load_audio(tmp_path / "anchor_100.wav")
    assert out.shape[1] == int(round(2.4 * sr))
    assert deltas.stretch_ratio == pytest.approx(1.2)
    assert deltas.loudness_db == 0.0


# --- members ----------------------------------------------------------------------------------

def test_member_render_length_deltas_and_loudness(sample, tmp_path):
    _, _, y, sr = sample
    a = make_region(sample, "a", 0.0, 2.0)
    quiet = make_region(sample, "m", 2.0, 4.0, metrics=measure(y[:, 2 * sr:4 * sr], sr,
                                                                loudness_lufs=-30.0, noise_floor_db=-200.0))
    req = request(sample, quiet, profile(sample, a), session_bpm=100.0, sample_tuning_cents=8.0)
    progress: list[tuple[float, str]] = []
    deltas = render_region(req, str(tmp_path / "member.wav"), progress=lambda f, m: progress.append((f, m)))
    out, osr = load_audio(tmp_path / "member.wav")
    assert osr == sr and out.shape[0] == 2
    assert out.shape[1] == int(round(2.4 * sr))
    assert deltas.stretch_ratio == pytest.approx(1.2)
    assert deltas.tuning_cents == pytest.approx(-8.0)
    assert deltas.noise_floor_db is not None       # region floor below anchor → a bed
    assert progress[0][0] == 0.0 and progress[-1] == (1.0, "done")
    # loudness: the render measured its own audio, so the gain is small, not the -30 metric gap
    assert abs(deltas.loudness_db) < 6.0
    # readout from metrics alone uses the metric gap
    d2 = compute_deltas(req)
    assert d2.loudness_db == pytest.approx(a.metrics.loudness_lufs - (-30.0), abs=1e-6)
    assert d2.stretch_ratio == pytest.approx(1.2) and d2.tuning_cents == pytest.approx(-8.0)
    assert d2.tilt_db == deltas.tilt_db and d2.width_ratio == deltas.width_ratio


def test_bypass_and_global_toggles(sample, tmp_path):
    _, _, y, sr = sample
    a = make_region(sample, "a", 0.0, 2.0)
    m = make_region(sample, "m", 2.0, 4.0, user_overrides=UserOverrides(bypass={"tilt": True}))
    tonal = TonalSettings()
    tonal.width.enabled = False
    req = request(sample, m, profile(sample, a), tonal=tonal, sample_tuning_cents=5.0)
    d = compute_deltas(req)
    assert set(d.bypassed) == {"tilt", "width"}
    assert d.tilt_db == [0.0] * 10 and d.width_ratio == [1.0] * 3
    assert d.tuning_cents == pytest.approx(-5.0)
    d2 = render_region(req, str(tmp_path / "bypass.wav"))
    assert d2.tilt_db == [0.0] * 10 and d2.width_ratio == [1.0] * 3


def test_no_anchor_means_identity_tonal_layer(sample, tmp_path):
    m = make_region(sample, "m", 2.0, 4.0)
    req = request(sample, m, None, sample_tuning_cents=5.0)
    d = compute_deltas(req)
    assert d.tuning_cents == 0.0 and d.loudness_db == 0.0 and d.bypassed == []


def test_transposition_modes_keep_length(sample, tmp_path):
    _, _, y, sr = sample
    a = make_region(sample, "a", 0.0, 2.0)
    m = make_region(sample, "m", 2.0, 4.0)
    for mode in ("plain", "elastic"):
        req = request(sample, m, profile(sample, a), session_bpm=100.0, transpose_semitones=3,
                      temporal=TemporalSettings(repitch_mode=mode))
        d = render_region(req, str(tmp_path / f"t_{mode}.wav"))
        out, _ = load_audio(tmp_path / f"t_{mode}.wav")
        assert out.shape[1] == int(round(2.4 * sr))
        assert d.transpose_semitones == 3


def test_repitch_forced_elastic_on_non_uniform(sample, tmp_path):
    _, _, y, sr = sample
    a = make_region(sample, "a", 0.0, 2.0)
    m = Region(id="u", sample_id="s", start_s=2.0, end_s=4.0, beats=[2.0, 2.52, 3.0, 3.47],
               first_onset_s=2.0, metrics=measure(y[:, 2 * sr:4 * sr], sr))
    req = request(sample, m, profile(sample, a), temporal=TemporalSettings(stretch_method="repitch"))
    render_region(req, str(tmp_path / "forced.wav"))
    out, _ = load_audio(tmp_path / "forced.wav")
    assert out.shape[1] > 1.9 * sr


# --- cache key -----------------------------------------------------------------------------

def test_cache_key_is_stable_and_hex(sample):
    a = make_region(sample, "a", 0.0, 2.0)
    req = request(sample, a, profile(sample, a), is_anchor=True)
    k1 = cache_key(req)
    k2 = cache_key(copy.deepcopy(req))
    assert k1 == k2
    assert len(k1) == 32 and all(c in "0123456789abcdef" for c in k1)


def test_changing_anchor_changes_every_members_key(sample):
    a = make_region(sample, "a", 0.0, 2.0)
    b = make_region(sample, "b", 2.0, 4.0)
    c = make_region(sample, "c", 4.0, 6.0)
    pa, pb = profile(sample, a), profile(sample, b)
    keys_a = {r.id: cache_key(request(sample, r, pa, is_anchor=(r.id == "a"))) for r in (a, b, c)}
    keys_b = {r.id: cache_key(request(sample, r, pb, is_anchor=(r.id == "b"))) for r in (a, b, c)}
    for rid in ("a", "b", "c"):
        assert keys_a[rid] != keys_b[rid]
    assert len(set(keys_a.values())) == 3


def test_changing_one_regions_overrides_changes_only_its_key(sample):
    a = make_region(sample, "a", 0.0, 2.0)
    b = make_region(sample, "b", 2.0, 4.0)
    c = make_region(sample, "c", 4.0, 6.0)
    pa = profile(sample, a)
    before = {r.id: cache_key(request(sample, r, pa, is_anchor=(r.id == "a"))) for r in (a, b, c)}
    b2 = b.model_copy(update={"user_overrides": UserOverrides(nudge_ms=3.0)})
    after = {r.id: cache_key(request(sample, r, pa, is_anchor=(r.id == "a"))) for r in (a, b2, c)}
    assert after["a"] == before["a"] and after["c"] == before["c"]
    assert after["b"] != before["b"]


def test_cache_key_sensitivity(sample):
    a = make_region(sample, "a", 0.0, 2.0)
    b = make_region(sample, "b", 2.0, 4.0)
    pa = profile(sample, a)
    base = request(sample, b, pa)
    k = cache_key(base)
    variants = [
        request(sample, b, pa, session_bpm=121.0),
        request(sample, b, pa, transpose_semitones=1),
        request(sample, b, pa, temporal=TemporalSettings(warp_strength=50.0)),
        request(sample, b, pa, tonal=TonalSettings(tilt={"enabled": True, "amount": 60})),
        request(sample, b, pa, sample_hash="deadbeef"),
        request(sample, b, pa, extra={"v": 1}),
        request(sample, b, pa, sample_beats=[0.1]),
        request(sample, b.model_copy(update={"end_s": 3.9}), pa),
        request(sample, b, profile(sample, a, tuning=5.0)),
    ]
    assert all(cache_key(v) != k for v in variants)
    assert len({cache_key(v) for v in variants}) == len(variants)


def test_render_writes_float32_at_source_rate(sample, tmp_path):
    path22, _ = synth_chords(tmp_path / "c22.wav", sr=22050, stereo=False)
    y, sr = load_audio(path22)
    r = Region(id="a", sample_id="s", start_s=0.0, end_s=2.0, declared_length_beats=4,
               first_onset_s=0.0, metrics=measure(y[:, : 2 * sr], sr))
    prof = AnchorProfile(region_id="a", metrics=r.metrics, tuning_cents=0.0, sample_path=str(path22),
                         sample_hash=file_hash(path22), start_s=0.0, end_s=2.0)
    req = RenderRequest(region=r, sample_path=str(path22), sample_hash=prof.sample_hash,
                        sample_tuning_cents=0.0, sample_bpm=120.0, sample_bpm_confidence=0.9,
                        sample_beats=[], anchor=prof, tonal=TonalSettings(), temporal=TemporalSettings(),
                        session_bpm=90.0)
    render_region(req, str(tmp_path / "mono.wav"))
    import soundfile as sf

    info = sf.info(str(tmp_path / "mono.wav"))
    assert info.samplerate == 22050 and info.channels == 1 and info.subtype == "FLOAT"
    assert info.frames == int(round(2.0 * 120 / 90 * 22050))
    assert Path(tmp_path / "mono.wav").exists()
