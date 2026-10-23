"""End-to-end analysis on synthesised progressions (handoff §4)."""

from __future__ import annotations

import logging

import numpy as np
import pytest
import soundfile as sf
from test_analysis_synth import PROGRESSION, chord_signal, decoded

from patine.analysis.pipeline import (
    AnalysisOptions,
    AnalysisResult,
    analyze_decoded,
    analyze_file,
    measure_region,
)
from patine.audio import load_audio
from patine.theory import ANALYSIS_QUALITIES, pc_name

EXPECTED = [(pc_name(root), quality) for root, quality in PROGRESSION]
SR = 44100
BPM = 120.0


@pytest.fixture(scope="module")
def prog_path(tmp_path_factory):
    y, _ = chord_signal(PROGRESSION, sr=SR, bpm=BPM, stereo=True)
    path = tmp_path_factory.mktemp("analysis") / "progression.wav"
    sf.write(str(path), y.T, SR, subtype="FLOAT")
    return path


@pytest.fixture(scope="module")
def analysis(prog_path) -> tuple[AnalysisResult, list[tuple[float, str]]]:
    progress: list[tuple[float, str]] = []
    result = analyze_file(
        prog_path,
        progress=lambda f, m: progress.append((f, m)),
        options=AnalysisOptions(beat_tracker="librosa"),
    )
    return result, progress


def test_file_facts(analysis):
    result, _ = analysis
    assert result.sample_rate == SR
    assert result.channels == 2
    assert result.duration == pytest.approx(16.0, abs=0.01)


def test_regions_tile_and_carry_the_right_chords(analysis):
    result, _ = analysis
    found = [(r.chord_root, r.chord_quality) for r in result.regions]
    assert found == EXPECTED
    assert result.regions[0].start_s < 0.2
    assert result.regions[-1].end_s == pytest.approx(16.0, abs=0.05)
    for prev, cur in zip(result.regions, result.regions[1:], strict=False):
        assert cur.start_s == pytest.approx(prev.end_s, abs=1e-6)
    for i, r in enumerate(result.regions):
        assert r.start_s == pytest.approx(2.0 * i, abs=0.2), f"region {i} starts at {r.start_s}"
        assert r.chord_quality in ANALYSIS_QUALITIES
        assert 0.0 <= r.chord_confidence <= 1.0
        assert r.chord_confidence > 0.6
        assert r.bass_note == r.chord_root        # root position voicings
        assert r.key_context == "C major"


def test_tempo_and_beats(analysis):
    result, _ = analysis
    assert result.detected_bpm == pytest.approx(BPM, rel=0.03)
    assert result.bpm_confidence > 0.5
    assert abs(len(result.beats) - 32) <= 2
    for r in result.regions:
        assert 3 <= len(r.beats) <= 5
        assert all(r.start_s <= b < r.end_s for b in r.beats)
    assert all(d in result.beats for d in result.downbeats)   # best effort, may be empty


def test_first_onset_snaps_inside_region(analysis):
    result, _ = analysis
    for r in result.regions:
        assert r.first_onset_s is not None
        assert r.start_s <= r.first_onset_s <= r.start_s + 0.15


def test_tuning_key_and_context(analysis):
    result, _ = analysis
    assert abs(result.tuning_cents) < 3.0
    assert result.key_estimate == "C major"
    assert result.key_confidence > 0.3
    ctx = result.context
    assert ctx.sample_rate == SR
    assert ctx.beats == result.beats
    assert len(ctx.onsets) >= 30
    assert ctx.chroma is not None and ctx.chroma.shape == (12, len(ctx.frame_times))
    assert len(ctx.chord_frames) == len(ctx.frame_times)
    roots = {root for root, _, _ in ctx.chord_frames if root is not None}
    assert {"C", "A", "F", "G", "D"} <= roots


def test_progress_is_monotonic(analysis):
    _, progress = analysis
    fractions = [f for f, _ in progress]
    assert fractions[0] == 0.0 and fractions[-1] == 1.0
    assert all(b >= a for a, b in zip(fractions, fractions[1:], strict=False))
    assert all(m for _, m in progress)


def test_metrics_present_and_sane(analysis):
    result, _ = analysis
    m = result.metrics
    assert m.loudness_lufs is not None and -30 < m.loudness_lufs < -10
    assert len(m.tilt_bands) == 10 and len(m.noise_spectrum) == 10 and len(m.width_bands) == 3
    assert all(w < 0.2 for w in m.width_bands)         # near-mono stereo file
    assert m.bass_mono_hz is not None and m.bass_mono_hz > 500
    assert m.sub_ratio is not None and m.sub_ratio < 0.05
    assert m.crest_db is not None and m.crest_db > 6
    assert m.low_rolloff_hz is not None and 20 <= m.low_rolloff_hz <= 120
    for r in result.regions:
        assert r.metrics.loudness_lufs == pytest.approx(m.loudness_lufs, abs=4.0)
        assert r.busyness < 0.15
        assert 100 < r.register_centroid_hz < 600


def test_measure_region_round_trips(analysis, prog_path):
    result, _ = analysis
    y, sr = load_audio(prog_path)
    ref = result.regions[1]
    again = measure_region(y, sr, ref.start_s, ref.end_s, result.context)
    assert (again.chord_root, again.chord_quality, again.bass_note) == (ref.chord_root, ref.chord_quality, ref.bass_note)
    assert again.metrics.loudness_lufs == pytest.approx(ref.metrics.loudness_lufs, abs=0.1)
    assert again.beats == ref.beats
    assert again.first_onset_s == ref.first_onset_s
    assert again.key_context == ref.key_context
    # without the in-memory features (e.g. context rebuilt from the database)
    ctx = result.context
    ctx_light = type(ctx)(
        sample_rate=ctx.sample_rate, tuning_cents=ctx.tuning_cents, beats=ctx.beats,
        downbeats=ctx.downbeats, onsets=ctx.onsets, key_estimate=ctx.key_estimate,
    )
    light = measure_region(y, sr, ref.start_s, ref.end_s, ctx_light)
    assert (light.chord_root, light.chord_quality) == (ref.chord_root, ref.chord_quality)
    assert light.chord_confidence > 0.5
    assert light.key_context is not None
    # user override keeps root/quality, measures the rest
    forced = measure_region(y, sr, ref.start_s, ref.end_s, ctx, chord_root="E", chord_quality="min")
    assert (forced.chord_root, forced.chord_quality) == ("E", "min")
    assert forced.chord_confidence < again.chord_confidence
    root_only = measure_region(y, sr, ref.start_s, ref.end_s, ctx, chord_root="A")
    assert (root_only.chord_root, root_only.chord_quality) == ("A", "min")
    with pytest.raises(ValueError):
        measure_region(y, sr, 3.0, 3.0, ctx)


def test_detuned_file_is_detected():
    y, _ = chord_signal(PROGRESSION[:4], sr=22050, stereo=False, detune_cents=20.0)
    result = analyze_decoded(decoded(y, 22050), options=AnalysisOptions(beat_tracker="librosa"))
    assert result.tuning_cents == pytest.approx(20.0, abs=3.0)
    assert [(r.chord_root, r.chord_quality) for r in result.regions] == EXPECTED[:4]


def test_melody_on_top_does_not_flip_chords():
    y, _ = chord_signal(PROGRESSION, sr=22050, stereo=False, melody=True)
    result = analyze_decoded(decoded(y, 22050), options=AnalysisOptions(beat_tracker="librosa"))
    assert [(r.chord_root, r.chord_quality) for r in result.regions] == EXPECTED
    assert all(r.busyness > 0.3 for r in result.regions)


def test_short_segments_merge_and_silence_is_not_a_region():
    a, _ = chord_signal(((0, "maj"),), sr=22050, stereo=False, beats_per_chord=4)
    b, _ = chord_signal(((2, "min"),), sr=22050, stereo=False, beats_per_chord=0.6)   # < one beat
    c, _ = chord_signal(((5, "maj"),), sr=22050, stereo=False, beats_per_chord=4)
    gap = np.zeros((1, 22050 * 2), dtype=np.float32)
    y = np.concatenate([a, b, c, gap, a], axis=1)
    result = analyze_decoded(decoded(y, 22050), options=AnalysisOptions(beat_tracker="librosa"))
    chords = [(r.chord_root, r.chord_quality) for r in result.regions]
    assert chords == [("C", "maj"), ("F", "maj"), ("C", "maj")]
    assert result.regions[1].start_s == pytest.approx(result.regions[0].end_s, abs=1e-6)
    assert result.regions[2].start_s - result.regions[1].end_s > 1.0     # the silence is a gap


def test_silence_yields_nothing():
    result = analyze_decoded(decoded(np.zeros((2, 22050 * 3), dtype=np.float32), 22050), options=AnalysisOptions(beat_tracker="librosa"))
    assert result.regions == []
    assert result.key_estimate is None
    assert result.metrics.loudness_lufs is None
    assert result.tuning_cents == 0.0


def test_optional_backends_fall_back(caplog):
    pytest.importorskip("numpy")
    for mod in ("demucs", "beat_this", "onnxruntime"):
        try:
            __import__(mod)
        except ImportError:
            pass
        else:
            pytest.skip(f"{mod} installed; fallback path not exercised")
    y, _ = chord_signal(PROGRESSION[:4], sr=22050, stereo=False)
    audio = decoded(y, 22050)
    with caplog.at_level(logging.WARNING, logger="patine.analysis"):
        result = analyze_decoded(
            audio,
            options=AnalysisOptions(use_source_separation=True, chord_model="learned", beat_tracker="beat_this"),
        )
    messages = " ".join(rec.getMessage() for rec in caplog.records)
    assert "demucs" in messages
    assert "learned chord model unavailable" in messages
    assert "beat_this" in messages
    assert [(r.chord_root, r.chord_quality) for r in result.regions] == EXPECTED[:4]
    assert result.detected_bpm == pytest.approx(BPM, rel=0.03)
