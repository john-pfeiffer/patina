"""Unit tests for the analysis submodules: tuning, tempo, onsets, chords,
key and activity (handoff §4.2–4.7), on synthesised arrays."""

from __future__ import annotations

import numpy as np
import pytest
from test_analysis_synth import PROGRESSION, chord_signal, clicks, noise, sine

from patine.analysis import activity, chords, onsets, tempo, tuning
from patine.analysis import key as keymod
from patine.analysis.chroma import ChromaFeatures, compute_chroma
from patine.analysis.learned import LearnedChordBackend
from patine.theory import ANALYSIS_QUALITIES, parse_note

SR = 22050


def mono(chord_list, **kw) -> np.ndarray:
    y, _ = chord_signal(chord_list, sr=SR, stereo=False, **kw)
    return y[0]


# --- §4.2 tuning ---------------------------------------------------------------------

@pytest.mark.parametrize("cents", [20.0, -30.0, 0.0])
def test_tuning_offset(cents):
    y = mono(PROGRESSION[:4], detune_cents=cents)
    assert tuning.estimate_tuning_cents(y, SR) == pytest.approx(cents, abs=2.5)


def test_tuning_of_silence_is_zero():
    assert tuning.estimate_tuning_cents(np.zeros(SR, dtype=np.float32), SR) == 0.0


# --- §4.3 tempo ----------------------------------------------------------------------

def test_tempo_of_click_track():
    y = mono(PROGRESSION[:4], bpm=100.0)
    r = tempo.track_beats(y, SR, backend="librosa")
    assert r.bpm == pytest.approx(100.0, rel=0.02)
    assert r.confidence > 0.6
    assert abs(len(r.beats) - 16) <= 2
    assert r.backend == "librosa"


def test_tempo_confidence_is_low_without_a_pulse():
    assert tempo.track_beats(noise(8.0, SR, 0.3), SR, backend="librosa").confidence < 0.35
    assert tempo.track_beats(mono(PROGRESSION[:4], clicks=False), SR, backend="librosa").confidence < 0.35


def test_bpm_from_beats_ignores_skips():
    beats = np.array([0.0, 0.5, 1.0, 2.0, 2.5, 3.0])     # one dropped beat
    assert tempo.bpm_from_beats(beats, 90.0) == pytest.approx(120.0)
    assert tempo.bpm_from_beats(np.array([1.0]), 90.0) == 90.0


# --- §4.4 onsets -----------------------------------------------------------------------

def test_detect_onsets_finds_the_clicks():
    y, truth = clicks(4.0, SR, 0.5)
    found = onsets.detect_onsets(None, SR, y=y)
    assert abs(len(found) - len(truth)) <= 2
    for t in truth[1:]:
        assert min(abs(f - t) for f in found) < 0.04


def test_first_onset_in():
    found = [0.10, 0.95, 1.52]
    assert onsets.first_onset_in(found, 0.5, 2.0) == 0.95
    assert onsets.first_onset_in(found, 0.96, 2.0) == pytest.approx(0.96)   # just after: clamp, not skip
    assert onsets.first_onset_in(found, 1.6, 2.0) == 1.6                    # none inside: region start
    assert onsets.first_onset_in([], 1.6, 2.0) == 1.6
    assert onsets.snap_to_onset(1.0, found, 0.06) == 0.95
    assert onsets.snap_to_onset(1.2, found, 0.06) == 1.2


# --- §4.5 chords -------------------------------------------------------------------------

@pytest.mark.parametrize("quality", ANALYSIS_QUALITIES)
def test_each_quality_is_recognised(quality):
    feats = compute_chroma(mono(((7, quality),), beats_per_chord=4), SR)
    frames = chords.estimate_frame_chords(feats, chords.TemplateBackend())
    segs = chords.segment_chords(frames, feats, min_duration_s=0.5)
    assert len(segs) == 1
    assert (segs[0].root, segs[0].quality) == (7, quality)
    assert segs[0].bass == 7
    assert segs[0].confidence > 0.6


@pytest.mark.parametrize("bass_interval, bass_pc", [(4, 4), (7, 7)])
def test_inversions_report_the_bass_note(bass_interval, bass_pc):
    feats = compute_chroma(mono(((0, "maj"),), bass_interval=bass_interval, beats_per_chord=4), SR)
    frames = chords.estimate_frame_chords(feats, chords.TemplateBackend())
    segs = chords.segment_chords(frames, feats, min_duration_s=0.5)
    assert [(s.root, s.quality, s.bass) for s in segs] == [(0, "maj", bass_pc)]


def test_abrupt_start_gets_an_onset():
    y, truth = clicks(2.0, SR, 0.5)
    assert onsets.detect_onsets(None, SR, y=y)[0] == 0.0
    assert onsets.starts_abruptly(np.concatenate([np.zeros(SR, dtype=np.float32), y]), SR) is False


def test_hmm_helpers():
    a = chords.transition_matrix(0.093, 1.5)
    assert a.shape == (25, 25)
    assert np.allclose(a.sum(axis=1), 1.0)
    assert a[0, 0] > 0.9
    rng = np.random.default_rng(0)
    probs = rng.random((25, 40))
    probs /= probs.sum(axis=0, keepdims=True)
    post = chords.posteriors(probs, a)
    assert post.shape == probs.shape
    assert np.allclose(post.sum(axis=0), 1.0)
    assert chords.state_label(24) == (None, None)
    assert chords.state_label(chords.state_index(9, "min")) == (9, "min")
    assert chords.normalise_sus(0, "sus2", 7) == (7, "sus4")
    assert chords.normalise_sus(7, "sus4", 0) == (0, "sus2")
    assert chords.normalise_sus(7, "sus4", 7) == (7, "sus4")


def test_no_chord_on_noise_and_silence():
    for y in (noise(4.0, SR, 0.3), np.zeros(SR * 4, dtype=np.float32)):
        feats = compute_chroma(y, SR)
        frames = chords.estimate_frame_chords(feats, chords.TemplateBackend())
        assert np.all(frames.states == chords.NO_CHORD)
        assert chords.dominant_chord(frames) is None
        assert chords.segment_chords(frames, feats, min_duration_s=0.5) == []


def test_backend_selection_falls_back_to_template(tmp_path):
    assert isinstance(chords.get_backend("template"), chords.TemplateBackend)
    assert isinstance(chords.get_backend("auto", None), chords.TemplateBackend)
    assert isinstance(chords.get_backend("learned", str(tmp_path)), chords.TemplateBackend)
    with pytest.raises((ImportError, FileNotFoundError)):
        LearnedChordBackend(tmp_path)


# --- §4.6 key ----------------------------------------------------------------------------

def test_key_estimates():
    major = compute_chroma(mono(((0, "maj"), (9, "min"), (5, "maj"), (7, "maj"))), SR)
    name, conf, scores = keymod.estimate_key(major)
    assert name == "C major" and conf > 0.3 and scores.shape == (24,)
    minor = compute_chroma(mono(((9, "min"), (2, "min"), (4, "maj"), (9, "min"))), SR)
    assert keymod.estimate_key(minor)[0] == "A minor"
    frames = [("A", "min", 1.0)] * 10 + [("D", "min", 1.0)] * 10 + [("E", "maj", 1.0)] * 10
    assert keymod.estimate_key(minor, chord_frames=frames)[0] == "A minor"
    assert keymod.estimate_key(compute_chroma(np.zeros(SR * 2, dtype=np.float32), SR)) == (None, 0.0, pytest.approx(np.zeros(24)))


def test_key_context_follows_the_local_key():
    hop_s = 0.1
    times = np.arange(0, 20.0, hop_s)
    frames = []
    for t in times:
        root, quality = (("C", "maj") if t < 10.0 else ("F#", "maj"))
        if int(t) % 3 == 1:
            root = "F" if t < 10.0 else "B"
        elif int(t) % 3 == 2:
            root = "G" if t < 10.0 else "C#"
        frames.append((root, quality, 1.0))
    chroma = np.stack([chords.chord_template(parse_note(r)[0], q) for r, q, _ in frames], axis=1)
    g = keymod.key_scores(keymod.chord_tone_histogram(frames))
    assert keymod.key_context(chroma, times, 2.0, 4.0, chord_frames=frames, global_scores=g, window_s=3.0) == "C major"
    assert keymod.key_context(chroma, times, 15.0, 17.0, chord_frames=frames, global_scores=g, window_s=3.0) == "Gb major"


# --- §4.7 activity -------------------------------------------------------------------------

def test_busyness_and_register():
    plain = compute_chroma(mono(PROGRESSION[:1], melody=False), SR)
    busy = compute_chroma(mono(PROGRESSION[:1], melody=True), SR)
    assert activity.busyness(plain) < 0.15
    assert activity.busyness(busy) > 0.3
    assert 100.0 < activity.register_centroid_hz(plain) < 400.0
    high = compute_chroma(mono(PROGRESSION[:1], bass=False, chord_octave=6, clicks=False), SR)
    assert activity.register_centroid_hz(high) > 800.0
    empty = plain.slice_frames(0, 0)
    assert isinstance(empty, ChromaFeatures)
    assert activity.busyness(empty) == 0.0 and activity.register_centroid_hz(empty) == 0.0


# --- recall: harmonic chroma, gates, windows, fallback, bare roots ------------------------

from test_analysis_synth import arpeggio_signal, oneshot_note, pad_under_drums  # noqa: E402

from patine.analysis.chroma import cosine_similarity  # noqa: E402


def test_harmonic_chroma_cleans_a_pad_under_drums():
    y, _ = pad_under_drums(((0, "maj"),), beats_per_chord=8, drum_db=6.0)
    full = compute_chroma(y, SR, hpss=False)
    harm = compute_chroma(y, SR, hpss=True)
    template = chords.chord_template(0, "maj")
    assert cosine_similarity(harm.mean_weighted(), template) >= cosine_similarity(full.mean_weighted(), template) + 0.05
    backend = chords.TemplateBackend()
    assert backend.similarities(harm)[chords.NO_CHORD].mean() < backend.similarities(full)[chords.NO_CHORD].mean()
    # the full-mix views are untouched so busyness and register still see everything
    assert np.allclose(harm.treble, full.treble)
    assert np.allclose(harm.cqt, full.cqt)


def test_no_chord_floor_and_gates():
    backend = chords.TemplateBackend()
    pure = compute_chroma(np.concatenate([np.zeros(SR // 10, dtype=np.float32), sine(220.0, 2.0, SR, 0.3)]), SR)
    sims = backend.similarities(pure)
    loud = pure.energy > 0.1 * pure.energy.max()
    assert np.all(sims[:chords.N_CHORD_STATES, loud].max(axis=0) - sims[chords.NO_CHORD, loud] >= 0.15)
    white = compute_chroma(noise(4.0, SR, 0.3), SR)
    ws = backend.similarities(white)
    frac = np.mean(ws[chords.NO_CHORD] >= ws[:chords.N_CHORD_STATES].max(axis=0) + 0.3)
    assert frac >= 0.9


def test_windowed_observations_hold_an_arpeggio():
    y, _ = arpeggio_signal(0, "maj", bpm=120.0, bars=2, ring_s=0.15)
    feats = compute_chroma(y, SR)
    backend = chords.TemplateBackend()
    target = chords.state_index(0, "maj")
    raw = chords.estimate_frame_chords(feats, backend, window_frames=1)
    win = chords.estimate_frame_chords(feats, backend, window_frames=5)
    assert np.mean(win.states == target) >= 0.85
    assert np.mean(win.states == target) >= np.mean(raw.states == target)
    assert chords.observation_window_frames(0.0929, 0.5, 0.9) == 5
    assert chords.observation_window_frames(0.0929, 0.5, 0.1) == 5          # low confidence → 0.5 s
    assert chords.observation_window_frames(0.0929, 0.25, 0.9) == 3         # clipped to the minimum
    assert chords.observation_window_frames(0.0929, 2.0, 0.9) == 9          # clipped to the maximum
    assert chords.observation_window_frames(0.0929, 0.5, 0.9, window_beats=0) == 1
    assert chords.expected_chord_seconds(0.5, 0.9) == pytest.approx(1.0)
    assert chords.expected_chord_seconds(None, 0.9) == 1.5


def test_fallback_recovers_a_chord_and_rejects_noise():
    y, _ = pad_under_drums(((0, "maj"),), beats_per_chord=8, drum_db=6.0)
    feats = compute_chroma(y, SR)
    all_none = chords.FrameChords(
        states=np.full(feats.n_frames, chords.NO_CHORD),
        probs=np.full((chords.N_STATES, feats.n_frames), 1.0 / chords.N_STATES),
        posterior=np.full((chords.N_STATES, feats.n_frames), 1.0 / chords.N_STATES),
    )
    seg = chords.fallback_segment(all_none, feats)
    assert seg is not None
    assert (seg.root, seg.quality) == (0, "maj")
    assert seg.confidence <= chords.FALLBACK_MAX_CONFIDENCE
    for bad in (noise(4.0, SR, 0.3), np.zeros(SR * 4, dtype=np.float32)):
        f = compute_chroma(bad, SR)
        empty = chords.FrameChords(
            states=np.full(f.n_frames, chords.NO_CHORD),
            probs=np.full((chords.N_STATES, f.n_frames), 1.0 / chords.N_STATES),
            posterior=np.full((chords.N_STATES, f.n_frames), 1.0 / chords.N_STATES),
        )
        assert chords.fallback_segment(empty, f) is None


def test_bare_root_rule():
    pluck = compute_chroma(oneshot_note(55, duration=1.5), SR)
    root, quality, bass, conf, _ = chords.label_segment(pluck, 0, pluck.n_frames, 7, "maj", allow_reroot=True)
    assert (root, quality) == (7, "5") and bass == 7 and conf > 0.0
    # a wrong root hypothesis is corrected only when re-rooting is allowed:
    # from D the strong G is a sus tone, so the bare-root rule stays silent
    _, wrong_q, _, _, _ = chords.label_segment(pluck, 0, pluck.n_frames, 2, "maj", allow_reroot=False)
    assert wrong_q != "5"
    fixed_root, fixed_q, _, _, _ = chords.label_segment(pluck, 0, pluck.n_frames, 2, "maj", allow_reroot=True)
    assert (fixed_root, fixed_q) == (7, "5")
    # sus chords and clean triads are never relabelled
    sus = compute_chroma(mono(((7, "sus4"),), beats_per_chord=4), SR)
    assert chords.label_segment(sus, 0, sus.n_frames, 7, "maj")[1] == "sus4"
    triad = compute_chroma(mono(((7, "maj"),), beats_per_chord=4), SR)
    assert chords.label_segment(triad, 0, triad.n_frames, 7, "maj")[1] == "maj"
