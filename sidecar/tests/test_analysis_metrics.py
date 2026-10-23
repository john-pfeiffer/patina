"""Tonal metrics (handoff §4.8) on signals with known answers."""

from __future__ import annotations

import numpy as np
import pytest
from test_analysis_synth import chord_signal, clicks, filtered_noise, noise, sine

from patine.analysis import metrics as M
from patine.models import TonalMetrics

SR = 44100


def test_loudness_of_minus_20_dbfs_sine():
    s = sine(1000.0, 3.0, SR, amp=0.1)                     # -20 dBFS peak
    assert M.loudness_lufs(np.stack([s, s]), SR) == pytest.approx(-20.0, abs=0.3)
    assert M.loudness_lufs(s, SR) == pytest.approx(-23.0, abs=0.3)   # one channel
    short = np.stack([s[: int(0.2 * SR)]] * 2)              # below the 400 ms gating block
    assert M.loudness_lufs(short, SR) == pytest.approx(-20.0, abs=0.3)
    assert M.loudness_lufs(np.zeros((2, SR), dtype=np.float32), SR) is None


def test_width_bands_stereo_vs_mono():
    left = noise(2.0, SR, seed=1)
    right = noise(2.0, SR, seed=2)
    wide = M.width_bands(M.compute_spectra(np.stack([left, right]), SR), SR)
    assert all(w > 0.5 for w in wide)
    dual = M.width_bands(M.compute_spectra(np.stack([left, left]), SR), SR)
    assert all(w == pytest.approx(0.0, abs=1e-6) for w in dual)
    assert M.measure_tonal_metrics(left, SR).width_bands == [0.0, 0.0, 0.0]


def test_sub_ratio():
    assert M.sub_ratio(M.compute_spectra(sine(40.0, 3.0, SR), SR)) > 0.95
    assert M.sub_ratio(M.compute_spectra(sine(1000.0, 3.0, SR), SR)) < 0.01


def test_tilt_bands_follow_the_spectrum():
    white = M.compute_spectra(noise(3.0, SR), SR)
    tilt = M.octave_band_levels_db(white.freqs, white.mono, white.df)
    assert len(tilt) == 10
    assert tilt[8] - tilt[3] == pytest.approx(15.0, abs=3.0)     # +3 dB per octave of band power
    low = M.compute_spectra(filtered_noise(3.0, SR, 1000.0, "low"), SR)
    tilt_low = M.octave_band_levels_db(low.freqs, low.mono, low.df)
    assert tilt_low[4] == pytest.approx(tilt[4], abs=1.0)
    assert tilt_low[7] < tilt[7] - 30.0
    assert tilt_low[9] == M.DB_FLOOR


def test_low_rolloff():
    assert M.low_rolloff_hz(M.compute_spectra(noise(3.0, SR), SR)) <= 25.0
    hp = M.low_rolloff_hz(M.compute_spectra(filtered_noise(3.0, SR, 80.0, "high"), SR))
    assert hp is not None and 40.0 <= hp <= 100.0
    assert M.low_rolloff_hz(M.compute_spectra(sine(1000.0, 3.0, SR), SR)) is None


def test_bass_mono_hz():
    bass = sine(80.0, 3.0, SR, amp=0.3)
    left = bass + filtered_noise(3.0, SR, 2000.0, "high", amp=0.15, seed=1)
    right = bass + filtered_noise(3.0, SR, 2000.0, "high", amp=0.15, seed=2)
    f = M.bass_mono_hz(M.compute_spectra(np.stack([left, right]), SR), SR)
    assert f is not None and 1000.0 <= f <= 2500.0
    assert M.bass_mono_hz(M.compute_spectra(np.stack([bass, bass]), SR), SR) == SR / 2
    assert M.bass_mono_hz(M.compute_spectra(bass, SR), SR) == SR / 2
    assert M.bass_mono_hz(M.compute_spectra(np.zeros((2, SR), dtype=np.float32), SR), SR) is None


def test_noise_floor_is_the_quietest_frames():
    tone = sine(440.0, 4.0, SR, amp=0.3)
    tone[int(1.0 * SR) : int(1.6 * SR)] = 0.0                # a gap where only the floor remains
    floor = noise(4.0, SR, amp=0.001)                      # -60 dBFS
    level, spectrum = M.noise_floor(tone + floor, SR)
    assert level == pytest.approx(-60.0, abs=2.0)
    assert len(spectrum) == 10
    assert all(b <= level + 0.5 for b in spectrum)
    assert M.noise_floor(np.zeros(SR, dtype=np.float32), SR)[0] == M.DB_FLOOR


def test_crest_and_wetness():
    s = sine(1000.0, 3.0, SR, amp=0.1)
    assert M.crest_db(s) == pytest.approx(3.01, abs=0.05)
    hits, onsets = clicks(3.0, SR, 0.5)
    assert M.crest_db(hits) > 15.0
    assert M.wetness(hits, SR, onsets) < 0.2                # dry, fast decay
    assert M.wetness(s, SR, [0.0, 0.5, 1.0]) > 0.9           # no decay at all
    tail, _ = clicks(3.0, SR, 0.5, decay_s=0.4)
    wet = M.wetness(tail, SR, onsets)
    assert 0.4 < wet < 0.95
    assert M.wetness(s[: int(0.1 * SR)], SR) is None
    assert M.crest_db(np.zeros(100, dtype=np.float32)) is None


def test_measure_tonal_metrics_populates_everything():
    y, _ = chord_signal(sr=SR, beats_per_chord=2)
    m = M.measure_tonal_metrics(y, SR, onsets_s=[0.0, 0.5, 1.0])
    assert isinstance(m, TonalMetrics)
    assert m.loudness_lufs is not None and m.low_rolloff_hz is not None
    assert m.bass_mono_hz is not None and m.sub_ratio is not None
    assert m.noise_floor_db is not None and m.crest_db is not None and m.wetness is not None
    assert len(m.tilt_bands) == 10 and len(m.noise_spectrum) == 10 and len(m.width_bands) == 3
    assert M.measure_tonal_metrics(np.zeros((2, 0), dtype=np.float32), SR) == TonalMetrics()
