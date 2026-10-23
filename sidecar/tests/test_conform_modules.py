"""§6 tonal modules, one by one, on synthetic audio: direction and caps."""

import numpy as np
import pytest

from patine.conform import filters as F
from patine.conform.bus import bus_compressor, reverb, space_send_level
from patine.conform.loudness import match_loudness, measure_lufs
from patine.conform.lowend import align_lowend, lowend_decision
from patine.conform.noise import bed_level_db, bed_noise_floor, harvest_floor
from patine.conform.tilt import match_tilt, tilt_correction_db
from patine.conform.tuning import correct_tuning, tuning_shift_cents
from patine.conform.width import match_width, width_scales
from patine.models import TILT_BAND_CENTRES_HZ, TonalMetrics

SR = 48000


def noise(seconds: float = 2.0, channels: int = 2, level: float = 0.05, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((channels, int(seconds * SR))) * level).astype(np.float32)


def pinkish(seconds: float = 2.0) -> np.ndarray:
    h = F.fir_from_band_curve(SR, list(TILT_BAND_CENTRES_HZ), [0, -3, -6, -9, -12, -15, -18, -21, -24, -27], 2001)
    return F.apply_fir_zero_delay(noise(seconds), h)


# --- tuning ----------------------------------------------------------------------------

def test_tuning_target_is_anchor_or_a440():
    assert tuning_shift_cents(10.0, None) == pytest.approx(-10.0)
    assert tuning_shift_cents(10.0, 4.0) == pytest.approx(-6.0)
    assert tuning_shift_cents(None, 4.0) == 0.0


def test_tuning_resamples_and_reports_speed():
    y = noise(1.0)
    out, speed = correct_tuning(y, SR, -12.0)
    assert speed == pytest.approx(2 ** (-12 / 1200))
    assert out.shape[1] == int(round(SR / speed))
    same, s1 = correct_tuning(y, SR, 0.01)
    assert s1 == 1.0 and same.shape == y.shape


# --- loudness --------------------------------------------------------------------------

@pytest.mark.parametrize("target", [-18.0, -12.0, -30.0])
def test_loudness_matches_within_half_lu(target):
    y = noise(2.0)
    out, gain = match_loudness(y, SR, target)
    assert abs(measure_lufs(out, SR) - target) < 0.5
    assert gain == pytest.approx(target - measure_lufs(y, SR), abs=0.01)


def test_loudness_ignores_silence_and_missing_target():
    y = noise(1.0)
    out, gain = match_loudness(y, SR, None)
    assert gain == 0.0 and out is y
    silent = np.zeros((2, SR), dtype=np.float32)
    out, gain = match_loudness(silent, SR, -18.0)
    assert gain == 0.0


def test_loudness_short_clip():
    y = noise(0.3)
    out, gain = match_loudness(y, SR, -20.0)
    assert out.shape == y.shape and abs(gain) < 40


# --- tilt ------------------------------------------------------------------------------

def test_tilt_correction_cap_and_amount():
    region = [0.0] * 10
    anchor = [12.0, -9.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0]
    corr = tilt_correction_db(region, anchor, 100)
    assert corr == pytest.approx([6.0, -6.0, 3.0, 0, 0, 0, 0, 0, 0, -2.0])
    assert tilt_correction_db(region, anchor, 40) == pytest.approx([2.4, -2.4, 1.2, 0, 0, 0, 0, 0, 0, -0.8])
    assert tilt_correction_db(None, anchor, 40) == [0.0] * 10
    assert tilt_correction_db(region, [-120.0] * 10, 100) == [0.0] * 10


def test_tilt_moves_band_energy_toward_anchor_never_past_cap():
    y = pinkish()
    rb = F.octave_band_energy_db(y, SR)
    wanted = [0, 0, 10, 0, 0, -10, 0, 3, 0, 0]
    ab = [r + d for r, d in zip(rb, wanted, strict=True)]
    out, applied = match_tilt(y, SR, rb, ab, 100)
    assert out.shape == y.shape
    ob = F.octave_band_energy_db(out, SR)
    delta = [o - r for o, r in zip(ob, rb, strict=True)]
    assert delta[2] > 3.0 and delta[2] <= 6.3
    assert delta[5] < -3.0 and delta[5] >= -6.3
    assert delta[7] > 1.0 and delta[7] <= 3.3
    assert all(abs(d) <= 6.3 for d in delta)
    out40, _ = match_tilt(y, SR, rb, ab, 40)
    d40 = [o - r for o, r in zip(F.octave_band_energy_db(out40, SR), rb, strict=True)]
    assert 0.5 < d40[2] < delta[2]
    assert delta[5] < d40[5] < -0.5


def test_tilt_identity_when_equal():
    y = pinkish(1.0)
    rb = F.octave_band_energy_db(y, SR)
    out, applied = match_tilt(y, SR, rb, rb, 40)
    assert applied == [0.0] * 10
    assert np.array_equal(out, y)


# --- low end ---------------------------------------------------------------------------

def stereo_with_sub() -> np.ndarray:
    y = noise(2.0)
    sub = (np.sin(2 * np.pi * 40 * np.arange(2 * SR) / SR) * 0.3).astype(np.float32)
    y[0] += sub
    y[1] -= sub          # sub in the Side channel: wide bass
    return y


def test_lowend_hpf_and_mono_only_remove_energy():
    y = stereo_with_sub()
    rm = TonalMetrics(sub_ratio=0.5, bass_mono_hz=30.0)
    am = TonalMetrics(sub_ratio=0.05, low_rolloff_hz=80.0, bass_mono_hz=150.0)
    out, hpf, mono = align_lowend(y, SR, rm, am)
    assert (hpf, mono) == (80.0, 150.0)
    assert out.shape == y.shape
    assert F.band_energy(out, SR, 0, 80) < 0.1 * F.band_energy(y, SR, 0, 80)
    side_before = F.band_energy((y[0] - y[1]) / 2, SR, 0, 150)
    side_after = F.band_energy((out[0] - out[1]) / 2, SR, 0, 150)
    assert side_after < 0.02 * side_before
    # never adds: every octave band at or below the result's before level
    for lo, hi in ((0, 40), (40, 80), (80, 160), (160, 320), (320, 2000), (2000, 20000)):
        assert F.band_energy(out, SR, lo, hi) <= F.band_energy(y, SR, lo, hi) * 1.02 + 1e-12


def test_lowend_decision_never_adds_low_end():
    rm = TonalMetrics(sub_ratio=0.01, bass_mono_hz=200.0)
    am = TonalMetrics(sub_ratio=0.3, low_rolloff_hz=80.0, bass_mono_hz=150.0)
    assert lowend_decision(rm, am, 2) == (None, None)     # region already tighter than anchor
    assert lowend_decision(TonalMetrics(sub_ratio=0.5), am, 1) == (80.0, None)  # mono input: no mono step
    assert lowend_decision(None, am, 2) == (None, None)
    y = stereo_with_sub()
    out, hpf, mono = align_lowend(y, SR, rm, am)
    assert hpf is None and mono is None and np.array_equal(out, y)


# --- noise floor -----------------------------------------------------------------------

def gapped(floor_db: float, seed: int) -> np.ndarray:
    y = np.zeros((2, 2 * SR), dtype=np.float32)
    y[:, :SR] = noise(1.0, seed=seed)
    y += noise(2.0, level=10 ** (floor_db / 20), seed=seed + 10)
    return y


def test_noise_bed_raises_floor_toward_anchor_never_lowers():
    region = gapped(-80.0, 1)
    anchor = gapped(-50.0, 2)
    rf, af = F.noise_floor_db(region, SR), F.noise_floor_db(anchor, SR)
    assert rf < af - 20
    out, level = bed_noise_floor(region, SR, rf, af, anchor, None)
    assert level is not None and level == pytest.approx(bed_level_db(rf, af))
    after = F.noise_floor_db(out, SR)
    assert after > rf
    assert abs(after - af) < 2.0
    assert out.shape == region.shape
    # loud content untouched beyond the bed: energy only goes up
    assert np.mean(out ** 2) >= np.mean(region ** 2)


def test_noise_never_subtracts_when_region_is_noisier():
    region = gapped(-50.0, 1)
    anchor = gapped(-80.0, 2)
    rf, af = F.noise_floor_db(region, SR), F.noise_floor_db(anchor, SR)
    out, level = bed_noise_floor(region, SR, rf, af, anchor, None)
    assert level is None and np.array_equal(out, region)


def test_noise_harvest_and_synth_fallback():
    anchor = gapped(-50.0, 2)
    piece = harvest_floor(anchor, SR)
    assert piece is not None and piece.shape[1] >= 0.2 * SR
    assert F.rms_db(piece) < -40
    busy = noise(2.0)                          # no quiet stretch: all programme
    assert harvest_floor(busy, SR) is None or F.rms_db(harvest_floor(busy, SR)) > -40
    region = gapped(-80.0, 1)
    rf = F.noise_floor_db(region, SR)
    out, level = bed_noise_floor(region, SR, rf, -50.0, busy, [-60, -60, -63, -66, -70, -75, -80, -85, -90, -95])
    assert level is not None
    after = F.noise_floor_db(out, SR)
    assert rf < after and abs(after - (-50.0)) < 3.0
    # synthesised floor follows the anchor's spectrum: more low than high
    bed = out[:, SR + SR // 10:]
    assert F.band_energy(bed, SR, 50, 400) > F.band_energy(bed, SR, 4000, 12000)


# --- width -----------------------------------------------------------------------------

def test_width_scales_amount_and_cap():
    assert width_scales([1.0, 1.0, 1.0], [0.25, 4.0, 1.0], 100) == pytest.approx([0.5, 1.5, 1.0])
    assert width_scales([1.0, 1.0, 1.0], [0.25, 4.0, 1.0], 50) == pytest.approx([0.75, 1.5, 1.0])
    assert width_scales([1.0, 1.0, 1.0], [0.25, 1.21, 1.0], 50) == pytest.approx([0.75, 1.05, 1.0])
    assert width_scales([0.0, 1.0, 1.0], [1.0, 1.0, 1.0], 100) == [1.0, 1.0, 1.0]   # mono band stays
    assert width_scales(None, [1.0, 1.0, 1.0], 100) == [1.0, 1.0, 1.0]


def test_width_moves_side_per_band_within_cap():
    rng = np.random.default_rng(5)
    y = (np.stack([rng.standard_normal(2 * SR), rng.standard_normal(2 * SR)]) * 0.1).astype(np.float32)
    wb = F.width_bands(y, SR)
    out, scales = match_width(y, SR, wb, [0.25, 4.0, 1.0], 100)
    assert scales[1] == 1.5
    ob = F.width_bands(out, SR)
    assert ob[0] < 0.6 * wb[0]                        # narrowed
    assert 1.6 * wb[1] < ob[1] < 2.25 * wb[1] * 1.05  # widened, never past 1.5²
    assert abs(ob[2] / wb[2] - 1.0) < 0.1
    # mid untouched
    assert np.allclose((out[0] + out[1]) / 2, (y[0] + y[1]) / 2, atol=1e-6)


def test_width_leaves_mono_untouched():
    x = noise(1.0, channels=1)
    out, scales = match_width(x, SR, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 100)
    assert np.array_equal(out, x) and scales == [1.0, 1.0, 1.0]
    dual = np.repeat(x, 2, axis=0)
    out2, _ = match_width(dual, SR, F.width_bands(dual, SR), [1.0, 1.0, 1.0], 100)
    assert np.allclose(out2[0], out2[1], atol=1e-6)


# --- bus -------------------------------------------------------------------------------

def test_bus_compressor_reduces_loud_passages_gently():
    y = noise(3.0, level=0.3)
    y[:, SR:2 * SR] *= 0.2
    out, thresh = bus_compressor(y, SR)
    assert out.shape == y.shape
    loud_in = np.sqrt(np.mean(y[:, : SR] ** 2))
    loud_out = np.sqrt(np.mean(out[:, : SR] ** 2))
    gr = 20 * np.log10(loud_in / loud_out)
    assert 1.0 < gr < 4.0
    quiet_in = np.sqrt(np.mean(y[:, SR + 4000:2 * SR] ** 2))
    quiet_out = np.sqrt(np.mean(out[:, SR + 4000:2 * SR] ** 2))
    assert 20 * np.log10(quiet_in / quiet_out) < gr


def test_space_send_only_for_drier_regions():
    assert space_send_level(0.1, 0.5) == pytest.approx(0.4)
    assert space_send_level(0.6, 0.5) == 0.0
    assert space_send_level(None, 0.5) == 0.0
    imp = np.zeros((2, SR), dtype=np.float32)
    imp[:, 100] = 1.0
    wet = reverb(imp, SR)
    assert wet.shape == imp.shape
    assert np.abs(wet[:, SR // 2:]).max() > 1e-4      # a tail
    assert np.abs(wet[:, SR // 2:]).max() < np.abs(wet[:, :SR // 4]).max()
