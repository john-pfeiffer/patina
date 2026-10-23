"""§4.8 Tonal metrics, per region and per whole file. These measure what makes
two recordings read as *different recordings*; the conform layer (§6) moves
them toward the anchor and never touches the music.

| Metric | How |
|---|---|
| `loudness_lufs` | ITU-R BS.1770 integrated (`pyloudnorm`); regions shorter than 1 s are tiled so the gated measurement is defined |
| `tilt_bands[10]` | Welch long-term average spectrum collapsed to 10 octave bands centred 31.5 Hz–16 kHz, band power in dB (full-scale sine = -3 dB) |
| `low_rolloff_hz` | frequency below which the spectrum sits >12 dB under the 100–300 Hz level |
| `bass_mono_hz` | highest frequency below which the L/R correlation stays >0.9 (Nyquist for mono) |
| `sub_ratio` | energy below 60 Hz over total energy |
| `noise_floor_db`, `noise_spectrum[10]` | level and octave-band shape of the quietest 5 % of ~90 ms frames |
| `width_bands[3]` | Side/Mid energy ratio in <200 Hz, 200 Hz–3 kHz, >3 kHz (0 for mono) |
| `crest_db` | peak-to-RMS |
| `wetness` | decay slope after onsets mapped to 0..1 (0 = dry hit, 1 = no decay) |

All spectra come from one Hann/50 % Welch pass over the native-rate channels
so a 3-minute file costs well under a second."""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass

import numpy as np
import pyloudnorm as pyln
from scipy.signal.windows import hann

from ..models import TILT_BAND_CENTRES_HZ, TonalMetrics

log = logging.getLogger(__name__)

DB_FLOOR = -120.0
_EPS = 1e-12
_SQRT2 = math.sqrt(2.0)


# --- spectra -------------------------------------------------------------------------

@dataclass
class Spectra:
    freqs: np.ndarray            # (F,)
    df: float
    mono: np.ndarray             # PSD density of the mono mix (|L+R|²/4 for stereo)
    left: np.ndarray | None = None
    right: np.ndarray | None = None
    cross: np.ndarray | None = None   # Re(L · conj(R))
    side: np.ndarray | None = None    # PSD of (L-R)/2

    @property
    def stereo(self) -> bool:
        return self.left is not None


def _welch_nperseg(sr: int, n: int) -> int:
    nperseg = 2 ** int(round(math.log2(max(sr * 0.19, 64))))
    nperseg = int(np.clip(nperseg, 1024, 32768))
    if n < nperseg:
        nperseg = max(64, 2 ** int(math.floor(math.log2(max(n, 64)))))
    return nperseg


def compute_spectra(y: np.ndarray, sr: int, *, chunk_frames: int = 256) -> Spectra:
    """Welch PSDs (density, one-sided) of the channels, their mono mix, their
    difference and their cross-spectrum, from a single framing pass."""
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 1:
        y = y[None, :]
    ch, n = y.shape
    nperseg = _welch_nperseg(sr, n)
    if n < nperseg:
        y = np.pad(y, ((0, 0), (0, nperseg - n)))
        n = nperseg
    hop = nperseg // 2
    n_frames = 1 + (n - nperseg) // hop
    win = hann(nperseg, sym=False).astype(np.float32)
    scale = 1.0 / (sr * float(np.sum(win**2)))
    stereo = ch >= 2
    nf = nperseg // 2 + 1
    acc_m = np.zeros(nf)
    acc_l = np.zeros(nf) if stereo else None
    acc_r = np.zeros(nf) if stereo else None
    acc_x = np.zeros(nf) if stereo else None
    acc_s = np.zeros(nf) if stereo else None
    for start in range(0, n_frames, chunk_frames):
        idx = np.arange(start, min(n_frames, start + chunk_frames))
        pos = idx[:, None] * hop + np.arange(nperseg)[None, :]
        frames = y[:, pos] * win[None, None, :]                     # (ch, k, nperseg)
        spec = np.fft.rfft(frames, axis=-1)                           # (ch, k, nf)
        if stereo:
            left, right = spec[0], spec[1]
            mid = 0.5 * (left + right)
            acc_m += np.sum(np.abs(mid) ** 2, axis=0)
            acc_l += np.sum(np.abs(left) ** 2, axis=0)
            acc_r += np.sum(np.abs(right) ** 2, axis=0)
            acc_x += np.sum((left * np.conj(right)).real, axis=0)
            acc_s += np.sum(np.abs(0.5 * (left - right)) ** 2, axis=0)
        else:
            acc_m += np.sum(np.abs(spec[0]) ** 2, axis=0)
    one_sided = np.full(nf, 2.0)
    one_sided[0] = 1.0
    if nperseg % 2 == 0:
        one_sided[-1] = 1.0
    norm = scale * one_sided / max(n_frames, 1)
    freqs = np.fft.rfftfreq(nperseg, d=1.0 / sr)
    return Spectra(
        freqs=freqs,
        df=float(sr) / nperseg,
        mono=acc_m * norm,
        left=None if acc_l is None else acc_l * norm,
        right=None if acc_r is None else acc_r * norm,
        cross=None if acc_x is None else acc_x * norm,
        side=None if acc_s is None else acc_s * norm,
    )


def _db(power: float) -> float:
    return float(max(DB_FLOOR, 10.0 * math.log10(max(float(power), _EPS))))


def band_power(freqs: np.ndarray, psd: np.ndarray, df: float, lo: float, hi: float) -> float:
    sel = (freqs >= lo) & (freqs < hi)
    return float(np.sum(psd[sel]) * df)


def octave_band_levels_db(freqs: np.ndarray, psd: np.ndarray, df: float) -> list[float]:
    """Band power (dB) in the 10 octave bands of `TILT_BAND_CENTRES_HZ`."""
    return [_db(band_power(freqs, psd, df, c / _SQRT2, c * _SQRT2)) for c in TILT_BAND_CENTRES_HZ]


# --- individual metrics -----------------------------------------------------------------

_METERS: dict[int, pyln.Meter] = {}


def loudness_lufs(y: np.ndarray, sr: int) -> float | None:
    """BS.1770 integrated loudness. Short regions are tiled to one second so
    the gated measurement exists; silence returns None."""
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 1:
        y = y[None, :]
    n = y.shape[1]
    if n == 0 or float(np.max(np.abs(y))) <= 0.0:
        return None
    data = y.T
    min_len = int(sr)
    if n < min_len:
        data = np.tile(data, (int(math.ceil(min_len / n)), 1))
    if data.shape[1] > 5:
        data = data[:, :5]
    meter = _METERS.get(sr)
    if meter is None:
        meter = pyln.Meter(sr)
        _METERS[sr] = meter
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        value = float(meter.integrated_loudness(np.ascontiguousarray(data, dtype=np.float64)))
    return value if math.isfinite(value) else None


def low_rolloff_hz(spectra: Spectra) -> float | None:
    """Frequency below which the spectrum (1/12-octave smoothed density) drops
    more than 12 dB under the mean 100–300 Hz level. 20 Hz when nothing rolls
    off above 20 Hz; None when there is no 100–300 Hz content to compare with."""
    f = spectra.freqs
    if f.size < 4:
        return None
    grid = 20.0 * 2.0 ** (np.arange(0, 12 * 5 + 1) / 12.0)        # 20 Hz .. 640 Hz
    grid = grid[grid < f[-1]]
    psd_db = 10.0 * np.log10(np.maximum(spectra.mono[1:], _EPS))
    on_grid = np.interp(np.log(grid), np.log(f[1:]), psd_db)
    kernel = np.ones(4) / 4.0
    smooth = np.convolve(np.pad(on_grid, (2, 1), mode="edge"), kernel, mode="valid")
    ref_sel = (grid >= 100.0) & (grid <= 300.0)
    if not np.any(ref_sel):
        return None
    ref = float(smooth[ref_sel].mean())
    if ref < -100.0:
        return None
    below = np.where(grid < 100.0)[0]
    for i in below[::-1]:
        if smooth[i] < ref - 12.0:
            return float(grid[min(i + 1, grid.size - 1)])
    return float(grid[0])


def _sixth_octave_bands(f_lo: float, f_hi: float) -> np.ndarray:
    n = int(math.floor(6.0 * math.log2(max(f_hi, f_lo * 1.0001) / f_lo))) + 1
    return f_lo * 2.0 ** (np.arange(n + 1) / 6.0)


def bass_mono_hz(spectra: Spectra, sr: int, *, threshold: float = 0.9, gate_db: float = -40.0) -> float | None:
    """Highest frequency below which the L/R correlation stays above
    `threshold`: 1/6-octave bands are scanned upward and the lower edge of
    the first decorrelated band is returned. Bands more than `gate_db` under
    the loudest band carry no image worth measuring (the noise floor in a
    gap between harmonics is uncorrelated but it is not bass) and are
    skipped. Nyquist for mono or fully correlated input; None for silence."""
    nyquist = float(sr) / 2.0
    if not spectra.stereo:
        return nyquist
    assert spectra.left is not None and spectra.right is not None and spectra.cross is not None
    edges = _sixth_octave_bands(20.0, nyquist)
    f = spectra.freqs
    levels, corrs, lows = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        sel = (f >= lo) & (f < hi)
        if not np.any(sel):
            continue
        pxx = float(spectra.left[sel].sum())
        pyy = float(spectra.right[sel].sum())
        pxy = float(spectra.cross[sel].sum())
        levels.append(pxx + pyy)
        corrs.append(pxy / (math.sqrt(pxx) * math.sqrt(pyy) + 1e-30))
        lows.append(float(lo))
    if not levels:
        return nyquist
    peak = max(levels)
    if peak <= 1e-20:
        return None
    gate = peak * 10.0 ** (gate_db / 10.0)
    for level, corr, lo in zip(levels, corrs, lows, strict=True):
        if level >= gate and corr < threshold:
            return lo
    return nyquist


def sub_ratio(spectra: Spectra) -> float | None:
    total = float(np.sum(spectra.mono) * spectra.df)
    if total <= 1e-20:
        return None
    sub = band_power(spectra.freqs, spectra.mono, spectra.df, 0.0, 60.0)
    return float(np.clip(sub / total, 0.0, 1.0))


def noise_floor(mono: np.ndarray, sr: int) -> tuple[float | None, list[float]]:
    """Level (dB) and octave-band shape of the quietest 5 % of ~90 ms frames.
    Digitally silent frames (< -100 dB) are left out when anything else exists."""
    x = np.asarray(mono, dtype=np.float32)
    nfr = int(np.clip(2 ** int(round(math.log2(max(sr * 0.09, 64)))), 256, 16384))
    if x.size < nfr:
        x = np.pad(x, (0, nfr - x.size))
    hop = nfr // 2
    n_frames = 1 + (x.size - nfr) // hop
    # per-frame mean square from a cumulative sum of squares: one pass, no frame matrix
    cs = np.concatenate([[0.0], np.cumsum(x.astype(np.float64) ** 2)])
    starts = np.arange(n_frames) * hop
    ms = (cs[starts + nfr] - cs[starts]) / nfr
    level_db = 10.0 * np.log10(np.maximum(ms, _EPS))
    valid = np.where(level_db > -100.0)[0]
    if valid.size == 0:
        return DB_FLOOR, [DB_FLOOR] * 10
    k = max(1, int(math.ceil(0.05 * valid.size)))
    order = valid[np.argsort(level_db[valid], kind="stable")[:k]]
    floor_db = _db(float(ms[order].mean()))
    win = hann(nfr, sym=False)
    frames = x[starts[order][:, None] + np.arange(nfr)[None, :]]
    spec = np.fft.rfft(frames * win[None, :], axis=-1)
    psd = np.mean(np.abs(spec) ** 2, axis=0) / (sr * float(np.sum(win**2)))
    psd[1:-1] *= 2.0
    freqs = np.fft.rfftfreq(nfr, d=1.0 / sr)
    return floor_db, octave_band_levels_db(freqs, psd, float(sr) / nfr)


def width_bands(spectra: Spectra, sr: int) -> list[float]:
    if not spectra.stereo:
        return [0.0, 0.0, 0.0]
    assert spectra.side is not None
    edges = [(0.0, 200.0), (200.0, 3000.0), (3000.0, float(sr))]
    out = []
    for lo, hi in edges:
        mid = band_power(spectra.freqs, spectra.mono, spectra.df, lo, hi)
        side = band_power(spectra.freqs, spectra.side, spectra.df, lo, hi)
        if mid <= 1e-20 and side <= 1e-20:
            out.append(0.0)
        else:
            out.append(float(np.clip(side / max(mid, 1e-20), 0.0, 10.0)))
    return out


def crest_db(y: np.ndarray) -> float | None:
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        return None
    peak = float(np.max(np.abs(y)))
    rms = float(np.sqrt(np.mean(y.astype(np.float64) ** 2)))
    if rms <= 1e-9 or peak <= 0.0:
        return None
    return float(20.0 * math.log10(peak / rms))


def _rms_envelope_db(x: np.ndarray, sr: int, *, win_s: float = 0.02, hop_s: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    win = max(8, int(win_s * sr))
    hop = max(1, int(hop_s * sr))
    if x.size < win:
        x = np.pad(x, (0, win - x.size))
    n_frames = 1 + (x.size - win) // hop
    pos = np.arange(n_frames)[:, None] * hop + np.arange(win)[None, :]
    ms = np.mean(x[pos].astype(np.float64) ** 2, axis=1)
    times = (np.arange(n_frames) * hop + win / 2) / sr
    return times, 10.0 * np.log10(np.maximum(ms, _EPS))


def wetness(mono: np.ndarray, sr: int, onsets_s: list[float] | None = None, *, window_s: float = 0.4) -> float | None:
    """Decay slope after onsets mapped to 0..1: a dry hit decaying faster
    than 80 dB/s scores 0, a sustained or heavily reverberant sound 1."""
    x = np.asarray(mono, dtype=np.float32)
    duration = x.size / float(sr)
    if duration < 0.15:
        return None
    times, env = _rms_envelope_db(x, sr)
    anchors = sorted(o for o in (onsets_s or []) if 0.0 <= o < duration - 0.1) or [0.0]

    def slopes(respect_next: bool) -> list[float]:
        out = []
        for i, o in enumerate(anchors):
            nxt = anchors[i + 1] if (respect_next and i + 1 < len(anchors)) else duration
            end = min(o + window_s, nxt - 0.01, duration)
            if end - o < 0.15:
                continue
            sel = (times >= o + 0.03) & (times < end)
            if sel.sum() < 4:
                continue
            out.append(float(np.polyfit(times[sel], env[sel], 1)[0]))
        return out

    s = slopes(True) or slopes(False)
    if not s:
        return None
    slope = float(np.median(s))
    return float(np.clip(1.0 + slope / 80.0, 0.0, 1.0))


# --- all together ------------------------------------------------------------------------

def measure_tonal_metrics(y: np.ndarray, sr: int, *, onsets_s: list[float] | None = None) -> TonalMetrics:
    """All §4.8 metrics for audio `y` shaped (channels, frames) at `sr`.
    `onsets_s` are onset times relative to the start of `y` (for wetness)."""
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 1:
        y = y[None, :]
    if y.shape[1] == 0:
        return TonalMetrics()
    mono = y.mean(axis=0).astype(np.float32) if y.shape[0] > 1 else y[0]
    spectra = compute_spectra(y, sr)
    floor_db, noise_spec = noise_floor(mono, sr)
    return TonalMetrics(
        loudness_lufs=loudness_lufs(y, sr),
        tilt_bands=octave_band_levels_db(spectra.freqs, spectra.mono, spectra.df),
        low_rolloff_hz=low_rolloff_hz(spectra),
        bass_mono_hz=bass_mono_hz(spectra, sr),
        sub_ratio=sub_ratio(spectra),
        noise_floor_db=floor_db,
        noise_spectrum=noise_spec,
        width_bands=width_bands(spectra, sr),
        crest_db=crest_db(y),
        wetness=wetness(mono, sr, onsets_s),
    )
