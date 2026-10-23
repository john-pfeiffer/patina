"""§6 bus-level modules: Density (a shared 2:1 compressor targeting 2–3 dB of
gain reduction) and Space (an algorithmic reverb send applied only to regions
drier than the anchor). Neither is rendered into region files; the UI runs
them in its playback graph and export's full mix runs these offline twins."""

from __future__ import annotations

import numpy as np
from scipy import signal

from .filters import as_2d

TARGET_GR_DB = 2.5


def _envelope_db(x: np.ndarray, sr: int, attack_ms: float, release_ms: float) -> np.ndarray:
    """Peak-following envelope in dB with separate attack/release (one-pole)."""
    a_att = float(np.exp(-1.0 / (attack_ms / 1000.0 * sr)))
    a_rel = float(np.exp(-1.0 / (release_ms / 1000.0 * sr)))
    rect = np.abs(x).astype(np.float64)
    env = np.empty_like(rect)
    e = 0.0
    for i, v in enumerate(rect):
        coef = a_att if v > e else a_rel
        e = coef * e + (1.0 - coef) * v
        env[i] = e
    return 20.0 * np.log10(np.maximum(env, 1e-9))


def bus_compressor(
    y: np.ndarray,
    sr: int,
    *,
    ratio: float = 2.0,
    attack_ms: float = 30.0,
    target_gr_db: float = TARGET_GR_DB,
) -> tuple[np.ndarray, float]:
    """Stereo-linked feed-forward compressor. The threshold is chosen from the
    programme so that the loud passages see about `target_gr_db` of gain
    reduction; release is automatic (fast for short overs, slow for sustained
    ones). Returns (audio, threshold used in dBFS)."""
    y = as_2d(y)
    n = y.shape[1]
    if n < 64 or target_gr_db <= 0:
        return y, 0.0
    # control signal: decimated for speed (envelope at ~ 1 kHz), then interpolated
    dec = max(1, sr // 1000)
    link = np.max(np.abs(y), axis=0)
    ctrl = link[: (n // dec) * dec].reshape(-1, dec).max(axis=1) if n >= dec else link
    csr = sr / dec
    fast = _envelope_db(ctrl, int(round(csr)), attack_ms, 120.0)
    slow = _envelope_db(ctrl, int(round(csr)), attack_ms, 600.0)
    level = np.maximum(fast, slow)       # auto release: the slower of the two holds sustained GR
    loud = level[level > -60.0]
    if loud.size == 0:
        return y, 0.0
    # 2:1 → GR = (L - T) / 2; set T so the 90th percentile sees target GR
    thresh = float(np.percentile(loud, 90.0) - target_gr_db * ratio / (ratio - 1.0))
    over = np.maximum(level - thresh, 0.0)
    gr_db = over * (1.0 - 1.0 / ratio)
    gain = 10.0 ** (-gr_db / 20.0)
    t_ctrl = np.arange(gain.size) * dec
    g = np.interp(np.arange(n), t_ctrl, gain) if gain.size > 1 else np.full(n, gain[0])
    return (y * g[None, :].astype(np.float32)).astype(np.float32), thresh


def space_send_level(region_wetness: float | None, anchor_wetness: float | None) -> float:
    """Send level ∝ wetness gap; 0 for regions as wet as or wetter than the
    anchor (they are never dried)."""
    if region_wetness is None or anchor_wetness is None:
        return 0.0
    gap = float(anchor_wetness) - float(region_wetness)
    return float(min(max(gap, 0.0), 1.0))


def _feedback_comb(x: np.ndarray, d: int, g: float) -> np.ndarray:
    """y[n] = x[n] + g · y[n - d], computed one delay-length block at a time."""
    n = x.size
    y = np.zeros(n + d)
    y[:d] = x[:d] if n >= d else np.pad(x, (0, d - n))
    for k in range(d, n, d):
        blk = x[k:k + d]
        y[k:k + blk.size] = blk + g * y[k - d:k - d + blk.size]
    return y[:n]


def _allpass(x: np.ndarray, d: int, g: float) -> np.ndarray:
    """y[n] = -g · x[n] + x[n - d] + g · y[n - d], blockwise."""
    n = x.size
    y = np.zeros(n + d)
    xp = np.concatenate([np.zeros(d), x])
    for k in range(0, n, d):
        m = min(d, n - k)
        y[k:k + m] = -g * x[k:k + m] + xp[k:k + m] + g * y[k - d:k - d + m] if k >= d else -g * x[k:k + m] + xp[k:k + m]
    return y[:n]


def reverb(y: np.ndarray, sr: int, *, decay_s: float = 1.8, damping: float = 0.35, seed: int = 7) -> np.ndarray:
    """Small Schroeder/Moorer algorithmic reverb (4 parallel combs with
    one-pole damping into 2 series all-passes), 100 % wet, same length as the
    input."""
    y = as_2d(y)
    ch, n = y.shape
    if n == 0:
        return y.copy()
    rng = np.random.default_rng(seed)
    comb_ms = np.array([29.7, 37.1, 41.1, 43.7])
    out = np.zeros_like(y)
    for c in range(ch):
        x = y[c].astype(np.float64)
        acc = np.zeros(n)
        for k, ms in enumerate(comb_ms):
            d = max(8, int(round((ms + 1.3 * k * (1 if c == 0 else -1)) / 1000.0 * sr)))
            g = 10.0 ** (-3.0 * d / sr / decay_s)
            comb = _feedback_comb(x, d, g)
            comb = signal.lfilter([1.0 - damping], [1.0, -damping], comb)
            acc += comb * (0.8 + 0.4 * rng.random())
        acc /= comb_ms.size
        for ms, g in ((5.0, 0.7), (1.7, 0.7)):
            acc = _allpass(acc, max(4, int(round(ms / 1000.0 * sr))), g)
        out[c] = acc.astype(np.float32)
    return out
