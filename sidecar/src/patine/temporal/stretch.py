"""§7.5 Stretch engine: Signalsmith Stretch through the `python-stretch`
binding (MIT). Elastic = constant pitch; repitch = resample so pitch follows
tempo. Non-uniform warp forces elastic unless `allow_wobble`.

How the binding behaves (measured, python-stretch 0.3.1):
  * `process(x)` on a float32 (channels, frames) buffer returns exactly
    round(frames / timeFactor) frames with the engine's input and output
    latency already compensated — an impulse at sample i comes out at i / tf.
  * every `process` call is independent (the state is flushed per call), so
    the rate cannot change mid-stream; a non-uniform plan is rendered one
    constant-ratio run at a time.
  * the first and last few thousand output samples roll off because the
    engine has no context there, so every run is processed with real
    neighbouring source audio padded on both sides and the output is trimmed
    back to the run's exact planned length.
  * timeFactor = 1 is not bit-exact (it is still a phase vocoder), so a run
    whose ratio is 1 and that needs no pitch shift bypasses the engine: that
    is what makes the anchor identity (§6) hold.

Runs are joined at beat boundaries with a 1.5 ms linear crossfade (beat
boundaries are transients; a short join is inaudible). The total output
length is the plan's destination length to the sample."""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import soxr

from ..models import TemporalSettings, UserOverrides
from .warp import WarpPlan

log = logging.getLogger(__name__)

StretchMethod = Literal["elastic", "repitch"]

RUN_RATIO_TOL = 1e-4       # consecutive segments within this relative ratio share one engine pass
IDENTITY_TOL = 1e-6        # a run with |ratio - 1| below this is copied, not processed
JOIN_MS = 1.5              # crossfade at run joins (§7.6: beat boundaries are transients)
MIN_CONTEXT_S = 0.12       # context padded around every engine run


def stretch_plan(
    y: np.ndarray,
    sr: int,
    plan: WarpPlan,
    *,
    method: Literal["elastic", "repitch"] = "elastic",
    transpose_semitones: float = 0.0,
    declick_ms: float = 1.5,
) -> np.ndarray:
    """Render `plan` from source `y` (channels, frames). Returns (channels, frames)
    whose total length is sum(dst_len_s) * sr (±1 sample). Transposition is
    applied here too: elastic transposition is a pitch shift at constant length;
    plain (resample) transposition changes length and the caller compensates by
    folding the factor into the plan.

    Plan times index `y` (seconds from its first sample); the renderer rebases
    the plan onto its slice with `warp.rebase_plan`."""
    y = _as_2d(y)
    channels, n_src = y.shape
    if not plan.segments:
        return np.zeros((channels, 0), dtype=np.float32)

    runs = _group_runs(plan, sr, n_src)
    n_total = runs[-1][3] if runs else 0
    out = np.zeros((channels, n_total), dtype=np.float32)
    half = int(round(JOIN_MS / 1000.0 * sr / 2.0))

    for i, (a, b, o_start, o_end) in enumerate(runs):
        out_len = o_end - o_start
        if out_len <= 0 or b <= a:
            continue
        lead = min(half, o_start) if i > 0 else 0
        trail = min(half, n_total - o_end) if i < len(runs) - 1 else 0
        seg = _render_run(y, sr, a, b, out_len, lead, trail, method, transpose_semitones)
        _place(out, seg, o_start - lead, lead, trail)

    if declick_ms > 0:
        out = declick(out, sr, declick_ms)
    return out


def resample_ratio(y: np.ndarray, sr: int, ratio: float) -> np.ndarray:
    """Plain repitch: play at `ratio` × speed (ratio 2 = octave up, half length)."""
    y = _as_2d(y)
    n = y.shape[1]
    if n == 0 or abs(ratio - 1.0) < IDENTITY_TOL:
        return y.copy()
    target = int(round(n / ratio))
    res = soxr.resample(np.ascontiguousarray(y.T), float(sr), float(sr) / float(ratio), quality="HQ")
    res = np.ascontiguousarray(res.T.astype(np.float32, copy=False))
    return _fit_length(res, target)


def resolve_method(
    settings: TemporalSettings, overrides: UserOverrides, plan: WarpPlan
) -> tuple[StretchMethod, bool]:
    """§7.5: the region's method (override > set), forced to elastic when the
    plan is non-uniform and the user has not opted into wobble. Returns
    (method, forced)."""
    method: StretchMethod = overrides.stretch_method or settings.stretch_method
    if method == "repitch" and not plan.uniform and not overrides.allow_wobble:
        return "elastic", True
    return method, False


def declick(y: np.ndarray, sr: int, ms: float = 1.5) -> np.ndarray:
    """§7.6 linear ramps at the start and end. Returns a copy."""
    y = _as_2d(y).copy()
    n = y.shape[1]
    r = min(int(round(ms / 1000.0 * sr)), n // 2)
    if r <= 0:
        return y
    ramp = np.linspace(0.0, 1.0, r, endpoint=False, dtype=np.float32)
    y[:, :r] *= ramp
    y[:, n - r:] *= ramp[::-1]
    return y


# --- internals -----------------------------------------------------------------

def _as_2d(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 1:
        y = y[None, :]
    return np.ascontiguousarray(y)


def _fit_length(y: np.ndarray, n: int) -> np.ndarray:
    if y.shape[1] == n:
        return y
    if y.shape[1] > n:
        return np.ascontiguousarray(y[:, :n])
    return np.pad(y, ((0, 0), (0, n - y.shape[1])))


def _group_runs(plan: WarpPlan, sr: int, n_src: int) -> list[tuple[int, int, int, int]]:
    """Merge consecutive segments of (nearly) equal ratio into runs and map
    them to integer source and output sample ranges (a, b, out_start, out_end).
    Output edges come from the cumulative destination time so rounding never
    accumulates."""
    runs: list[tuple[int, int, int, int]] = []
    t_dst = 0.0
    o_prev = 0
    cur: list[int] | None = None   # [a, b, o_start, o_end]
    cur_ratio = 1.0
    for seg in plan.segments:
        a = max(0, min(int(round(seg.src_start_s * sr)), n_src))
        b = max(0, min(int(round(seg.src_end_s * sr)), n_src))
        t_dst += seg.dst_len_s
        o_end = int(round(t_dst * sr))
        if cur is not None and a == cur[1] and abs(seg.ratio / cur_ratio - 1.0) <= RUN_RATIO_TOL:
            cur[1], cur[3] = b, o_end
        else:
            if cur is not None:
                runs.append(tuple(cur))  # type: ignore[arg-type]
            cur, cur_ratio = [a, b, o_prev, o_end], seg.ratio
        o_prev = o_end
    if cur is not None:
        runs.append(tuple(cur))  # type: ignore[arg-type]
    return runs


def _context_slice(y: np.ndarray, a: int, b: int, ctx_l: int, ctx_r: int) -> np.ndarray:
    """y[a - ctx_l : b + ctx_r] with zeros where the source has no samples, so
    the slice is always exactly ctx_l + (b - a) + ctx_r long."""
    n = y.shape[1]
    lo, hi = a - ctx_l, b + ctx_r
    core = y[:, max(0, lo):min(n, hi)]
    pad_l = max(0, -lo)
    pad_r = max(0, hi - n)
    if pad_l or pad_r:
        core = np.pad(core, ((0, 0), (pad_l, pad_r)))
    return np.ascontiguousarray(core, dtype=np.float32)


def _render_run(
    y: np.ndarray,
    sr: int,
    a: int,
    b: int,
    out_len: int,
    lead: int,
    trail: int,
    method: str,
    transpose: float,
) -> np.ndarray:
    """One constant-ratio run → exactly lead + out_len + trail samples, the
    first of which sits `lead` output samples before the run's start."""
    ratio = out_len / (b - a)                   # dst / src
    want = lead + out_len + trail
    identity = abs(ratio - 1.0) < IDENTITY_TOL

    if identity and abs(transpose) < 1e-9:
        return _fit_length(_context_slice(y, a, b, lead, trail), want)

    if method == "repitch":
        ctx = int(round(MIN_CONTEXT_S * sr))
        res = resample_ratio(_context_slice(y, a, b, ctx, ctx), sr, 1.0 / ratio)
        run_at = int(round(ctx * ratio))          # where the run starts inside `res`
        if abs(transpose) < 1e-9:
            return _window(res, run_at - lead, want)
        # pitch followed tempo for the stretch; the transposition is elastic on top
        return _engine_pass(res, sr, run_at, run_at + out_len, out_len, 1.0, transpose, lead, trail)

    return _engine_pass(y, sr, a, b, out_len, ratio, transpose, lead, trail)


def _engine_pass(
    src: np.ndarray,
    sr: int,
    a: int,
    b: int,
    out_len: int,
    ratio: float,
    transpose: float,
    lead: int,
    trail: int,
) -> np.ndarray:
    """Run Signalsmith Stretch over src[a:b] padded with real neighbouring
    audio (zeros beyond the buffer) at dst/src `ratio` with `transpose`
    semitones. Returns lead + out_len + trail output samples starting `lead`
    samples before the stretched src[a]."""
    import python_stretch as ps

    engine = ps.Signalsmith.Stretch()
    engine.preset(src.shape[0], float(sr))
    ctx = max(int(engine.blockSamples()), int(round(MIN_CONTEXT_S * sr)))
    padded = _context_slice(src, a, b, ctx, ctx)
    engine.timeFactor = float(1.0 / ratio)
    if abs(transpose) >= 1e-9:
        engine.setTransposeSemitones(float(transpose))
    processed = engine.process(np.ascontiguousarray(padded, dtype=np.float32))
    run_at = int(round(ctx * ratio))
    return _window(processed, run_at - lead, lead + out_len + trail)


def _window(x: np.ndarray, start: int, n: int) -> np.ndarray:
    """x[:, start:start+n], zero-padded where x is too short."""
    out = np.zeros((x.shape[0], n), dtype=np.float32)
    lo = max(0, start)
    hi = min(x.shape[1], start + n)
    if hi > lo:
        out[:, lo - start:hi - start] = x[:, lo:hi]
    return out


def _place(out: np.ndarray, seg: np.ndarray, at: int, lead: int, trail: int) -> None:
    """Add `seg` at output sample `at`. A run's `lead` samples are the half of
    a join before its boundary and `trail` the half after; the join crossfade
    spans both halves (2·lead in, 2·trail out) so the complementary linear
    ramps of neighbouring runs sum to unity across the boundary."""
    seg = seg.copy()
    n = seg.shape[1]
    fi = min(2 * lead, n)
    fo = min(2 * trail, n)
    if fi > 0:
        seg[:, :fi] *= np.linspace(0.0, 1.0, fi, endpoint=False, dtype=np.float32)
    if fo > 0:
        seg[:, n - fo:] *= np.linspace(1.0, 0.0, fo, endpoint=False, dtype=np.float32)
    out[:, at:at + n] += seg
