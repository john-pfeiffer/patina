"""§7.1 Tempo and §7.3 beat grid for one region.

Two paths, declared first:
  * declared_length_beats set  -> bpm = 60 * beats / duration; linear grid (tier 3)
  * else detected              -> sample bpm × tempo_multiplier, feel-adjusted;
                                  the tracked beats inside the region (tier 2)
                                  become the grid; fall back to a linear grid
                                  when the tracker found fewer than 2 beats.
Manual markers (tier 1) are applied in warp.py, on top of this grid.

Grid origin (§7.2): beat 0 of the grid is the region's *snap point* — its
`first_onset_s` when transient snap is on and the region does not keep its
pre-roll, otherwise `start_s`. Sample 0 of every render is that point, so
retrigger (§7.6) restarts from it and the kit's beat markers count from it.
Nudge is a playback adjustment applied in warp.py; it does not move the grid.

Detected path details:
  * `tempo_multiplier` (½× / 2× buttons) and `tempo_feel` compound into one
    factor m = multiplier × {quarter: 1, eighth: ½, half: 2}. The real tempo
    is the tracker's × m, and every tracked interval is m beats long, so the
    grid keeps every tracked beat and reports `grid_step_beats = m`.
  * The bpm reported for a region with ≥ 2 tracked beats is measured inside
    the region (60 × m / median tracked interval): "a region is not a BPM, it
    is a list of beat positions". The file-wide estimate is only the fallback.
  * The first tracked beat after the origin gets the nearest integer step
    (a beat within half a period of the origin *is* the origin); later beats
    advance by round(Δt / period) steps so a skipped tracked beat does not
    shift the grid.
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field
from typing import Literal

from ..models import Region

log = logging.getLogger(__name__)

FEEL_FACTOR: dict[str, float] = {"quarter": 1.0, "eighth": 0.5, "half": 2.0}
FALLBACK_BPM = 120.0
MIN_REGION_S = 1e-3


@dataclass
class RegionTempo:
    bpm: float
    source: Literal["declared", "detected", "fallback"]
    beat_grid: list[float]        # absolute seconds in the file, first entry = grid origin
    length_beats: float           # region length expressed in beats at `bpm`
    confidence: float             # 1.0 for declared
    grid_step_beats: float = field(default=1.0)   # beats between consecutive grid entries

    @property
    def beat_period_s(self) -> float:
        """Seconds per beat at `bpm`."""
        return 60.0 / self.bpm

    @property
    def grid_interval_s(self) -> float:
        """Nominal seconds between consecutive grid entries."""
        return self.beat_period_s * self.grid_step_beats

    @property
    def origin_s(self) -> float:
        return self.beat_grid[0]


def snap_point(region: Region, *, transient_snap: bool = True) -> float:
    """§7.2: where playback and the grid begin in the source, before nudge."""
    ov = region.user_overrides
    onset = region.first_onset_s
    if transient_snap and not ov.keep_preroll and onset is not None:
        if region.start_s <= onset < region.end_s - MIN_REGION_S:
            return float(onset)
    return float(region.start_s)


def _tempo_factor(region: Region) -> float:
    ov = region.user_overrides
    mult = float(ov.tempo_multiplier) if ov.tempo_multiplier and ov.tempo_multiplier > 0 else 1.0
    return mult * FEEL_FACTOR.get(ov.tempo_feel, 1.0)


def _linear_grid(origin: float, end: float, interval: float) -> list[float]:
    grid = [origin]
    k = 1
    while origin + k * interval <= end + 1e-9:
        grid.append(origin + k * interval)
        k += 1
    return grid


def derive_region_tempo(
    region: Region,
    sample_bpm: float | None,
    sample_bpm_confidence: float | None,
    sample_beats: list[float],
    *,
    transient_snap: bool = True,
) -> RegionTempo:
    """Tempo and beat grid for `region` (§7.1, §7.3 tiers 2–3)."""
    origin = snap_point(region, transient_snap=transient_snap)
    end = float(region.end_s)
    span = end - origin
    if span <= MIN_REGION_S:
        raise ValueError(f"region {region.id}: empty span after snap ({origin:.4f}..{end:.4f})")

    declared = region.declared_length_beats
    if declared is not None and declared > 0:
        bpm = 60.0 * float(declared) / float(region.duration_s)
        period = 60.0 / bpm
        grid = _linear_grid(origin, end, period)
        return RegionTempo(bpm, "declared", grid, span / period, 1.0, 1.0)

    step = _tempo_factor(region)
    nominal_bpm = float(sample_bpm) * step if sample_bpm and sample_bpm > 0 else None
    confidence = float(sample_bpm_confidence) if sample_bpm_confidence is not None else 0.0

    source_beats = region.beats if region.beats else sample_beats
    inside = sorted(float(b) for b in source_beats if origin <= b <= end)

    if len(inside) >= 2:
        diffs = [b - a for a, b in zip(inside, inside[1:], strict=False) if b - a > MIN_REGION_S]
        if diffs:
            interval = float(statistics.median(diffs))
            grid = _grid_from_tracked(origin, end, inside, interval)
            bpm = 60.0 * step / interval
            if nominal_bpm is not None and abs(bpm / nominal_bpm - 1.0) > 0.25:
                log.info(
                    "region %s: tracked tempo %.1f differs from file estimate %.1f",
                    region.id, bpm, nominal_bpm,
                )
            # length in beats: last grid entry's beat plus the partial tail at the reference interval
            last_beat = (len(grid) - 1) * step
            length_beats = last_beat + (end - grid[-1]) / interval * step
            return RegionTempo(bpm, "detected", grid, length_beats, confidence or 0.5, step)

    bpm = nominal_bpm if nominal_bpm is not None else FALLBACK_BPM
    if nominal_bpm is None:
        log.info("region %s: no tempo information, assuming %.0f bpm", region.id, FALLBACK_BPM)
        confidence = 0.0
    period = 60.0 / bpm
    grid = _linear_grid(origin, end, period)
    return RegionTempo(bpm, "fallback", grid, span / period, confidence, 1.0)


def _grid_from_tracked(origin: float, end: float, beats: list[float], interval: float) -> list[float]:
    """Tier 2 grid: the origin, then every tracked beat (§7.3). A tracked beat
    within half an interval of the origin is merged into it. The returned
    positions are one per grid step; a skipped tracked beat (a gap of ~2
    intervals) is filled by linear interpolation so steps stay consistent."""
    grid = [origin]
    prev_t = origin
    prev_step = 0
    for t in beats:
        n = int(round((t - prev_t) / interval))
        if n <= 0:
            continue  # merged into the previous grid entry
        # fill interpolated steps for skipped tracked beats
        for j in range(1, n):
            grid.append(prev_t + (t - prev_t) * j / n)
        grid.append(t)
        prev_t = t
        prev_step += n
    return [g for g in grid if g <= end + 1e-9]
