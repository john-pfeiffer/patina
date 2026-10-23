"""§7.3–7.4 Warp map: a region is a list of beat positions; each inter-beat
interval gets its own stretch ratio. Tiers: manual markers > auto beats > linear
grid. Warp strength blends each interval's ratio toward the region's average
ratio (100 % = every beat on the grid, 0 % = average ratio only).

The plan is a piecewise-linear map from source seconds to session beats built
from *mapping points* (source_s, beat):

  * the tempo's grid (tier 2 or 3) contributes (grid[k], k × grid_step_beats);
  * manual markers (tier 1) are added verbatim and always win: a grid point
    that sits on a marker's beat, or that would make the map non-monotone
    next to a marker, is dropped. Dragging a marker therefore only changes the
    two intervals around it, as in any DAW.

Playback starts at the snap point plus nudge (§7.2); the source from there to
the region end is cut at every mapping point into `WarpSegment`s whose
destination length is their beat span at the session tempo. Warp strength
then blends the ratios inside every span between manual markers toward that
span's average, which keeps each span's (and the plan's) total destination
length exact. Beyond the grid's last entry the map extrapolates at the
region's nominal beat period, so the plan's total destination length equals
`tempo.length_beats` beats at the session tempo (nudge aside)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace

from ..models import Region, UserOverrides
from .grid import RegionTempo

log = logging.getLogger(__name__)

MIN_SEGMENT_S = 1e-3          # mapping points closer than this merge
MARKER_BEAT_TOL = 0.05        # grid point within this many beats of a marker yields to it
UNIFORM_TOL = 1e-4            # relative ratio spread below which a plan counts as uniform


@dataclass
class WarpSegment:
    src_start_s: float      # absolute seconds in the source file
    src_end_s: float
    dst_len_s: float        # length after stretching, at session tempo

    @property
    def ratio(self) -> float:
        """dst / src: > 1 slows down (stretches), < 1 speeds up."""
        return self.dst_len_s / max(1e-9, self.src_end_s - self.src_start_s)

    @property
    def src_len_s(self) -> float:
        return self.src_end_s - self.src_start_s


@dataclass
class WarpPlan:
    segments: list[WarpSegment]
    play_start_s: float     # where playback begins in the source (after snap + nudge)
    average_ratio: float
    uniform: bool           # True when every ratio is (nearly) the same

    @property
    def src_len_s(self) -> float:
        return sum(s.src_len_s for s in self.segments)

    @property
    def dst_len_s(self) -> float:
        return sum(s.dst_len_s for s in self.segments)

    @property
    def src_end_s(self) -> float:
        return self.segments[-1].src_end_s if self.segments else self.play_start_s


@dataclass(frozen=True)
class _Point:
    src: float
    beat: float
    manual: bool


def build_warp_plan(
    region: Region,
    tempo: RegionTempo,
    session_bpm: float,
    warp_strength: float,
    overrides: UserOverrides,
    *,
    transient_snap: bool = True,
) -> WarpPlan:
    """Warp plan for `region` at `session_bpm` (§7.2–7.4). `transient_snap`
    is accepted for symmetry with `derive_region_tempo`; the grid origin
    already encodes it, and `overrides.nudge_ms` moves the play start."""
    if session_bpm <= 0:
        raise ValueError("session_bpm must be positive")
    beat_dst = 60.0 / float(session_bpm)
    end = float(region.end_s)

    # §7.2 play start: snap point (grid origin) + nudge, kept inside the region.
    play_start = tempo.origin_s + float(overrides.nudge_ms) / 1000.0
    play_start = min(max(play_start, float(region.start_s)), end - MIN_SEGMENT_S)

    points = _mapping_points(region, tempo, overrides)
    slope = 1.0 / tempo.beat_period_s  # beats per second for extrapolation

    def beat_at(t: float) -> float:
        return _interp_beat(points, t, slope)

    # Cut boundaries: play start, every mapping point strictly inside, region end.
    cuts = [play_start]
    manual_cuts: set[int] = set()
    for p in points:
        if p.src <= cuts[-1] + MIN_SEGMENT_S or p.src >= end - MIN_SEGMENT_S:
            continue
        cuts.append(p.src)
        if p.manual:
            manual_cuts.add(len(cuts) - 1)
    cuts.append(end)

    segments = [
        WarpSegment(a, b, (beat_at(b) - beat_at(a)) * beat_dst)
        for a, b in zip(cuts, cuts[1:], strict=False)
    ]
    for seg in segments:
        if seg.dst_len_s <= 0:
            raise ValueError(f"region {region.id}: non-monotone warp map at {seg.src_start_s:.4f}s")

    strength = min(max(float(warp_strength), 0.0), 100.0) / 100.0
    if strength < 1.0 - 1e-9:
        segments = _apply_strength(segments, strength, manual_cuts)

    total_src = sum(s.src_len_s for s in segments)
    total_dst = sum(s.dst_len_s for s in segments)
    average = total_dst / max(1e-9, total_src)
    uniform = all(abs(s.ratio / average - 1.0) <= UNIFORM_TOL for s in segments)
    return WarpPlan(segments, play_start, average, uniform)


def _mapping_points(region: Region, tempo: RegionTempo, overrides: UserOverrides) -> list[_Point]:
    manual = [
        _Point(float(m.source_s), float(m.beat), True)
        for m in overrides.warp_markers
        if region.start_s - MIN_SEGMENT_S <= m.source_s <= region.end_s + MIN_SEGMENT_S
    ]
    if len(manual) != len(overrides.warp_markers):
        log.warning("region %s: ignoring warp markers outside the region", region.id)
    manual.sort(key=lambda p: p.beat)

    grid = [
        _Point(float(t), k * tempo.grid_step_beats, False)
        for k, t in enumerate(tempo.beat_grid)
    ]
    if not manual:
        return grid

    kept: list[_Point] = list(manual)
    for g in grid:
        if any(abs(g.beat - m.beat) < MARKER_BEAT_TOL or abs(g.src - m.src) < MIN_SEGMENT_S for m in manual):
            continue
        kept.append(g)
    kept.sort(key=lambda p: (p.beat, not p.manual))

    # Enforce a strictly increasing source axis; grid points yield to markers.
    out: list[_Point] = []
    for i, p in enumerate(kept):
        if out and p.src <= out[-1].src + MIN_SEGMENT_S:
            if p.manual and not out[-1].manual:
                out.pop()
                while out and p.src <= out[-1].src + MIN_SEGMENT_S and not out[-1].manual:
                    out.pop()
                if out and p.src <= out[-1].src + MIN_SEGMENT_S:
                    log.warning("region %s: dropping inconsistent warp marker at %.3fs", region.id, p.src)
                    continue
            else:
                if p.manual:
                    log.warning("region %s: dropping inconsistent warp marker at %.3fs", region.id, p.src)
                continue
        if not p.manual:
            nxt = next((q for q in kept[i + 1:] if q.manual), None)
            if nxt is not None and nxt.src <= p.src + MIN_SEGMENT_S:
                continue
        out.append(p)
    return out


def _interp_beat(points: list[_Point], t: float, slope: float) -> float:
    """Piecewise-linear source→beat map through `points`, linear extrapolation
    at the nominal tempo beyond the ends."""
    if not points:
        return t * slope
    if t <= points[0].src:
        return points[0].beat - (points[0].src - t) * slope
    if t >= points[-1].src:
        return points[-1].beat + (t - points[-1].src) * slope
    lo, hi = 0, len(points) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if points[mid].src <= t:
            lo = mid
        else:
            hi = mid
    a, b = points[lo], points[hi]
    f = (t - a.src) / max(1e-12, b.src - a.src)
    return a.beat + f * (b.beat - a.beat)


def _apply_strength(segments: list[WarpSegment], strength: float, manual_cuts: set[int]) -> list[WarpSegment]:
    """§7.4: inside every span between manual markers, blend each ratio toward
    the span's average; total length per span is preserved exactly."""
    out: list[WarpSegment] = []
    span: list[WarpSegment] = []

    def flush() -> None:
        if not span:
            return
        src = sum(s.src_len_s for s in span)
        dst = sum(s.dst_len_s for s in span)
        avg = dst / max(1e-9, src)
        for s in span:
            r = avg + strength * (s.ratio - avg)
            out.append(WarpSegment(s.src_start_s, s.src_end_s, s.src_len_s * r))
        span.clear()

    for i, seg in enumerate(segments):
        if i in manual_cuts:
            flush()
        span.append(seg)
    flush()
    return out


# --- plan transforms used by the renderer ---------------------------------------

def rebase_plan(plan: WarpPlan, origin_s: float) -> WarpPlan:
    """Express source times relative to `origin_s` (e.g. the start of a slice)."""
    return replace(
        plan,
        segments=[WarpSegment(s.src_start_s - origin_s, s.src_end_s - origin_s, s.dst_len_s) for s in plan.segments],
        play_start_s=plan.play_start_s - origin_s,
    )


def scale_plan_source(plan: WarpPlan, factor: float) -> WarpPlan:
    """Scale source times by `factor` after a resample changed the source's
    time axis (tuning, plain repitch transposition). Destination lengths are
    untouched, so the stretch stage absorbs the change (§6 Tuning)."""
    segs = [
        WarpSegment(s.src_start_s * factor, s.src_end_s * factor, s.dst_len_s) for s in plan.segments
    ]
    total_src = sum(s.src_len_s for s in segs)
    total_dst = sum(s.dst_len_s for s in segs)
    avg = total_dst / max(1e-9, total_src)
    uniform = all(abs(s.ratio / avg - 1.0) <= UNIFORM_TOL for s in segs)
    return WarpPlan(segs, plan.play_start_s * factor, avg, uniform)


def beat_times_s(plan: WarpPlan, tempo: RegionTempo) -> list[tuple[float, float]]:
    """(beat, destination time) for every grid entry inside the plan — the kit's
    beat markers re-expressed at the session tempo (§8.4)."""
    out: list[tuple[float, float]] = []
    t_dst = 0.0
    for seg in plan.segments:
        for k, g in enumerate(tempo.beat_grid):
            if seg.src_start_s - 1e-9 <= g < seg.src_end_s - 1e-9:
                f = (g - seg.src_start_s) / max(1e-12, seg.src_len_s)
                out.append((k * tempo.grid_step_beats, t_dst + f * seg.dst_len_s))
        t_dst += seg.dst_len_s
    return out
