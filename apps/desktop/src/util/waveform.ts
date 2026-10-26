// Pure math for the region editor's waveform view: time ↔ pixel mapping,
// snapping to markers, and bound/snap-point validation. No DOM, so it is
// unit-testable in node.

export interface View {
  /** Window start and end, seconds. */
  start: number;
  end: number;
  /** Canvas width, CSS pixels. */
  width: number;
}

export function timeToX(view: View, t: number): number {
  const span = Math.max(1e-6, view.end - view.start);
  return ((t - view.start) / span) * view.width;
}

export function xToTime(view: View, x: number): number {
  const span = Math.max(1e-6, view.end - view.start);
  return view.start + (x / view.width) * span;
}

/** Zoom the view around `focus` (seconds) by `factor` (< 1 zooms in), kept inside [0, duration]
 * and never narrower than `minSpan`. */
export function zoomView(view: View, focus: number, factor: number, duration: number, minSpan = 0.05): View {
  const span = Math.min(duration, Math.max(minSpan, (view.end - view.start) * factor));
  const frac = span <= 0 ? 0.5 : Math.min(1, Math.max(0, (focus - view.start) / Math.max(1e-6, view.end - view.start)));
  let start = focus - frac * span;
  let end = start + span;
  if (start < 0) {
    start = 0;
    end = span;
  }
  if (end > duration) {
    end = duration;
    start = Math.max(0, duration - span);
  }
  return { ...view, start, end };
}

/** Pan the view by `dt` seconds, kept inside [0, duration]. */
export function panView(view: View, dt: number, duration: number): View {
  const span = view.end - view.start;
  let start = view.start + dt;
  start = Math.max(0, Math.min(Math.max(0, duration - span), start));
  return { ...view, start, end: start + span };
}

/** A view that shows [start, end] with `pad` of its length on either side. */
export function fitView(start: number, end: number, duration: number, width: number, pad = 0.25): View {
  const len = Math.max(0.05, end - start);
  const s = Math.max(0, start - len * pad);
  const e = Math.min(duration, end + len * pad);
  return { start: s, end: Math.max(e, s + 0.05), width };
}

/** The marker nearest to `t` within `tolerance` seconds, or null. */
export function nearestMarker(markers: readonly number[], t: number, tolerance: number): number | null {
  let best: number | null = null;
  let bestD = tolerance;
  for (const m of markers) {
    const d = Math.abs(m - t);
    if (d <= bestD) {
      best = m;
      bestD = d;
    }
  }
  return best;
}

export type SnapMode = 'onset' | 'beat' | 'free';

/** Snap `t` according to the mode; the tolerance is in seconds (derived from pixels by the caller). */
export function snapTime(t: number, mode: SnapMode, onsets: readonly number[], beats: readonly number[], tolerance: number): number {
  if (mode === 'free') return t;
  const near = nearestMarker(mode === 'onset' ? onsets : beats, t, tolerance);
  return near ?? t;
}

export interface Bounds {
  start: number;
  end: number;
  /** Absolute seconds or null for the detected onset. */
  snap: number | null;
}

export const MIN_REGION_S = 0.05;

/** Clamp a dragged bound into the file and keep the region at least MIN_REGION_S long;
 * the snap point follows so it stays inside. */
export function clampBounds(b: Bounds, duration: number): Bounds {
  let start = Math.max(0, Math.min(b.start, duration - MIN_REGION_S));
  let end = Math.max(start + MIN_REGION_S, Math.min(b.end, duration));
  if (end - start < MIN_REGION_S) start = Math.max(0, end - MIN_REGION_S);
  let snap = b.snap;
  if (snap !== null) snap = Math.max(start, Math.min(snap, end - MIN_REGION_S));
  return { start, end, snap };
}

/** Where playback starts for a draft: the explicit snap, else the first onset inside, else the start. */
export function effectiveSnap(b: Bounds, onsets: readonly number[], keepPreroll: boolean): number {
  if (b.snap !== null) return b.snap;
  if (keepPreroll) return b.start;
  const inside = onsets.find((o) => o >= b.start - 0.12 && o < b.end - MIN_REGION_S);
  return inside === undefined ? b.start : Math.max(b.start, inside);
}
