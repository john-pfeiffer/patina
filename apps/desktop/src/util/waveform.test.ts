import { describe, expect, it } from 'vitest';
import { clampBounds, effectiveSnap, fitView, nearestMarker, panView, snapTime, timeToX, xToTime, zoomView } from './waveform';

describe('waveform view math', () => {
  const view = { start: 2, end: 6, width: 400 };

  it('maps time and pixels both ways', () => {
    expect(timeToX(view, 2)).toBe(0);
    expect(timeToX(view, 6)).toBe(400);
    expect(xToTime(view, 100)).toBe(3);
    expect(xToTime(view, timeToX(view, 4.321))).toBeCloseTo(4.321, 9);
  });

  it('zooms around a focus and stays inside the file', () => {
    const z = zoomView(view, 4, 0.5, 10);
    expect(z.end - z.start).toBeCloseTo(2);
    expect(z.start).toBeCloseTo(3);
    const out = zoomView(view, 4, 10, 10);
    expect(out).toMatchObject({ start: 0, end: 10 });
    const tiny = zoomView(view, 4, 0.0001, 10);
    expect(tiny.end - tiny.start).toBeCloseTo(0.05);
  });

  it('pans inside the file', () => {
    expect(panView(view, 1, 10)).toMatchObject({ start: 3, end: 7 });
    expect(panView(view, -5, 10)).toMatchObject({ start: 0, end: 4 });
    expect(panView(view, 50, 10)).toMatchObject({ start: 6, end: 10 });
  });

  it('fits a region with padding', () => {
    expect(fitView(1, 3, 10, 400)).toMatchObject({ start: 0.5, end: 3.5 });
    expect(fitView(0, 1, 1, 400)).toMatchObject({ start: 0, end: 1 });
  });

  it('snaps to the nearest marker within tolerance', () => {
    expect(nearestMarker([0, 0.5, 1.0], 0.52, 0.05)).toBe(0.5);
    expect(nearestMarker([0, 0.5, 1.0], 0.7, 0.05)).toBeNull();
    expect(snapTime(0.52, 'onset', [0.5], [0.6], 0.05)).toBe(0.5);
    expect(snapTime(0.58, 'beat', [0.5], [0.6], 0.05)).toBe(0.6);
    expect(snapTime(0.52, 'free', [0.5], [0.6], 0.05)).toBe(0.52);
  });

  it('clamps bounds and keeps the snap inside', () => {
    expect(clampBounds({ start: -1, end: 20, snap: 15 }, 10)).toEqual({ start: 0, end: 10, snap: 9.95 });
    const b = clampBounds({ start: 4, end: 4.01, snap: null }, 10);
    expect(b.end - b.start).toBeCloseTo(0.05);
  });

  it('derives the effective snap point like the sidecar', () => {
    expect(effectiveSnap({ start: 1, end: 3, snap: 2 }, [1.2], false)).toBe(2);
    expect(effectiveSnap({ start: 1, end: 3, snap: null }, [0.5, 1.2, 2.5], false)).toBe(1.2);
    expect(effectiveSnap({ start: 1, end: 3, snap: null }, [0.95], false)).toBe(1);
    expect(effectiveSnap({ start: 1, end: 3, snap: null }, [1.2], true)).toBe(1);
    expect(effectiveSnap({ start: 1, end: 3, snap: null }, [], false)).toBe(1);
  });
});
