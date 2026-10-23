import { describe, expect, it } from 'vitest';
import type { Pattern } from '../api/types';
import {
  beatsToSeconds,
  beatsToTicks,
  formatPosition,
  progressionEvents,
  resizeSteps,
  slotEvents,
  slotStarts,
  stepAt,
  stepCount,
  stepFires,
  totalBeats,
} from './schedule';

const hold: Pattern = { mode: 'hold', division: '1/8', steps: [] };
const retrig = (division: Pattern['division'], steps: Pattern['steps'] = []): Pattern => ({
  mode: 'retrigger',
  division,
  steps,
});

describe('tempo conversions', () => {
  it('converts beats to seconds at the session bpm', () => {
    expect(beatsToSeconds(4, 120)).toBe(2);
    expect(beatsToSeconds(1, 60)).toBe(1);
  });
  it('rounds ticks at the transport PPQ', () => {
    expect(beatsToTicks(1.5, 192)).toBe(288);
    expect(beatsToTicks(1 / 3, 192)).toBe(64);
  });
});

describe('slot layout', () => {
  it('accumulates slot starts and total length', () => {
    const slots = [{ length_beats: 4 }, { length_beats: 2 }, { length_beats: 4 }];
    expect(slotStarts(slots)).toEqual([0, 4, 6]);
    expect(totalBeats(slots)).toBe(10);
  });
  it('ignores negative lengths', () => {
    expect(totalBeats([{ length_beats: -3 }, { length_beats: 4 }])).toBe(4);
  });
});

describe('hold', () => {
  it('plays the region once for the whole slot', () => {
    const ev = slotEvents({ length_beats: 4, pattern: hold }, 8);
    expect(ev).toHaveLength(1);
    expect(ev[0]).toMatchObject({ startBeats: 8, durationBeats: 4, offsetBeats: 0, velocity: 1, stepIndex: -1 });
  });
  it('produces nothing for a zero-length slot', () => {
    expect(slotEvents({ length_beats: 0, pattern: hold }, 0)).toEqual([]);
  });
});

describe('retrigger', () => {
  it('counts whole steps at the division', () => {
    expect(stepCount(4, '1/4')).toBe(4);
    expect(stepCount(4, '1/8')).toBe(8);
    expect(stepCount(4, '1/16')).toBe(16);
    expect(stepCount(3, '1/8')).toBe(6);
    expect(stepCount(1.1, '1/4')).toBe(1);
  });
  it('restarts the region every division with gate x division durations', () => {
    const ev = slotEvents({ length_beats: 2, pattern: retrig('1/8', [{ ...stepAt(retrig('1/8'), 0), gate: 0.5 }]) }, 4);
    expect(ev.map((e) => e.startBeats)).toEqual([4, 4.5, 5, 5.5]);
    expect(ev.every((e) => e.durationBeats === 0.25)).toBe(true);
  });
  it('cycles a short step list and uses the default step for an empty one', () => {
    const steps = [
      { active: true, gate: 1, velocity: 0.5, probability: 1, offset_beats: 0 },
      { active: false, gate: 1, velocity: 1, probability: 0.2, offset_beats: 1 },
    ];
    const ev = slotEvents({ length_beats: 2, pattern: retrig('1/4', steps) }, 0);
    expect(ev.map((e) => e.velocity)).toEqual([0.5, 1]);
    expect(ev.map((e) => e.active)).toEqual([true, false]);
    expect(ev.map((e) => e.offsetBeats)).toEqual([0, 1]);
    const four = slotEvents({ length_beats: 4, pattern: retrig('1/4', steps) }, 0);
    expect(four.map((e) => e.stepIndex)).toEqual([0, 1, 0, 1]);
    const dflt = slotEvents({ length_beats: 1, pattern: retrig('1/4') }, 0);
    expect(dflt[0]).toMatchObject({ velocity: 1, probability: 1, active: true, offsetBeats: 0 });
  });
  it('clamps gate, velocity and probability to [0, 1]', () => {
    const ev = slotEvents(
      { length_beats: 1, pattern: retrig('1/4', [{ active: true, gate: 3, velocity: -1, probability: 7, offset_beats: -2 }]) },
      0,
    );
    expect(ev[0]).toMatchObject({ durationBeats: 1, velocity: 0, probability: 1, offsetBeats: 0 });
  });
  it('rolls probability per step and never fires an inactive step', () => {
    const [ev] = slotEvents({ length_beats: 1, pattern: retrig('1/4', [{ ...stepAt(hold, 0), probability: 0.5 }]) }, 0);
    expect(ev).toBeDefined();
    if (!ev) return;
    expect(stepFires(ev, 0.4)).toBe(true);
    expect(stepFires(ev, 0.6)).toBe(false);
    expect(stepFires({ ...ev, active: false }, 0)).toBe(false);
    expect(stepFires({ ...ev, probability: 1 }, 0.999)).toBe(true);
  });
});

describe('progressionEvents', () => {
  it('offsets each slot by the slots before it', () => {
    const ev = progressionEvents([
      { id: 'a', length_beats: 4, pattern: hold },
      { id: 'b', length_beats: 2, pattern: retrig('1/4') },
    ]);
    expect(ev.map((e) => [e.slotId, e.event.startBeats])).toEqual([
      ['a', 0],
      ['b', 4],
      ['b', 5],
    ]);
  });
});

describe('formatPosition', () => {
  it('prints 1-based bars.beats', () => {
    expect(formatPosition(0)).toBe('1.1');
    expect(formatPosition(3.99)).toBe('1.4');
    expect(formatPosition(4)).toBe('2.1');
    expect(formatPosition(9.5)).toBe('3.2');
    expect(formatPosition(-1)).toBe('1.1');
    expect(formatPosition(3, 3)).toBe('2.1');
  });
});

describe('resizeSteps', () => {
  it('keeps existing steps and pads with defaults', () => {
    const steps = [{ active: false, gate: 0.5, velocity: 0.5, probability: 0.5, offset_beats: 1 }];
    const out = resizeSteps(steps, 3);
    expect(out).toHaveLength(3);
    expect(out[0]).toEqual(steps[0]);
    expect(out[2]).toMatchObject({ active: true, gate: 1 });
    expect(resizeSteps(out, 1)).toEqual([steps[0]]);
  });
});
