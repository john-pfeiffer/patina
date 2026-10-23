// Pure scheduling math for the playback engine (handoff §8.2–8.3). Everything
// here is in beats; the engine converts to seconds / transport ticks at the
// session bpm. No Tone.js imports so this file is unit-testable in node.

import type { Division, Pattern, Step } from '../api/types';

export const DIVISION_BEATS: Record<Division, number> = { '1/4': 1, '1/8': 0.5, '1/16': 0.25 };

export const DEFAULT_STEP: Step = { active: true, gate: 1, velocity: 1, probability: 1, offset_beats: 0 };

export const BEATS_PER_BAR = 4;

export interface SlotLike {
  length_beats: number;
  pattern: Pattern;
}

export interface StepEvent {
  /** Beat offset from the start of the progression. */
  startBeats: number;
  /** How long the buffer plays before the hard cut. */
  durationBeats: number;
  /** Beat offset into the region to start from (retrigger `offset_beats`). */
  offsetBeats: number;
  velocity: number;
  probability: number;
  active: boolean;
  /** Index into the pattern's step list, or -1 for a hold. */
  stepIndex: number;
}

export function beatsToSeconds(beats: number, bpm: number): number {
  return (beats * 60) / bpm;
}

export function beatsToTicks(beats: number, ppq: number): number {
  return Math.round(beats * ppq);
}

/** Cumulative start beat of every slot, in order. */
export function slotStarts(slots: readonly { length_beats: number }[]): number[] {
  const out: number[] = [];
  let acc = 0;
  for (const s of slots) {
    out.push(acc);
    acc += Math.max(0, s.length_beats);
  }
  return out;
}

export function totalBeats(slots: readonly { length_beats: number }[]): number {
  return slots.reduce((acc, s) => acc + Math.max(0, s.length_beats), 0);
}

/** Number of retrigger steps a slot holds at its division (partial last step dropped). */
export function stepCount(lengthBeats: number, division: Division): number {
  const div = DIVISION_BEATS[division];
  return Math.max(0, Math.floor(lengthBeats / div + 1e-9));
}

/** Step `i` of a pattern; patterns shorter than the slot cycle, empty patterns mean the default step. */
export function stepAt(pattern: Pattern, i: number): Step {
  if (pattern.steps.length === 0) return DEFAULT_STEP;
  const step = pattern.steps[i % pattern.steps.length];
  return step ?? DEFAULT_STEP;
}

const clamp01 = (x: number) => Math.min(1, Math.max(0, x));

/** Events for one slot whose first beat is `startBeats` into the progression. */
export function slotEvents(slot: SlotLike, startBeats: number): StepEvent[] {
  const length = Math.max(0, slot.length_beats);
  if (length === 0) return [];
  if (slot.pattern.mode === 'hold') {
    return [
      {
        startBeats,
        durationBeats: length,
        offsetBeats: 0,
        velocity: 1,
        probability: 1,
        active: true,
        stepIndex: -1,
      },
    ];
  }
  const div = DIVISION_BEATS[slot.pattern.division];
  const n = stepCount(length, slot.pattern.division);
  const events: StepEvent[] = [];
  for (let i = 0; i < n; i++) {
    const step = stepAt(slot.pattern, i);
    events.push({
      startBeats: startBeats + i * div,
      durationBeats: clamp01(step.gate) * div,
      offsetBeats: Math.max(0, step.offset_beats),
      velocity: clamp01(step.velocity),
      probability: clamp01(step.probability),
      active: step.active,
      stepIndex: i % Math.max(1, slot.pattern.steps.length),
    });
  }
  return events;
}

/** Events for a whole progression, tagged with the slot they belong to. */
export function progressionEvents<T extends SlotLike & { id: string }>(
  slots: readonly T[],
): { slotId: string; event: StepEvent }[] {
  const starts = slotStarts(slots);
  const out: { slotId: string; event: StepEvent }[] = [];
  slots.forEach((slot, i) => {
    for (const event of slotEvents(slot, starts[i] ?? 0)) out.push({ slotId: slot.id, event });
  });
  return out;
}

/** The step a probability roll keeps: `roll` is a uniform [0,1) sample. */
export function stepFires(event: StepEvent, roll: number): boolean {
  if (!event.active) return false;
  return roll <= event.probability;
}

/** `bars.beats` for a transport position in beats (1-based, like a DAW). */
export function formatPosition(beats: number, beatsPerBar = BEATS_PER_BAR): string {
  const b = Math.max(0, beats);
  const bar = Math.floor(b / beatsPerBar) + 1;
  const beat = Math.floor(b % beatsPerBar) + 1;
  return `${bar}.${beat}`;
}

/** Retrigger patterns are stored as step lists; resize keeps existing steps. */
export function resizeSteps(steps: readonly Step[], n: number): Step[] {
  const out = steps.slice(0, n);
  while (out.length < n) out.push({ ...DEFAULT_STEP });
  return out;
}
