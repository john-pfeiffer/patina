// Renders the real PlaybackEngine through Tone.Offline in headless Chromium
// and listens to the output: each slot window must carry its own region's
// tone, a short region must restart to fill a hold slot, and the loop must
// wrap. This is the scheduling and graph path that the unit tests cannot see.

import * as Tone from 'tone';
import { describe, expect, it, vi } from 'vitest';

const SR = 44100;

/** Mono float32 WAV of a sine. */
function sineWav(hz: number, seconds: number): ArrayBuffer {
  const n = Math.round(seconds * SR);
  const buf = new ArrayBuffer(44 + n * 4);
  const v = new DataView(buf);
  const str = (o: number, s: string) => [...s].forEach((c, i) => v.setUint8(o + i, c.charCodeAt(0)));
  str(0, 'RIFF');
  v.setUint32(4, 36 + n * 4, true);
  str(8, 'WAVE');
  str(12, 'fmt ');
  v.setUint32(16, 16, true);
  v.setUint16(20, 3, true); // IEEE float
  v.setUint16(22, 1, true);
  v.setUint32(24, SR, true);
  v.setUint32(28, SR * 4, true);
  v.setUint16(32, 4, true);
  v.setUint16(34, 32, true);
  str(36, 'data');
  v.setUint32(40, n * 4, true);
  for (let i = 0; i < n; i++) v.setFloat32(44 + i * 4, 0.5 * Math.sin((2 * Math.PI * hz * i) / SR), true);
  return buf;
}

const FILES: Record<string, ArrayBuffer> = {
  '/renders/a.wav': sineWav(220, 0.6), // shorter than its 1 s slot: hold must restart it
  '/renders/b.wav': sineWav(330, 1.5), // longer than its slot: cut at the slot end
};

vi.mock('../api/client', () => ({
  fetchAudio: async (path: string) => {
    const f = FILES[path];
    if (!f) throw new Error(`no fixture for ${path}`);
    return f.slice(0);
  },
}));

const { PlaybackEngine, engine } = await import('./engine');
type Program = import('./engine').EngineProgram;

const HOLD = { mode: 'hold' as const, division: '1/4' as const, steps: [] };

function program(density: boolean): Program {
  return {
    bpm: 120,
    loop: true,
    slots: [
      { id: 'A', regionId: 'ra', length_beats: 2, pattern: HOLD, wetness: null, renderPath: '/renders/a.wav' },
      { id: 'B', regionId: 'rb', length_beats: 2, pattern: HOLD, wetness: null, renderPath: '/renders/b.wav' },
    ],
    density: { enabled: density, referenceLufs: -18 },
    space: { enabled: false, amount: 100, anchorWetness: null },
  };
}

function rms(x: Float32Array, from: number, to: number): number {
  const a = Math.round(from * SR);
  const b = Math.round(to * SR);
  let acc = 0;
  for (let i = a; i < b; i++) acc += x[i]! * x[i]!;
  return Math.sqrt(acc / (b - a));
}

/** Zero crossings per second ≈ 2 × frequency for a sine. */
function crossingsPerSecond(x: Float32Array, from: number, to: number): number {
  const a = Math.round(from * SR);
  const b = Math.round(to * SR);
  let n = 0;
  for (let i = a + 1; i < b; i++) if (x[i - 1]! < 0 !== x[i]! < 0) n++;
  return n / (to - from);
}

async function render(density: boolean): Promise<Float32Array> {
  const out = await Tone.Offline(
    async ({ transport }) => {
      const engine = new PlaybackEngine();
      engine.onError = (m) => {
        throw new Error(m);
      };
      engine.sync(program(density));
      for (let i = 0; i < 200 && engine.snapshot().loadedSlots < 2; i++) await new Promise((r) => setTimeout(r, 25));
      expect(engine.snapshot().loadedSlots).toBe(2);
      transport.start(0);
    },
    2.5,
    1,
    SR,
  );
  return out.toArray(0) as Float32Array;
}

describe('PlaybackEngine (offline render)', () => {
  it('plays each slot from the transport, restarts a short hold and wraps the loop', async () => {
    const x = await render(true);
    // Slot A, first pass (region is 0.6 s; the hold restarts it at 0.6).
    expect(rms(x, 0.05, 0.55)).toBeGreaterThan(0.1);
    expect(crossingsPerSecond(x, 0.05, 0.55)).toBeCloseTo(440, -1);
    expect(rms(x, 0.65, 0.95)).toBeGreaterThan(0.1);
    // Slot B.
    expect(rms(x, 1.05, 1.95)).toBeGreaterThan(0.1);
    expect(crossingsPerSecond(x, 1.1, 1.9)).toBeCloseTo(660, -1);
    // Loop wrapped at 2 s: slot A again.
    expect(rms(x, 2.05, 2.45)).toBeGreaterThan(0.1);
    expect(crossingsPerSecond(x, 2.05, 2.45)).toBeCloseTo(440, -1);
  }, 30_000);

  it('still reaches the output with the density compressor bypassed', async () => {
    const x = await render(false);
    expect(rms(x, 0.05, 0.55)).toBeGreaterThan(0.1);
    expect(rms(x, 1.05, 1.95)).toBeGreaterThan(0.1);
  }, 30_000);
});

describe('clock diagnostics (live context)', () => {
  it('fires all four clicks on the context clock', async () => {
    expect(await engine.testClockClicks()).toBe(4);
    const snap = engine.snapshot();
    expect(snap.clockSource).toBe('worker');
    expect(snap.clockNote).toBeNull();
  }, 10_000);
});
