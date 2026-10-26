import { describe, expect, it } from 'vitest';
import { probeBlobWorker, STALL_MS, TickMonitor } from './clock';

describe('TickMonitor', () => {
  it('reports ticking only while ticks keep arriving', () => {
    const m = new TickMonitor();
    expect(m.ticking(1000)).toBe(false);
    m.record(1000);
    expect(m.ticking(1000 + STALL_MS)).toBe(true);
    expect(m.ticking(1000 + STALL_MS + 1)).toBe(false);
  });

  it('counts ticks per second and the running total', () => {
    const m = new TickMonitor();
    for (let t = 0; t < 2000; t += 50) m.record(t);
    expect(m.total).toBe(40);
    expect(m.rate(1999)).toBe(20);
    m.reset();
    expect(m.total).toBe(0);
    expect(m.rate(1999)).toBe(0);
  });
});

describe('probeBlobWorker', () => {
  it('resolves false where workers do not exist (node)', async () => {
    expect(await probeBlobWorker(100)).toBe(false);
  });
});
