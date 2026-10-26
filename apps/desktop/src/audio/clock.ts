// Clock health for the playback engine. Tone.js drives its Transport from a
// Web Worker built from a blob: URL; a webview that refuses that worker (a
// Content Security Policy without `worker-src blob:`, or a custom-scheme
// origin) leaves the Transport position advancing while nothing scheduled on
// it ever fires. Tone only catches a synchronous constructor failure, so the
// engine probes the worker itself and falls back to Tone's timeout clock.

export type ClockSource = 'worker' | 'timeout' | 'offline';

export interface ClockHealth {
  source: ClockSource;
  /** A tick arrived within `STALL_MS` of now (only meaningful while running). */
  ticking: boolean;
  ticksPerSecond: number;
  /** Why the engine left the worker clock, or null when it never had to. */
  fallbackReason: string | null;
}

/** No tick for this long while the transport runs means the clock is dead. */
export const STALL_MS = 400;

/** Resolves true when a blob: worker can be created and answers a message. */
export function probeBlobWorker(timeoutMs = 500): Promise<boolean> {
  return new Promise((resolve) => {
    let worker: Worker | null = null;
    let url: string | null = null;
    let settled = false;
    const done = (ok: boolean) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      try {
        worker?.terminate();
      } catch {
        // already gone
      }
      if (url) URL.revokeObjectURL(url);
      resolve(ok);
    };
    const timer = setTimeout(() => done(false), timeoutMs);
    try {
      if (typeof Worker === 'undefined' || typeof Blob === 'undefined') {
        done(false);
        return;
      }
      const blob = new Blob(['self.postMessage("ok");'], { type: 'text/javascript' });
      url = URL.createObjectURL(blob);
      worker = new Worker(url);
      worker.onmessage = () => done(true);
      worker.onerror = () => done(false);
    } catch {
      done(false);
    }
  });
}

/** Counts ticks and judges whether the clock is alive. Pure bookkeeping so it
 * is unit-testable; the engine feeds it `now()` from `performance.now()`. */
export class TickMonitor {
  private stamps: number[] = [];
  /** Ticks seen since the last reset. */
  total = 0;

  record(now: number): void {
    this.total++;
    this.stamps.push(now);
    // keep one second of history
    while (this.stamps.length > 0 && now - (this.stamps[0] ?? now) > 1000) this.stamps.shift();
  }

  /** Ticks seen in the last second. */
  rate(now: number): number {
    return this.stamps.filter((t) => now - t <= 1000).length;
  }

  ticking(now: number): boolean {
    const last = this.stamps[this.stamps.length - 1];
    return last !== undefined && now - last <= STALL_MS;
  }

  reset(): void {
    this.stamps = [];
    this.total = 0;
  }
}
