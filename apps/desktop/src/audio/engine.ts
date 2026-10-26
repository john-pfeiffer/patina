// Tone.js playback engine (handoff §2 "offline conform, online play", §8.3).
// Tone.Transport is the clock. One Player per slot holds the last good render;
// a newer key that is still rendering never interrupts playback — the store
// marks the slot stale and the engine keeps the buffer it has. Cuts are hard:
// 1.5 ms declick ramps, no crossfades. The only real-time DSP is gain, the
// Density bus compressor and the Space reverb send.

import * as Tone from 'tone';
import { fetchAudio } from '../api/client';
import type { Pattern } from '../api/types';
import { probeBlobWorker, STALL_MS, TickMonitor, type ClockSource } from './clock';
import { beatsToSeconds, beatsToTicks, progressionEvents, stepFires, totalBeats, type StepEvent } from './schedule';

export const DECLICK_S = 0.0015;
const REVERB_DECAY_S = 2.5;
const DENSITY_RATIO = 2;
const DENSITY_ATTACK_S = 0.03;
const DENSITY_RELEASE_S = 0.25;
/** With a 2:1 ratio, 5 dB over threshold is ~2.5 dB of gain reduction. */
const DENSITY_HEADROOM_DB = 5;
const DEFAULT_REFERENCE_LUFS = -18;

export interface EngineSlot {
  id: string;
  regionId: string | null;
  length_beats: number;
  pattern: Pattern;
  /** Region wetness (§4.8) for the Space send; null when unmeasured. */
  wetness: number | null;
  /** Sidecar path of the ready render, or null while nothing new is ready. */
  renderPath: string | null;
}

export interface DensitySetting {
  enabled: boolean;
  referenceLufs: number | null;
}

export interface SpaceSetting {
  enabled: boolean;
  /** Percent, 0–100. */
  amount: number;
  anchorWetness: number | null;
}

export interface EngineProgram {
  bpm: number;
  loop: boolean;
  slots: EngineSlot[];
  density: DensitySetting;
  space: SpaceSetting;
}

export interface EngineTransport {
  playing: boolean;
  positionBeats: number;
  /** Web Audio context state: 'running' is the only one that makes sound. */
  contextState: string;
  sampleRate: number;
  /** Slots whose render buffer is decoded and ready to fire / slots in the program. */
  loadedSlots: number;
  totalSlots: number;
  /** Tone's clock: 'worker' (default), 'timeout' (fallback) or 'offline' (tests). */
  clockSource: ClockSource;
  /** Clock ticks are arriving (only meaningful while playing). */
  ticking: boolean;
  ticksPerSecond: number;
  /** Set when the engine had to leave the worker clock, or the clock is dead. */
  clockNote: string | null;
  baseLatency: number | null;
  outputLatency: number | null;
  lookAhead: number;
  /** Output device id when the webview reports one; '' is the system default. */
  sinkId: string | null;
}

export interface OutputDevice {
  id: string;
  label: string;
}

const LOG_LINES = 60;

/** Output devices the webview reports. Empty on webviews that do not expose
 * `audiooutput` devices (WKWebView on macOS routes to the system output). */
export async function listOutputDevices(): Promise<OutputDevice[]> {
  if (typeof navigator === 'undefined' || !navigator.mediaDevices?.enumerateDevices) return [];
  try {
    const all = await navigator.mediaDevices.enumerateDevices();
    return all.filter((d) => d.kind === 'audiooutput').map((d, i) => ({ id: d.deviceId, label: d.label || `output ${i + 1}` }));
  } catch {
    return [];
  }
}

function rawContext(): { setSinkId?: (id: string) => Promise<void>; sinkId?: unknown; baseLatency?: number; outputLatency?: number } {
  return Tone.getContext().rawContext as unknown as ReturnType<typeof rawContext>;
}

/** `getContext()` is typed as the abstract base, which omits the clock source
 * every concrete context has. */
function clockOf(ctx: ReturnType<typeof Tone.getContext> = Tone.getContext()): { clockSource: ClockSource } {
  return ctx as unknown as { clockSource: ClockSource };
}

/** `AudioContext.setSinkId` exists on Chromium-based webviews only. */
export function canSelectOutput(): boolean {
  return typeof rawContext().setSinkId === 'function';
}

export async function selectOutput(deviceId: string): Promise<void> {
  const raw = rawContext();
  if (typeof raw.setSinkId !== 'function') throw new Error('this webview cannot change the output device; use the system sound settings');
  await raw.setSinkId(deviceId === 'default' ? '' : deviceId);
}

interface SlotNode {
  player: Tone.Player;
  gain: Tone.Gain;
  send: Tone.Gain;
  regionId: string | null;
  path: string | null;
  buffer: Tone.ToneAudioBuffer | null;
  override: Tone.ToneAudioBuffer | null;
  loadSeq: number;
}

interface Graph {
  busIn: Tone.Gain;
  comp: Tone.Compressor;
  reverb: Tone.Reverb;
}

const clamp = (x: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, x));

/** Send level for a region drier than the anchor (§6 Space): proportional to the
 * wetness gap, scaled by the module amount, never for wetter-than-anchor regions. */
export function spaceSendLevel(space: SpaceSetting, wetness: number | null): number {
  if (!space.enabled || space.anchorWetness === null || wetness === null) return 0;
  const gap = space.anchorWetness - wetness;
  if (gap <= 0) return 0;
  return clamp(gap * (space.amount / 100), 0, 1);
}

/** Compressor threshold that lands ~2–3 dB of GR on material matched to the anchor. */
export function densityThresholdDb(referenceLufs: number | null): number {
  return clamp((referenceLufs ?? DEFAULT_REFERENCE_LUFS) - DENSITY_HEADROOM_DB, -50, -6);
}

export class PlaybackEngine {
  private graph: Graph | null = null;
  private densityRouted: boolean | null = null;
  private readonly slots = new Map<string, SlotNode>();
  private readonly buffers = new Map<string, Promise<Tone.ToneAudioBuffer>>();
  private audition: Tone.Player | null = null;
  private readonly listeners = new Set<(t: EngineTransport) => void>();
  private ticker: ReturnType<typeof setInterval> | null = null;
  private readonly ticks = new TickMonitor();
  private tickContext: ReturnType<typeof Tone.getContext> | null = null;
  private clockNote: string | null = null;
  private probed = false;
  private watchdog: ReturnType<typeof setTimeout> | null = null;
  private readonly log: string[] = [];
  onError: (message: string) => void = () => undefined;

  // --- diagnostics ------------------------------------------------------------------

  /** Console log plus a short ring buffer the Preferences page can show. */
  private note(level: 'info' | 'warn' | 'error', message: string, data?: Record<string, unknown>): void {
    const line = `${new Date().toISOString().slice(11, 23)} ${message}${data ? ' ' + JSON.stringify(data) : ''}`;
    this.log.push(line);
    if (this.log.length > LOG_LINES) this.log.shift();
    console[level](`[engine] ${message}`, data ?? '');
  }

  getLog(): readonly string[] {
    return this.log;
  }

  private readonly onTick = (): void => {
    this.ticks.record(performance.now());
  };

  /** Listen to the current context's clock and probe the worker once. Tone
   * only falls back to its timeout clock when the Worker constructor throws;
   * a webview that refuses the blob: worker asynchronously leaves a transport
   * that advances but never fires (the silent-playback bug). */
  private attachClock(): void {
    const ctx = Tone.getContext();
    if (this.tickContext !== ctx) {
      this.tickContext = ctx;
      this.ticks.reset();
      ctx.on('tick', this.onTick);
    }
    if (this.probed || clockOf(ctx).clockSource === 'offline') return;
    this.probed = true;
    if (clockOf(ctx).clockSource === 'timeout') {
      this.clockNote = 'worker clock unavailable in this webview; using the timeout clock';
      this.note('warn', 'clock', { source: 'timeout', reason: this.clockNote });
      return;
    }
    void probeBlobWorker().then((ok) => {
      if (ok || clockOf().clockSource !== 'worker') return;
      this.fallback('the webview refused the blob: clock worker (content security policy or unsupported origin); using the timeout clock');
    });
  }

  private fallback(reason: string): void {
    const ctx = Tone.getContext();
    if (clockOf(ctx).clockSource === 'worker') clockOf(ctx).clockSource = 'timeout';
    this.clockNote = reason;
    this.ticks.reset();
    this.note('warn', 'clock fallback', { source: clockOf(ctx).clockSource, reason });
    this.emit();
  }

  /** After Play, make sure ticks actually arrive; otherwise fall back, and if
   * even the timeout clock is dead say so instead of playing silence. */
  private armWatchdog(stage = 0): void {
    this.disarmWatchdog();
    const seen = this.ticks.total;
    this.watchdog = setTimeout(() => {
      this.watchdog = null;
      if (!this.playing) return;
      if (this.ticks.total > seen) return;
      const ctx = Tone.getContext();
      if (clockOf(ctx).clockSource === 'worker') {
        this.fallback('the clock worker never ticked after Play (blocked by the webview); switched to the timeout clock');
        this.armWatchdog(stage + 1);
      } else if (stage < 2) {
        this.armWatchdog(stage + 1);
      } else {
        this.clockNote = 'the audio clock is not ticking: nothing scheduled on the transport can play';
        this.note('error', 'clock dead', { source: clockOf(ctx).clockSource });
        this.onError(this.clockNote);
        this.emit();
      }
    }, STALL_MS);
  }

  private disarmWatchdog(): void {
    if (this.watchdog) clearTimeout(this.watchdog);
    this.watchdog = null;
  }

  /** A 0.4 s tone straight to the output, bypassing the engine graph. */
  async testToneDirect(): Promise<void> {
    await Tone.start();
    this.blip(new Tone.Oscillator(440, 'sine').toDestination(), 0.4);
    this.note('info', 'test tone direct', { context: Tone.getContext().state });
  }

  /** A 0.4 s tone through the slot bus (density compressor, destination). */
  async testToneBus(): Promise<void> {
    await Tone.start();
    const g = this.ensureGraph();
    this.blip(new Tone.Oscillator(660, 'sine').connect(g.busIn), 0.4);
    this.note('info', 'test tone bus', { context: Tone.getContext().state, density: this.densityRouted });
  }

  /** Four clicks timed by the context clock, the same ticker the Transport
   * uses: silence here means scheduled playback cannot work. Resolves with
   * how many fired. */
  async testClockClicks(): Promise<number> {
    await Tone.start();
    this.attachClock();
    const ctx = Tone.getContext();
    let fired = 0;
    for (let i = 0; i < 4; i++) {
      ctx.setTimeout(() => {
        fired++;
        this.blip(new Tone.Oscillator(i === 0 ? 1320 : 880, 'square').toDestination(), 0.05);
      }, i * 0.25);
    }
    await new Promise((r) => setTimeout(r, 1200));
    this.note('info', 'clock clicks', { fired, source: clockOf(ctx).clockSource, ticksPerSecond: this.ticks.rate(performance.now()) });
    return fired;
  }

  private blip(osc: Tone.Oscillator, seconds: number): void {
    osc.volume.value = -14;
    const now = Tone.now();
    osc.start(now).stop(now + seconds);
    osc.onstop = () => setTimeout(() => osc.dispose(), 0);
  }

  private get transport(): ReturnType<typeof Tone.getTransport> {
    return Tone.getTransport();
  }

  // --- graph --------------------------------------------------------------------

  private ensureGraph(): Graph {
    this.attachClock();
    if (this.graph) return this.graph;
    const busIn = new Tone.Gain(1);
    const comp = new Tone.Compressor({
      threshold: densityThresholdDb(null),
      ratio: DENSITY_RATIO,
      attack: DENSITY_ATTACK_S,
      release: DENSITY_RELEASE_S,
      knee: 6,
    });
    const reverb = new Tone.Reverb({ decay: REVERB_DECAY_S, preDelay: 0.01, wet: 1 });
    reverb.connect(busIn);
    this.graph = { busIn, comp, reverb };
    this.routeDensity({ enabled: true, referenceLufs: null });
    return this.graph;
  }

  /** Bypass is a re-route, not a wet/dry mix; only touched when the toggle changes. */
  private routeDensity(density: DensitySetting): void {
    const g = this.ensureGraph();
    g.comp.threshold.value = densityThresholdDb(density.referenceLufs);
    if (this.densityRouted === density.enabled) return;
    this.densityRouted = density.enabled;
    const out = Tone.getDestination();
    g.busIn.disconnect();
    g.comp.disconnect();
    if (density.enabled) {
      g.busIn.connect(g.comp);
      g.comp.connect(out);
    } else {
      g.busIn.connect(out);
    }
  }

  private makePlayer(gain: Tone.Gain): Tone.Player {
    return new Tone.Player({ fadeIn: DECLICK_S, fadeOut: DECLICK_S }).connect(gain);
  }

  /** A Player cannot be emptied (`buffer =` ignores an unloaded buffer), so
   * "no buffer" means a fresh Player. Disposing the old one is the hard cut. */
  private replacePlayer(node: SlotNode): void {
    node.player.dispose();
    node.player = this.makePlayer(node.gain);
  }

  private node(slotId: string): SlotNode {
    const existing = this.slots.get(slotId);
    if (existing) return existing;
    const g = this.ensureGraph();
    const gain = new Tone.Gain(1);
    const send = new Tone.Gain(0);
    gain.connect(g.busIn);
    gain.connect(send);
    send.connect(g.reverb);
    const node: SlotNode = {
      player: this.makePlayer(gain),
      gain,
      send,
      regionId: null,
      path: null,
      buffer: null,
      override: null,
      loadSeq: 0,
    };
    this.slots.set(slotId, node);
    return node;
  }

  private disposeNode(slotId: string): void {
    const node = this.slots.get(slotId);
    if (!node) return;
    this.slots.delete(slotId);
    node.player.dispose();
    node.gain.dispose();
    node.send.dispose();
    node.override?.dispose();
    this.release(node.path);
  }

  // --- buffers ----------------------------------------------------------------------

  private async decode(path: string): Promise<Tone.ToneAudioBuffer> {
    const bytes = await fetchAudio(path);
    const audio = await Tone.getContext().decodeAudioData(bytes);
    return new Tone.ToneAudioBuffer(audio);
  }

  /** Render files are content-addressed, so one decode per path is safe to share. */
  private getBuffer(path: string): Promise<Tone.ToneAudioBuffer> {
    let p = this.buffers.get(path);
    if (!p) {
      p = this.decode(path);
      this.buffers.set(path, p);
      p.catch(() => this.buffers.delete(path));
    }
    return p;
  }

  private release(path: string | null): void {
    if (!path) return;
    for (const n of this.slots.values()) if (n.path === path) return;
    const p = this.buffers.get(path);
    this.buffers.delete(path);
    p?.then((b) => b.dispose()).catch(() => undefined);
  }

  private setSlotRender(slot: EngineSlot): void {
    const node = this.node(slot.id);
    if (node.regionId !== slot.regionId) {
      // A different region: the old buffer is the wrong music, not a stale conform.
      node.regionId = slot.regionId;
      const old = node.path;
      node.path = null;
      node.buffer = null;
      node.loadSeq++;
      if (!node.override) this.replacePlayer(node);
      this.release(old);
    }
    if (!slot.renderPath || node.path === slot.renderPath) return;
    const seq = ++node.loadSeq;
    this.getBuffer(slot.renderPath)
      .then((buffer) => {
        if (node.loadSeq !== seq || !this.slots.has(slot.id)) return;
        const old = node.path;
        node.path = slot.renderPath;
        node.buffer = buffer;
        if (!node.override) node.player.buffer = buffer;
        this.release(old);
        this.note('info', 'buffer loaded', { slot: slot.id, path: slot.renderPath, seconds: Number(buffer.duration.toFixed(3)), loaded: node.player.loaded });
        this.emit();
      })
      .catch((e: unknown) => {
        this.note('error', 'buffer load failed', { slot: slot.id, path: slot.renderPath, error: e instanceof Error ? e.message : String(e) });
        this.onError(`render load failed: ${e instanceof Error ? e.message : String(e)}`);
      });
  }

  // --- scheduling -------------------------------------------------------------------

  private fire(node: SlotNode, event: StepEvent, time: number): void {
    if (!node.player.loaded) {
      this.note('warn', 'trigger skipped: buffer not loaded', { region: node.regionId, path: node.path, startBeats: event.startBeats });
      return;
    }
    if (!stepFires(event, Math.random())) return;
    const bpm = this.transport.bpm.value;
    const bufferDuration = node.player.buffer.duration;
    const offset = beatsToSeconds(event.offsetBeats, bpm);
    if (offset >= bufferDuration) return;
    const wanted = beatsToSeconds(event.durationBeats, bpm);
    node.gain.gain.setValueAtTime(event.velocity, time);
    if (event.stepIndex !== -1) {
      const duration = Math.min(wanted, bufferDuration - offset);
      if (duration > 0) node.player.start(time, offset, duration);
      return;
    }
    // Hold fills the slot: a region shorter than the slot restarts from its
    // snap point (hard cut) until the slot ends, never trailing silence.
    let at = time;
    let remaining = wanted;
    let from = offset;
    while (remaining > 0.001) {
      const len = Math.min(bufferDuration - from, remaining);
      if (len <= 0) break;
      node.player.start(at, from, len);
      at += len;
      remaining -= len;
      from = 0;
    }
  }

  private schedule(program: EngineProgram): void {
    const t = this.transport;
    const ppq = t.PPQ;
    t.cancel(0);
    const total = totalBeats(program.slots);
    if (total <= 0) {
      t.loop = false;
      return;
    }
    t.loopStart = 0;
    t.loopEnd = `${beatsToTicks(total, ppq)}i`;
    t.loop = program.loop;
    let scheduled = 0;
    for (const { slotId, event } of progressionEvents(program.slots)) {
      const node = this.slots.get(slotId);
      if (!node) continue;
      t.schedule((time) => this.fire(node, event, time), `${beatsToTicks(event.startBeats, ppq)}i`);
      scheduled++;
    }
    if (!program.loop) t.schedule(() => this.stop(), `${beatsToTicks(total, ppq)}i`);
    this.note('info', 'scheduled', {
      events: scheduled,
      totalBeats: total,
      bpm: t.bpm.value,
      loop: program.loop,
      slotsWithRender: program.slots.filter((s) => s.renderPath).length,
      slots: program.slots.length,
    });
  }

  /** Apply a program. Cheap when nothing changed; reschedules otherwise. */
  sync(program: EngineProgram | null): void {
    if (!program) {
      this.transport.cancel(0);
      for (const id of [...this.slots.keys()]) this.disposeNode(id);
      return;
    }
    this.ensureGraph();
    this.transport.bpm.value = program.bpm;
    this.routeDensity(program.density);
    const live = new Set(program.slots.map((s) => s.id));
    for (const id of [...this.slots.keys()]) if (!live.has(id)) this.disposeNode(id);
    for (const slot of program.slots) {
      this.setSlotRender(slot);
      this.node(slot.id).send.gain.value = spaceSendLevel(program.space, slot.wetness);
    }
    this.schedule(program);
  }

  // --- transport ----------------------------------------------------------------------

  async play(): Promise<void> {
    await Tone.start();
    this.ensureGraph();
    if (this.transport.state === 'started') return;
    const snap = this.snapshot();
    this.note('info', 'play', { context: snap.contextState, clock: snap.clockSource, buffers: `${snap.loadedSlots}/${snap.totalSlots}` });
    this.transport.start();
    this.startTicker();
    this.armWatchdog();
  }

  stop(): void {
    this.disarmWatchdog();
    this.transport.stop();
    this.transport.ticks = 0;
    for (const node of this.slots.values()) {
      node.player.stop();
      this.clearOverride(node);
    }
    this.stopAudition();
    this.stopTicker();
    this.emit();
  }

  get playing(): boolean {
    return this.transport.state === 'started';
  }

  get positionBeats(): number {
    return this.transport.ticks / this.transport.PPQ;
  }

  subscribe(listener: (t: EngineTransport) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  private emit(): void {
    const t = this.snapshot();
    for (const l of this.listeners) l(t);
  }

  snapshot(): EngineTransport {
    const ctx = Tone.getContext();
    const raw = rawContext();
    let loaded = 0;
    for (const n of this.slots.values()) if (n.player.loaded) loaded++;
    const now = performance.now();
    return {
      playing: this.playing,
      positionBeats: this.positionBeats,
      contextState: ctx.state,
      sampleRate: ctx.sampleRate,
      loadedSlots: loaded,
      totalSlots: this.slots.size,
      clockSource: clockOf(ctx).clockSource,
      ticking: this.ticks.ticking(now),
      ticksPerSecond: this.ticks.rate(now),
      clockNote: this.clockNote,
      baseLatency: typeof raw.baseLatency === 'number' ? raw.baseLatency : null,
      outputLatency: typeof raw.outputLatency === 'number' ? raw.outputLatency : null,
      lookAhead: ctx.lookAhead,
      sinkId: typeof raw.sinkId === 'string' ? raw.sinkId : null,
    };
  }

  private startTicker(): void {
    this.stopTicker();
    this.ticker = setInterval(() => this.emit(), 50);
    this.emit();
  }

  private stopTicker(): void {
    if (this.ticker) clearInterval(this.ticker);
    this.ticker = null;
  }

  // --- audition -----------------------------------------------------------------------

  /** Solo audition of any sidecar audio path (raw region slice or a render). */
  async auditionRegion(path: string): Promise<void> {
    await Tone.start();
    this.stopAudition();
    const buffer = await this.decode(path);
    const player = new Tone.Player({ fadeIn: DECLICK_S, fadeOut: DECLICK_S }).toDestination();
    player.buffer = buffer;
    player.onstop = () => {
      if (this.audition === player) this.audition = null;
      // Tone still touches the source after onstop; dispose on the next tick.
      setTimeout(() => {
        player.dispose();
        buffer.dispose();
      }, 0);
    };
    this.audition = player;
    player.start();
  }

  private stopAudition(): void {
    const p = this.audition;
    this.audition = null;
    p?.stop();
  }

  /** Temporarily substitute a slot's buffer while the progression plays (§5.1
   * "in context"). The substitute is dropped on stop(). */
  async auditionInContext(slotId: string, path: string): Promise<void> {
    const node = this.slots.get(slotId);
    if (!node) {
      this.onError('audition in context: slot is not scheduled');
      return;
    }
    const buffer = await this.decode(path);
    node.override?.dispose();
    node.override = buffer;
    node.player.buffer = buffer;
    if (!this.playing) {
      // This runs from a render event, not a click: a context that is still
      // suspended cannot be resumed here on webviews that need a gesture.
      await Tone.start();
      if (Tone.getContext().state !== 'running') {
        this.onError('press Play once to enable audio, then audition in context');
        return;
      }
      await this.play();
    }
  }

  private clearOverride(node: SlotNode): void {
    if (!node.override) return;
    node.override.dispose();
    node.override = null;
    if (node.buffer) node.player.buffer = node.buffer;
    else this.replacePlayer(node);
  }
}

export const engine = new PlaybackEngine();
