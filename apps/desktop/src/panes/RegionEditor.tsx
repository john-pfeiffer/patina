import { useCallback, useEffect, useRef, useState } from 'react';
import { api } from '../api/client';
import { ANALYSIS_QUALITIES_UI, NOTE_NAMES, type Peaks, type Region, type Sample } from '../api/types';
import { engine } from '../audio/engine';
import { Dialog } from '../components/Dialog';
import { NumberField } from '../components/NumberField';
import { regionsById, samplesById, selectCurrentProgression, useStore } from '../store';
import { chordSymbol } from '../util/chords';
import { errorMessage, num, secs } from '../util/format';
import {
  clampBounds,
  effectiveSnap,
  fitView,
  panView,
  snapTime,
  timeToX,
  xToTime,
  zoomView,
  type Bounds,
  type SnapMode,
  type View,
} from '../util/waveform';
import { RegionEdit } from './RegionEdit';

const HANDLE_PX = 7;
const SNAP_PX = 9;
const BINS = 1600;
const HEIGHT = 170;

type DragKind = 'start' | 'end' | 'snap' | 'pan';
type Drag = { kind: DragKind; originX: number; originView: View; originBounds: Bounds } | null;

/** The waveform with the draggable bounds and snap point. Pure drawing;
 * everything it needs comes in as props. */
function Waveform({
  sample,
  bounds,
  view,
  peaks,
  snapMode,
  keepPreroll,
  onBounds,
  onView,
}: {
  sample: Sample;
  bounds: Bounds;
  view: View;
  peaks: Peaks | null;
  snapMode: SnapMode;
  keepPreroll: boolean;
  onBounds: (b: Bounds) => void;
  onView: (v: View) => void;
}) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const drag = useRef<Drag>(null);
  const duration = sample.duration ?? view.end;
  const snapAt = effectiveSnap(bounds, sample.onsets, keepPreroll);

  useEffect(() => {
    const c = canvas.current;
    if (!c) return;
    const dpr = window.devicePixelRatio || 1;
    c.width = Math.round(view.width * dpr);
    c.height = Math.round(HEIGHT * dpr);
    const g = c.getContext('2d');
    if (!g) return;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    const css = getComputedStyle(c);
    const col = (name: string, fallback: string) => css.getPropertyValue(name).trim() || fallback;
    g.fillStyle = '#fff';
    g.fillRect(0, 0, view.width, HEIGHT);
    // region shading
    const xs = timeToX(view, bounds.start);
    const xe = timeToX(view, bounds.end);
    g.fillStyle = col('--bg-selected', '#dfe6ee');
    g.fillRect(xs, 0, Math.max(1, xe - xs), HEIGHT);
    // beats
    for (const b of sample.beats) {
      if (b < view.start || b > view.end) continue;
      const x = timeToX(view, b);
      g.strokeStyle = sample.downbeats.includes(b) ? col('--border-strong', '#8a8a8a') : col('--border', '#b8b8b8');
      g.lineWidth = 1;
      g.beginPath();
      g.moveTo(x + 0.5, 0);
      g.lineTo(x + 0.5, HEIGHT);
      g.stroke();
    }
    // waveform
    if (peaks && peaks.peaks.length > 0) {
      const mid = HEIGHT / 2;
      const span = peaks.end - peaks.start;
      g.fillStyle = col('--accent', '#2d4f7a');
      const n = peaks.peaks.length;
      for (let i = 0; i < n; i++) {
        const t0 = peaks.start + (i / n) * span;
        const t1 = peaks.start + ((i + 1) / n) * span;
        const x0 = timeToX(view, t0);
        const x1 = timeToX(view, t1);
        if (x1 < 0 || x0 > view.width) continue;
        const h = Math.max(1, (peaks.peaks[i] ?? 0) * (HEIGHT / 2 - 12));
        g.fillRect(x0, mid - h, Math.max(1, x1 - x0), h * 2);
      }
    }
    // onsets: short ticks at the bottom
    g.strokeStyle = col('--stale', '#8a5a00');
    g.lineWidth = 1;
    for (const o of sample.onsets) {
      if (o < view.start || o > view.end) continue;
      const x = timeToX(view, o);
      g.beginPath();
      g.moveTo(x + 0.5, HEIGHT - 14);
      g.lineTo(x + 0.5, HEIGHT);
      g.stroke();
    }
    // bounds
    const edge = (x: number) => {
      g.strokeStyle = col('--fg', '#1c1c1c');
      g.lineWidth = 2;
      g.beginPath();
      g.moveTo(x, 0);
      g.lineTo(x, HEIGHT);
      g.stroke();
    };
    edge(xs);
    edge(xe);
    // snap point: triangle at the top plus a line
    const xsnap = timeToX(view, snapAt);
    g.strokeStyle = col('--ok', '#2f6f3a');
    g.fillStyle = col('--ok', '#2f6f3a');
    g.lineWidth = bounds.snap === null ? 1 : 2;
    g.setLineDash(bounds.snap === null ? [3, 3] : []);
    g.beginPath();
    g.moveTo(xsnap + 0.5, 0);
    g.lineTo(xsnap + 0.5, HEIGHT);
    g.stroke();
    g.setLineDash([]);
    g.beginPath();
    g.moveTo(xsnap - 7, 0);
    g.lineTo(xsnap + 7, 0);
    g.lineTo(xsnap, 10);
    g.closePath();
    g.fill();
  }, [view, bounds, peaks, sample, snapAt]);

  const hit = (x: number): DragKind => {
    if (Math.abs(x - timeToX(view, snapAt)) <= SNAP_PX) return 'snap';
    if (Math.abs(x - timeToX(view, bounds.start)) <= HANDLE_PX) return 'start';
    if (Math.abs(x - timeToX(view, bounds.end)) <= HANDLE_PX) return 'end';
    return 'pan';
  };

  const onDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    drag.current = { kind: hit(x), originX: x, originView: view, originBounds: bounds };
    e.currentTarget.setPointerCapture(e.pointerId);
  };
  const onMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const d = drag.current;
    if (!d) {
      const k = hit(x);
      e.currentTarget.style.cursor = k === 'pan' ? 'grab' : 'col-resize';
      return;
    }
    const tolerance = (SNAP_PX / view.width) * (view.end - view.start);
    if (d.kind === 'pan') {
      const dt = xToTime(d.originView, d.originX) - xToTime(d.originView, x);
      onView(panView(d.originView, dt, duration));
      return;
    }
    const raw = xToTime(view, x);
    const t = snapTime(raw, snapMode, sample.onsets, sample.beats, tolerance);
    if (d.kind === 'start') onBounds(clampBounds({ ...bounds, start: Math.min(t, bounds.end - 0.05) }, duration));
    else if (d.kind === 'end') onBounds(clampBounds({ ...bounds, end: Math.max(t, bounds.start + 0.05) }, duration));
    else onBounds(clampBounds({ ...bounds, snap: t }, duration));
  };
  const onUp = (e: React.PointerEvent<HTMLCanvasElement>) => {
    drag.current = null;
    e.currentTarget.releasePointerCapture(e.pointerId);
  };
  const onWheel = (e: React.WheelEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const focus = xToTime(view, e.clientX - rect.left);
    onView(zoomView(view, focus, e.deltaY > 0 ? 1.25 : 0.8, duration));
  };

  return (
    <canvas
      ref={canvas}
      className="waveform"
      style={{ width: view.width, height: HEIGHT }}
      onPointerDown={onDown}
      onPointerMove={onMove}
      onPointerUp={onUp}
      onPointerCancel={onUp}
      onWheel={onWheel}
      title="drag the edges to trim, drag the green marker to set the transient (snap point), drag elsewhere to pan, wheel to zoom"
    />
  );
}

/** Visual editor for a region: bounds and transient over the waveform with
 * beat and onset overlays, chord fields, audition, and the per-region
 * temporal/bypass settings. Bounds, snap and chord are applied together;
 * the rest patches immediately as before. */
export function RegionEditor() {
  const target = useStore((s) => s.ui.editor);
  const samples = useStore((s) => s.samples);
  const setUi = useStore((s) => s.setUi);
  const patchRegion = useStore((s) => s.patchRegion);
  const createRegion = useStore((s) => s.createRegion);
  const previewInContext = useStore((s) => s.previewInContext);
  const patchSlot = useStore((s) => s.patchSlot);
  const progression = useStore(selectCurrentProgression);
  const region = target ? (regionsById(samples).get(target.regionId) ?? null) : null;
  const sample = region ? (samplesById(samples).get(region.sample_id) ?? null) : null;
  const slot = target?.slotId ? (progression?.slots.find((x) => x.id === target.slotId) ?? null) : null;
  const close = useCallback(() => setUi({ dialog: null, editor: null }), [setUi]);

  if (!target || !region || !sample) {
    return (
      <Dialog title="Region" onClose={close}>
        <p className="muted">This region is no longer in the library.</p>
      </Dialog>
    );
  }
  return <RegionEditorBody key={region.id} region={region} sample={sample} slotId={slot?.id ?? null} slotLabel={slot ? `${slot.idx + 1} (${slot.chord_symbol})` : null} onClose={close} patchRegion={patchRegion} createRegion={createRegion} previewInContext={previewInContext} patchSlot={patchSlot} />;
}

function RegionEditorBody({
  region,
  sample,
  slotId,
  slotLabel,
  onClose,
  patchRegion,
  createRegion,
  previewInContext,
  patchSlot,
}: {
  region: Region;
  sample: Sample;
  slotId: string | null;
  slotLabel: string | null;
  onClose: () => void;
  patchRegion: (id: string, patch: Parameters<typeof api.regions.patch>[1]) => Promise<void>;
  createRegion: (sampleId: string, body: Parameters<typeof api.samples.createRegion>[1]) => Promise<Region | null>;
  previewInContext: (region: Region, slotId: string) => Promise<void>;
  patchSlot: (id: string, patch: { region_id?: string; transpose_semitones?: number }) => Promise<void>;
}) {
  const duration = sample.duration ?? region.end_s;
  const width = 760;
  const [bounds, setBounds] = useState<Bounds>({ start: region.start_s, end: region.end_s, snap: region.user_overrides.snap_s });
  const [view, setView] = useState<View>(() => fitView(region.start_s, region.end_s, duration, width));
  const [peaks, setPeaks] = useState<Peaks | null>(null);
  const [snapMode, setSnapMode] = useState<SnapMode>('onset');
  const [root, setRoot] = useState<string>(region.chord_root ?? '');
  const [quality, setQuality] = useState<string>(region.chord_quality ?? '');
  const [bass, setBass] = useState<string>(region.bass_note ?? '');
  const [saving, setSaving] = useState(false);
  const o = region.user_overrides;
  const keepPreroll = o.keep_preroll;

  const dirty =
    Math.abs(bounds.start - region.start_s) > 1e-6 ||
    Math.abs(bounds.end - region.end_s) > 1e-6 ||
    (bounds.snap ?? null) !== (o.snap_s ?? null) ||
    root !== (region.chord_root ?? '') ||
    quality !== (region.chord_quality ?? '') ||
    bass !== (region.bass_note ?? '');

  // Peaks follow the view (debounced); a zoomed view asks the sidecar for a narrower window.
  useEffect(() => {
    const t = setTimeout(() => {
      void api.samples
        .peaks(sample.id, BINS, view.start, view.end)
        .then(setPeaks)
        .catch((e: unknown) => useStore.setState({ lastError: errorMessage(e) }));
    }, 60);
    return () => clearTimeout(t);
  }, [sample.id, view.start, view.end]);

  const snapAt = effectiveSnap(bounds, sample.onsets, keepPreroll);
  const snapSource = bounds.snap !== null ? 'set by hand' : keepPreroll ? 'region start (pre-roll kept)' : sample.onsets.some((x) => Math.abs(x - snapAt) < 1e-6) ? 'detected onset' : 'region start (no onset inside)';
  const beatsInside = sample.beats.filter((b) => b >= bounds.start && b < bounds.end).length;
  const lengthBeats = region.declared_length_beats ?? (beatsInside || null);
  const draftSymbol = chordSymbol(root || null, quality || null, bass || null);

  const apply = async () => {
    setSaving(true);
    try {
      const patch: Parameters<typeof api.regions.patch>[1] = {};
      if (Math.abs(bounds.start - region.start_s) > 1e-6) patch.start_s = bounds.start;
      if (Math.abs(bounds.end - region.end_s) > 1e-6) patch.end_s = bounds.end;
      if ((bounds.snap ?? null) !== (o.snap_s ?? null)) patch.user_overrides = { ...o, snap_s: bounds.snap };
      if (root && root !== (region.chord_root ?? '')) patch.chord_root = root;
      if (quality && quality !== (region.chord_quality ?? '')) patch.chord_quality = quality;
      if (bass && bass !== (region.bass_note ?? '')) patch.bass_note = bass;
      await patchRegion(region.id, patch);
    } finally {
      setSaving(false);
    }
  };

  const saveAsNew = async () => {
    setSaving(true);
    try {
      const created = await createRegion(sample.id, {
        start_s: bounds.start,
        end_s: bounds.end,
        chord_symbol: draftSymbol ?? undefined,
        declared_length_beats: region.declared_length_beats ?? undefined,
      });
      if (created && bounds.snap !== null) await patchRegion(created.id, { user_overrides: { ...created.user_overrides, snap_s: bounds.snap } });
      if (created && slotId) await patchSlot(slotId, { region_id: created.id });
      if (created) useStore.getState().openEditor(created.id, slotId);
    } finally {
      setSaving(false);
    }
  };

  const solo = () => void engine.auditionRegion(api.samples.audioPath(sample.id, snapAt, bounds.end)).catch((e: unknown) => useStore.setState({ lastError: errorMessage(e) }));

  const tools = (
    <>
      <button type="button" onClick={() => setView(fitView(bounds.start, bounds.end, duration, width))} title="zoom to the region">
        region
      </button>
      <button type="button" onClick={() => setView({ start: 0, end: duration, width })} title="show the whole file">
        file
      </button>
      <button type="button" onClick={() => setView((v) => zoomView(v, (v.start + v.end) / 2, 0.6, duration))}>
        +
      </button>
      <button type="button" onClick={() => setView((v) => zoomView(v, (v.start + v.end) / 2, 1.6, duration))}>
        −
      </button>
    </>
  );

  return (
    <Dialog title={`Region · ${sample.name}${slotLabel ? ` · slot ${slotLabel}` : ''}`} onClose={onClose} tools={tools} width={width + 24}>
      <div className="editor">
        <Waveform sample={sample} bounds={bounds} view={{ ...view, width }} peaks={peaks} snapMode={snapMode} keepPreroll={keepPreroll} onBounds={setBounds} onView={setView} />
        <div className="row editor-ruler">
          <span className="readout">
            view {secs(view.start)}–{secs(view.end)} of {secs(duration)}
          </span>
          <span className="spacer" />
          <span className="field-label">drag snaps to</span>
          {(['onset', 'beat', 'free'] as SnapMode[]).map((m) => (
            <button key={m} type="button" className={snapMode === m ? 'active' : ''} onClick={() => setSnapMode(m)}>
              {m}
            </button>
          ))}
        </div>

        <div className="editor-grid">
          <div className="editor-col">
            <h3>Bounds</h3>
            <div className="row">
              <NumberField label="start" value={Number(bounds.start.toFixed(3))} min={0} step={0.01} unit="s" width={76} onCommit={(v) => v !== null && setBounds(clampBounds({ ...bounds, start: v }, duration))} />
              <NumberField label="end" value={Number(bounds.end.toFixed(3))} min={0} step={0.01} unit="s" width={76} onCommit={(v) => v !== null && setBounds(clampBounds({ ...bounds, end: v }, duration))} />
              <span className="readout">
                {num(bounds.end - bounds.start, 3)} s · {lengthBeats !== null ? `${num(lengthBeats, 1)} b` : '— b'}
              </span>
            </div>
            <h3>Transient (snap point)</h3>
            <div className="row">
              <span className="readout" title="where playback and the beat grid start">
                {secs(snapAt)} · {snapSource}
              </span>
              <button type="button" disabled={bounds.snap === null} onClick={() => setBounds({ ...bounds, snap: null })} title="use the detected first onset again">
                reset
              </button>
              <button
                type="button"
                onClick={() => {
                  const inside = sample.onsets.filter((x) => x >= bounds.start && x < bounds.end - 0.05);
                  const cur = snapAt;
                  const next = inside.find((x) => x > cur + 1e-6) ?? inside[0];
                  if (next !== undefined) setBounds({ ...bounds, snap: next });
                }}
                title="move the snap point to the next detected transient inside the region"
              >
                next onset
              </button>
            </div>
            <p className="muted">
              Transients that landed on the wrong eighth: drag the green marker onto the right attack (it snaps to onsets), or use next onset. Keep pre-roll and nudge are below.
            </p>
            <h3>Chord</h3>
            <div className="row">
              <label className="field">
                <span className="field-label">root</span>
                <select value={root} onChange={(e) => setRoot(e.target.value)}>
                  <option value="">—</option>
                  {NOTE_NAMES.map((n) => (
                    <option key={n} value={n}>
                      {n}
                    </option>
                  ))}
                </select>
              </label>
              <label className="field">
                <span className="field-label">quality</span>
                <select value={quality} onChange={(e) => setQuality(e.target.value)}>
                  <option value="">—</option>
                  {ANALYSIS_QUALITIES_UI.map((q) => (
                    <option key={q} value={q}>
                      {q}
                    </option>
                  ))}
                </select>
              </label>
              <label className="field">
                <span className="field-label">bass</span>
                <select value={bass} onChange={(e) => setBass(e.target.value)}>
                  <option value="">—</option>
                  {NOTE_NAMES.map((n) => (
                    <option key={n} value={n}>
                      {n}
                    </option>
                  ))}
                </select>
              </label>
              <span className="chord">{draftSymbol ?? '?'}</span>
              <span className="readout" title="chord confidence">
                conf {num(region.chord_confidence, 2)}
              </span>
            </div>
          </div>
          <div className="editor-col">
            <h3>Hear</h3>
            <div className="row">
              <button type="button" onClick={solo} title="play the draft selection from its snap point, raw">
                solo draft
              </button>
              <button
                type="button"
                disabled={!slotId || dirty}
                title={!slotId ? 'open the editor from a slot to audition in context' : dirty ? 'apply first, then audition conformed in context' : 'conform and audition at the slot, in the progression'}
                onClick={() => slotId && void previewInContext(region, slotId)}
              >
                in ctx
              </button>
            </div>
            <h3>Save</h3>
            <div className="row">
              <button type="button" className={dirty ? 'active' : ''} disabled={!dirty || saving} onClick={() => void apply()} title="write bounds, snap point and chord to this region; it becomes yours and survives rescans">
                apply
              </button>
              <button type="button" disabled={!dirty || saving} onClick={() => setBounds({ start: region.start_s, end: region.end_s, snap: o.snap_s })} title="discard the draft">
                revert
              </button>
              <button type="button" disabled={saving} onClick={() => void saveAsNew()} title={slotId ? 'keep this region as it is; add the draft as a new region and put it in the slot' : 'keep this region as it is; add the draft as a new region on the sample'}>
                save as new{slotId ? ' + assign' : ''}
              </button>
            </div>
            <p className="muted">
              {region.source === 'user' ? 'This region is yours: a rescan of the folder leaves it alone.' : 'Applying makes this region yours, so a rescan never re-cuts it.'}
            </p>
          </div>
        </div>

        <h3>Temporal &amp; bypass</h3>
        <RegionEdit region={region} />
      </div>
    </Dialog>
  );
}
