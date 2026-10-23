import { useState } from 'react';
import {
  MODULE_NAMES,
  type ModuleName,
  type RepitchMode,
  type StretchMethod,
  type TemporalSettings,
  type TonalSettings,
} from '../api/types';
import { Knob } from '../components/Knob';
import { NumberField } from '../components/NumberField';
import { samplesById, selectCurrentSet, sessionBpm, useStore } from '../store';
import { MemberRow } from './MemberRow';

const AMOUNT_MODULES: ReadonlySet<ModuleName> = new Set<ModuleName>(['tilt', 'width', 'space']);
const MODULE_HINT: Record<ModuleName, string> = {
  tuning: 'correct tuning_cents to A440 by resampling',
  loudness: 'match integrated LUFS to the anchor',
  tilt: 'octave-band difference curve, capped ±6 dB/band, scaled by amount',
  lowend: 'high-pass at anchor low_rolloff, mono below anchor bass_mono; never adds low end',
  noise: 'bed the anchor’s harvested floor under quieter regions; never cleans',
  width: 'scale Side vs Mid per band toward the anchor, max 1.5× widening',
  density: 'bus compressor 2:1 (playback graph, not rendered)',
  space: 'reverb send only for regions drier than the anchor (playback graph)',
};

function SetPicker() {
  const sets = useStore((s) => s.sets);
  const currentSetId = useStore((s) => s.currentSetId);
  const selectSet = useStore((s) => s.selectSet);
  const createSet = useStore((s) => s.createSet);
  const removeSet = useStore((s) => s.removeSet);
  const [name, setName] = useState('');
  return (
    <div className="row">
      <select value={currentSetId ?? ''} onChange={(e) => void selectSet(e.target.value || null)}>
        {sets.length === 0 && <option value="">(no sets)</option>}
        {sets.map((s) => (
          <option key={s.id} value={s.id}>
            {s.name}
          </option>
        ))}
      </select>
      <input
        type="text"
        placeholder="new set name"
        value={name}
        style={{ width: 110 }}
        onChange={(e) => setName(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter' && name.trim()) void createSet(name.trim()).then(() => setName(''));
        }}
      />
      <button type="button" onClick={() => void createSet(name.trim() || 'Untitled set').then(() => setName(''))}>
        new
      </button>
      <button type="button" disabled={!currentSetId} onClick={() => currentSetId && void removeSet(currentSetId)} title="delete set">
        ×
      </button>
    </div>
  );
}

export function SetPane() {
  const set = useStore(selectCurrentSet);
  const samples = useStore((s) => s.samples);
  const deltas = useStore((s) => s.deltas);
  const memberRenders = useStore((s) => s.memberRenders);
  const patchSet = useStore((s) => s.patchSet);

  const names = samplesById(samples);
  const bpm = sessionBpm(set);

  const setModule = (name: ModuleName, patch: Partial<TonalSettings[ModuleName]>) => {
    if (!set) return;
    const tonal: TonalSettings = { ...set.tonal_settings, [name]: { ...set.tonal_settings[name], ...patch } };
    void patchSet({ tonal_settings: tonal });
  };
  const setTemporal = (patch: Partial<TemporalSettings>) => {
    if (!set) return;
    void patchSet({ temporal_settings: { ...set.temporal_settings, ...patch } });
  };

  return (
    <section className="pane pane-set">
      <header className="pane-head">
        <h2>Set</h2>
        <SetPicker />
      </header>
      {!set ? (
        <div className="pane-body">
          <p className="muted">Create a set, then add regions from the library.</p>
        </div>
      ) : (
        <div className="pane-body">
          <div className="block">
            <div className="row">
              <NumberField
                label="session bpm"
                value={set.session_bpm}
                min={20}
                max={400}
                step={0.1}
                width={64}
                placeholder={String(bpm)}
                onCommit={(v) => v !== null && void patchSet({ session_bpm: v, bpm_pinned: true })}
              />
              <label className="check" title="pinned bpm does not follow the anchor">
                <input type="checkbox" checked={set.bpm_pinned} onChange={(e) => void patchSet({ bpm_pinned: e.target.checked })} />
                pinned
              </label>
              <button type="button" onClick={() => void patchSet({ session_bpm: bpm / 2, bpm_pinned: true })} title="halve session bpm">
                ½×
              </button>
              <button type="button" onClick={() => void patchSet({ session_bpm: bpm * 2, bpm_pinned: true })} title="double session bpm">
                2×
              </button>
              {set.session_bpm === null && <span className="muted">(no anchor tempo yet; engine uses {bpm})</span>}
            </div>
            <div className="row">
              <label className="field">
                <span className="field-label">anchor</span>
                <select
                  value={set.anchor_region_id ?? ''}
                  onChange={(e) => e.target.value && void patchSet({ anchor_region_id: e.target.value })}
                >
                  {!set.anchor_region_id && <option value="">(none)</option>}
                  {set.members.map((m) => (
                    <option key={m.region.id} value={m.region.id}>
                      {names.get(m.region.sample_id)?.name ?? m.region.sample_id} · {m.region.chord_root ?? '?'}
                      {m.region.chord_quality === 'min' ? 'm' : ''}
                    </option>
                  ))}
                </select>
              </label>
            </div>
          </div>

          <div className="block">
            <h3>Members ({set.members.length})</h3>
            {set.members.length === 0 && <p className="muted">No members. Use “+ set” on a region in the library.</p>}
            {set.members.map((m) => (
              <MemberRow
                key={m.region.id}
                member={m}
                sampleName={names.get(m.region.sample_id)?.name ?? m.region.sample_id}
                isAnchor={m.region.id === set.anchor_region_id}
                deltas={deltas[m.region.id]}
                render={memberRenders[m.region.id]}
              />
            ))}
          </div>

          <div className="block">
            <h3>Tonal modules</h3>
            {MODULE_NAMES.map((name) => {
              const mod = set.tonal_settings[name];
              return (
                <div key={name} className="row module-row" title={MODULE_HINT[name]}>
                  <label className="check module-name">
                    <input type="checkbox" checked={mod.enabled} onChange={(e) => setModule(name, { enabled: e.target.checked })} />
                    {name}
                  </label>
                  {AMOUNT_MODULES.has(name) && (
                    <Knob
                      label="amount"
                      value={mod.amount ?? 100}
                      min={0}
                      max={100}
                      step={5}
                      unit="%"
                      disabled={!mod.enabled}
                      onCommit={(v) => setModule(name, { amount: v })}
                    />
                  )}
                  {(name === 'density' || name === 'space') && <span className="muted">bus</span>}
                </div>
              );
            })}
          </div>

          <div className="block">
            <h3>Temporal</h3>
            <div className="row">
              <label className="field">
                <span className="field-label">stretch</span>
                <select
                  value={set.temporal_settings.stretch_method}
                  onChange={(e) => setTemporal({ stretch_method: e.target.value as StretchMethod })}
                >
                  <option value="elastic">elastic (constant pitch)</option>
                  <option value="repitch">repitch (pitch follows tempo)</option>
                </select>
              </label>
              <label className="field" title="Plain resamples (length absorbed by the stretch stage); Elastic shifts at constant length">
                <span className="field-label">repitch mode</span>
                <select
                  value={set.temporal_settings.repitch_mode}
                  onChange={(e) => setTemporal({ repitch_mode: e.target.value as RepitchMode })}
                >
                  <option value="plain">plain</option>
                  <option value="elastic">elastic</option>
                </select>
              </label>
            </div>
            <div className="row">
              <Knob
                label="warp strength"
                value={set.temporal_settings.warp_strength}
                min={0}
                max={100}
                step={5}
                unit="%"
                onCommit={(v) => setTemporal({ warp_strength: v })}
              />
              <label className="check" title="start playback at the first onset (per-region “keep pre-roll” overrides)">
                <input
                  type="checkbox"
                  checked={set.temporal_settings.transient_snap}
                  onChange={(e) => setTemporal({ transient_snap: e.target.checked })}
                />
                transient snap
              </label>
              <span className="readout" title="declick ramp at region start and end">
                declick {set.temporal_settings.declick_ms} ms
              </span>
            </div>
          </div>
        </div>
      )}
    </section>
  );
}
