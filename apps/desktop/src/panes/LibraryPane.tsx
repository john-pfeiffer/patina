import { useState } from 'react';
import { isTauri, pickAudioFiles } from '../api/shell';
import type { Sample } from '../api/types';
import { ProgressBar } from '../components/ProgressBar';
import { regionsById, selectCurrentSet, selectSelectedSlot, useStore } from '../store';
import { errorMessage, num, pct, secs, signed } from '../util/format';
import { RegionRow } from './RegionRow';

function AddFiles() {
  const ingest = useStore((s) => s.ingest);
  const [paths, setPaths] = useState('');
  const [busy, setBusy] = useState(false);
  if (isTauri()) {
    return (
      <button
        type="button"
        disabled={busy}
        onClick={() => {
          setBusy(true);
          pickAudioFiles()
            .then((picked) => (picked.length ? ingest(picked) : undefined))
            .catch((e: unknown) => useStore.setState({ lastError: errorMessage(e) }))
            .finally(() => setBusy(false));
        }}
      >
        {busy ? 'adding…' : 'add files…'}
      </button>
    );
  }
  return (
    <span className="row">
      <input
        type="text"
        placeholder="/abs/path/one.wav, /abs/dir"
        value={paths}
        style={{ width: 220 }}
        onChange={(e) => setPaths(e.target.value)}
        onKeyDown={(e) => {
          if (e.key !== 'Enter') return;
          const list = paths.split(/[,\n]/).map((p) => p.trim()).filter(Boolean);
          if (list.length) void ingest(list).then(() => setPaths(''));
        }}
      />
      <button
        type="button"
        onClick={() => {
          const list = paths.split(/[,\n]/).map((p) => p.trim()).filter(Boolean);
          if (list.length) void ingest(list).then(() => setPaths(''));
        }}
      >
        add paths
      </button>
    </span>
  );
}

function SampleHeader({ sample, collapsed, onToggle }: { sample: Sample; collapsed: boolean; onToggle: () => void }) {
  const reanalyze = useStore((s) => s.reanalyze);
  const removeSample = useStore((s) => s.removeSample);
  const st = sample.analysis_status;
  return (
    <div className="sample-head">
      <button type="button" className="disclosure" onClick={onToggle} aria-label={collapsed ? 'expand' : 'collapse'}>
        {collapsed ? '▸' : '▾'}
      </button>
      <span className="sample-name" title={sample.path}>
        {sample.name}
      </span>
      {st === 'done' ? (
        <span className="badge badge-ok">analysed</span>
      ) : st === 'failed' ? (
        <span className="badge badge-bad" title={sample.analysis_error ?? ''}>
          failed
        </span>
      ) : (
        <ProgressBar value={sample.analysis_progress} label={`${st} ${Math.round(sample.analysis_progress * 100)}%`} />
      )}
      <span className="readout">{secs(sample.duration)}</span>
      <span className="readout">
        {sample.sample_rate ?? '—'} Hz · {sample.channels ?? '—'} ch
      </span>
      <span className="readout" title={`bpm confidence ${pct(sample.bpm_confidence)}`}>
        {num(sample.detected_bpm)} bpm
      </span>
      <span className="readout" title={`key confidence ${pct(sample.key_confidence)}`}>
        key {sample.key_estimate ?? '—'}
      </span>
      <span className="readout" title="tuning offset from A440">
        {signed(sample.tuning_cents)} ct
      </span>
      <span className="spacer" />
      <button type="button" onClick={() => void reanalyze(sample.id)} disabled={st === 'running' || st === 'pending'}>
        reanalyze
      </button>
      <button type="button" onClick={() => void removeSample(sample.id)} title="remove sample from library">
        ×
      </button>
    </div>
  );
}

export function LibraryPane() {
  const samples = useStore((s) => s.samples);
  const currentSet = useStore(selectCurrentSet);
  const selectedSlot = useStore(selectSelectedSlot);
  const collapsed = useStore((s) => s.ui.collapsedSamples);
  const setUi = useStore((s) => s.setUi);
  const memberIds = new Set(currentSet?.members.map((m) => m.region.id) ?? []);
  const regionCount = regionsById(samples).size;

  return (
    <section className="pane pane-library">
      <header className="pane-head">
        <h2>Library</h2>
        <span className="readout">
          {samples.length} samples · {regionCount} regions
        </span>
        <span className="spacer" />
        <AddFiles />
      </header>
      <div className="pane-body">
        {samples.length === 0 && <p className="muted">No samples. Add audio files or directories; analysis runs in the background.</p>}
        {samples.map((sample) => {
          const isCollapsed = collapsed[sample.id] === true;
          return (
            <div key={sample.id} className="sample">
              <SampleHeader
                sample={sample}
                collapsed={isCollapsed}
                onToggle={() => {
                  const next = { ...collapsed };
                  if (isCollapsed) delete next[sample.id];
                  else next[sample.id] = true;
                  setUi({ collapsedSamples: next });
                }}
              />
              {!isCollapsed &&
                sample.regions.map((region) => (
                  <RegionRow
                    key={region.id}
                    region={region}
                    isMember={memberIds.has(region.id)}
                    hasSet={currentSet !== null}
                    selectedSlot={selectedSlot}
                  />
                ))}
              {!isCollapsed && sample.regions.length === 0 && sample.analysis_status === 'done' && (
                <p className="muted indent">no harmonically stable regions found</p>
              )}
            </div>
          );
        })}
      </div>
    </section>
  );
}
