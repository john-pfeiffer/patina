import { useState } from 'react';
import { isTauri, pickAudioFiles } from '../api/shell';
import type { Sample } from '../api/types';
import { ProgressBar } from '../components/ProgressBar';
import { regionsById, selectCurrentSet, selectSelectedSlot, useStore } from '../store';
import { errorMessage, num, pct, secs, signed } from '../util/format';
import { groupLibrary, type LibraryFolder } from '../util/library';
import { RegionRow } from './RegionRow';

function AddFiles() {
  const ingest = useStore((s) => s.ingest);
  const setUi = useStore((s) => s.setUi);
  const [paths, setPaths] = useState('');
  const [busy, setBusy] = useState(false);
  const folders = (
    <button type="button" onClick={() => setUi({ dialog: 'preferences', prefsTab: 'library' })} title="scan whole folders (your Splice library) from Preferences › Library">
      folders…
    </button>
  );
  if (isTauri()) {
    return (
      <>
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
        {folders}
      </>
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
      {folders}
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
      ) : st === 'missing' ? (
        <span className="badge badge-stale" title="not found at the last scan of its folder">
          missing
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
      <button type="button" onClick={() => void reanalyze(sample.id)} disabled={st === 'running' || st === 'pending' || st === 'missing'}>
        reanalyze
      </button>
      <button type="button" onClick={() => void removeSample(sample.id)} title="remove sample from library">
        ×
      </button>
    </div>
  );
}

function SampleBlock({ sample }: { sample: Sample }) {
  const currentSet = useStore(selectCurrentSet);
  const selectedSlot = useStore(selectSelectedSlot);
  const collapsed = useStore((s) => s.ui.collapsedSamples);
  const setUi = useStore((s) => s.setUi);
  const memberIds = new Set(currentSet?.members.map((m) => m.region.id) ?? []);
  const isCollapsed = collapsed[sample.id] === true;
  return (
    <div className="sample">
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
          <RegionRow key={region.id} region={region} isMember={memberIds.has(region.id)} hasSet={currentSet !== null} selectedSlot={selectedSlot} />
        ))}
      {!isCollapsed && sample.regions.length === 0 && sample.analysis_status === 'done' && (
        <p className="muted indent">no harmonically stable regions found</p>
      )}
    </div>
  );
}

function Folder({ folder }: { folder: LibraryFolder }) {
  const collapsedFolders = useStore((s) => s.ui.collapsedFolders);
  const setUi = useStore((s) => s.setUi);
  const collapsed = collapsedFolders[folder.key] === true;
  const open = !collapsed;
  const toggle = () => {
    const next = { ...collapsedFolders };
    if (collapsed) delete next[folder.key];
    else next[folder.key] = true;
    setUi({ collapsedFolders: next });
  };
  const analysed = folder.samples.filter((s) => s.analysis_status === 'done').length;
  return (
    <div className="lib-folder">
      {folder.folder !== '' && (
        <div className="lib-folder-head">
          <button type="button" className="disclosure" onClick={toggle} aria-label={open ? 'collapse folder' : 'expand folder'}>
            {open ? '▾' : '▸'}
          </button>
          <span title={folder.folder}>{folder.folder}</span>
          <span className="readout">
            {folder.samples.length} · {analysed} analysed
          </span>
        </div>
      )}
      {open && (
        <div className={folder.folder !== '' ? 'lib-folder-body' : ''}>
          {folder.samples.map((sample) => (
            <SampleBlock key={sample.id} sample={sample} />
          ))}
        </div>
      )}
    </div>
  );
}

export function LibraryPane() {
  const samples = useStore((s) => s.samples);
  const roots = useStore((s) => s.roots);
  const filter = useStore((s) => s.ui.libraryFilter);
  const collapsedFolders = useStore((s) => s.ui.collapsedFolders);
  const setUi = useStore((s) => s.setUi);
  const regionCount = regionsById(samples).size;
  const groups = groupLibrary(samples, roots, filter);
  const matching = groups.reduce((acc, g) => acc + g.count, 0);

  return (
    <section className="pane pane-library">
      <header className="pane-head">
        <h2>Library</h2>
        <span className="readout">
          {samples.length} samples · {regionCount} regions
          {filter.trim() ? ` · ${matching} match` : ''}
        </span>
        <label className="field lib-filter" title="filter by file name, folder, chord, quality or key">
          <input type="text" placeholder="filter: name, folder, Am, maj7, F minor" value={filter} onChange={(e) => setUi({ libraryFilter: e.target.value })} />
        </label>
        <span className="spacer" />
        <AddFiles />
      </header>
      <div className="pane-body">
        {samples.length === 0 && roots.length === 0 && (
          <p className="muted">No samples. Add audio files, or add a folder (your Splice library) under Preferences › Library; analysis runs in the background.</p>
        )}
        {groups.map((g) => {
          const key = g.root ? `root:${g.root.id}` : 'root:loose';
          const collapsed = collapsedFolders[key] === true;
          return (
            <div key={key} className="lib-group">
              <div className="lib-group-head">
                <button
                  type="button"
                  className="disclosure"
                  onClick={() => {
                    const next = { ...collapsedFolders };
                    if (collapsed) delete next[key];
                    else next[key] = true;
                    setUi({ collapsedFolders: next });
                  }}
                  aria-label={collapsed ? 'expand' : 'collapse'}
                >
                  {collapsed ? '▸' : '▾'}
                </button>
                <span title={g.root?.path ?? 'files added one by one'}>{g.title}</span>
                <span className="readout">
                  {g.count} · {g.analysed} analysed
                  {g.missing > 0 ? ` · ${g.missing} missing` : ''}
                </span>
                {g.root?.status === 'scanning' && <span className="badge badge-busy">scanning</span>}
                {g.root?.status === 'error' && (
                  <span className="badge badge-bad" title={g.root.error ?? ''}>
                    scan error
                  </span>
                )}
              </div>
              {!collapsed && g.folders.map((f) => <Folder key={f.key} folder={f} />)}
              {!collapsed && g.folders.length === 0 && (
                <p className="muted indent">{g.root?.status === 'scanning' ? 'scanning…' : 'no audio files found in this folder'}</p>
              )}
            </div>
          );
        })}
      </div>
    </section>
  );
}
