import { useState } from 'react';
import { isTauri, pickDirectory } from '../api/shell';
import { EXPORT_KINDS, type Job } from '../api/types';
import { engine } from '../audio/engine';
import { formatPosition } from '../audio/schedule';
import { NumberField } from '../components/NumberField';
import { ProgressBar } from '../components/ProgressBar';
import { TextField } from '../components/TextField';
import { selectCurrentProgression, selectCurrentSet, selectQueueCounts, sessionBpm, useStore } from '../store';
import { errorMessage } from '../util/format';

const MAX_JOBS_SHOWN = 12;

function jobLabel(job: Job): string {
  const p = job.payload;
  if (job.kind === 'analyze' && typeof p.sample_id === 'string') return `analyse ${p.sample_id.slice(0, 8)}`;
  if (job.kind === 'render' && typeof p.cache_key === 'string') return `render ${p.cache_key.slice(0, 8)}`;
  if (job.kind === 'export' && typeof p.kind === 'string') return `export ${p.kind}`;
  if (job.kind === 'scan') return 'scan folder';
  return job.kind;
}

function JobRow({ job }: { job: Job }) {
  const cancelJob = useStore((s) => s.cancelJob);
  const active = job.status === 'queued' || job.status === 'running';
  const files = job.result && Array.isArray(job.result.files) ? (job.result.files as unknown[]).filter((f): f is string => typeof f === 'string') : [];
  const take = job.result && typeof job.result.take === 'number' ? job.result.take : null;
  return (
    <div className={`job job-${job.status}`} title={job.error ?? job.message ?? ''}>
      <span className="job-label">
        {jobLabel(job)}
        {take !== null ? <span className="muted"> · take {take}</span> : null}
      </span>
      {active ? <ProgressBar value={job.progress} label={`${job.status} ${Math.round(job.progress * 100)}%`} /> : <span className={`badge ${job.status === 'done' ? 'badge-ok' : job.status === 'failed' ? 'badge-bad' : 'badge-muted'}`}>{job.status}</span>}
      {job.message && active && <span className="muted">{job.message}</span>}
      {job.error && <span className="warn">{job.error}</span>}
      {files.length > 0 && (
        <span className="muted" title={files.join('\n')}>
          {files.length} file{files.length === 1 ? '' : 's'}
        </span>
      )}
      {active && (
        <button type="button" onClick={() => void cancelJob(job.id)}>
          cancel
        </button>
      )}
    </div>
  );
}

function ExportControls() {
  const outDir = useStore((s) => s.ui.outDir);
  const loops = useStore((s) => s.ui.exportLoops);
  const setUi = useStore((s) => s.setUi);
  const startExport = useStore((s) => s.startExport);
  const progression = useStore(selectCurrentProgression);
  const canExport = progression !== null && outDir !== '';
  return (
    <div className="row transport-export">
      <span className="field-label">export</span>
      {isTauri() ? (
        <button
          type="button"
          onClick={() =>
            void pickDirectory()
              .then((dir) => dir && setUi({ outDir: dir }))
              .catch((e: unknown) => useStore.setState({ lastError: errorMessage(e) }))
          }
        >
          out dir…
        </button>
      ) : null}
      <TextField value={outDir} placeholder="/abs/output/dir" width={220} onCommit={(v) => setUi({ outDir: v })} />
      <NumberField label="loops" value={loops} min={1} max={64} step={1} width={48} onCommit={(v) => setUi({ exportLoops: Math.max(1, Math.round(v ?? 1)) })} />
      {EXPORT_KINDS.map((kind) => (
        <button key={kind} type="button" disabled={!canExport} onClick={() => void startExport(kind)} title={`export ${kind} as a new take (never overwrites)`}>
          {kind}
        </button>
      ))}
      <span className="muted">each export is a new take</span>
    </div>
  );
}

export function TransportPane() {
  const transport = useStore((s) => s.transport);
  const set = useStore(selectCurrentSet);
  const progression = useStore(selectCurrentProgression);
  const connection = useStore((s) => s.connection);
  const jobs = useStore((s) => s.jobs);
  const patchProgression = useStore((s) => s.patchProgression);
  const setUi = useStore((s) => s.setUi);
  const [jobsOpen, setJobsOpen] = useState(false);
  const counts = selectQueueCounts(jobs);
  const jobList = Object.values(jobs)
    .sort((a, b) => (a.updated_at < b.updated_at ? 1 : -1))
    .slice(0, MAX_JOBS_SHOWN);
  const activeJobs = Object.values(jobs).filter((j) => j.status === 'queued' || j.status === 'running').length;
  const canPlay = progression !== null && progression.slots.length > 0;
  const assignedSlots = progression?.slots.filter((sl) => sl.region_id).length ?? 0;
  const clockStalled = transport.playing && transport.ticking === false;
  const engineTitle = connection.info
    ? `${connection.url}\nsidecar v${connection.info.version} · analysis v${connection.info.analysis_version} · render v${connection.info.render_version}${connection.info.ml_available ? ' · ml' : ''}\nevents ${connection.sse}`
    : (connection.url ?? '');

  return (
    <section className="pane pane-transport">
      <div className="transport-row">
        <button type="button" className="big" disabled={!canPlay || transport.playing} onClick={() => void engine.play().catch((e: unknown) => useStore.setState({ lastError: errorMessage(e) }))}>
          ▶ play
        </button>
        <button type="button" className="big" disabled={!transport.playing} onClick={() => engine.stop()}>
          ■ stop
        </button>
        <span className="position readout" title="bars.beats">
          {formatPosition(transport.positionBeats)}
        </span>
        <span className="readout" title={set?.bpm_pinned ? 'pinned' : 'follows anchor'}>
          {sessionBpm(set).toFixed(1)} bpm{set?.bpm_pinned ? ' (pinned)' : ''}
        </span>
        <label className="check">
          <input type="checkbox" checked={progression?.loop ?? true} disabled={!progression} onChange={(e) => void patchProgression({ loop: e.target.checked })} />
          loop
        </label>
        {canPlay && assignedSlots === 0 && (
          <span className="warn" title="Set members are conformed, but only regions assigned to slots play and export">
            no slot has a region yet: select a slot and pick a candidate, or press Reroll
          </span>
        )}
        <span className="spacer" />
        <span className={`badge ${connection.status === 'ready' ? 'badge-ok' : connection.status === 'error' ? 'badge-bad' : 'badge-busy'}`} title={engineTitle}>
          {connection.status === 'starting' ? 'starting engine…' : connection.status === 'connecting' ? 'connecting…' : `engine ${connection.status}`}
        </span>
        {connection.sse !== 'open' && connection.status === 'ready' && <span className="badge badge-busy">events {connection.sse}</span>}
        {connection.error && <span className="warn">{connection.error}</span>}
        <button
          type="button"
          className={`badge badge-button ${clockStalled ? 'badge-bad' : transport.contextState === 'running' ? 'badge-ok' : 'badge-busy'}`}
          title={`Web Audio context state · sample rate · slot buffers decoded · clock${transport.clockNote ? `\n${transport.clockNote}` : ''}\nclick for audio preferences`}
          onClick={() => setUi({ dialog: 'preferences', prefsTab: 'audio' })}
        >
          audio {transport.contextState ?? 'idle'}
          {transport.sampleRate ? ` · ${Math.round(transport.sampleRate / 1000)} kHz` : ''}
          {transport.totalSlots !== undefined ? ` · ${transport.loadedSlots ?? 0}/${transport.totalSlots} buffers` : ''}
          {transport.clockSource ? ` · clock ${transport.clockSource}${transport.playing ? (transport.ticking ? ' ✓' : ' STALLED') : ''}` : ''}
        </button>
        {counts.scanActive > 0 && (
          <span className="badge badge-busy" title="library folders being scanned">
            scanning {counts.scanActive}
          </span>
        )}
        {(counts.analyzeRunning + counts.analyzeQueued > 0 || counts.renderRunning + counts.renderQueued > 0 || counts.exportActive > 0) && (
          <span className="readout" title="analysis · render · export queues (running / queued)">
            {counts.analyzeRunning + counts.analyzeQueued > 0 ? `analyse ${counts.analyzeRunning}/${counts.analyzeQueued} ` : ''}
            {counts.renderRunning + counts.renderQueued > 0 ? `render ${counts.renderRunning}/${counts.renderQueued} ` : ''}
            {counts.exportActive > 0 ? `export ${counts.exportActive}` : ''}
          </span>
        )}
        <button type="button" className={jobsOpen ? 'active' : ''} onClick={() => setJobsOpen((v) => !v)} title="recent jobs">
          jobs{activeJobs > 0 ? ` (${activeJobs})` : ''}
        </button>
        <button type="button" onClick={() => setUi({ dialog: 'preferences' })} title="preferences (⌘,)">
          prefs…
        </button>
      </div>
      <ExportControls />
      {jobsOpen && (
        <div className="jobs">
          {jobList.length === 0 && <span className="muted">no jobs yet</span>}
          {jobList.map((j) => (
            <JobRow key={j.id} job={j} />
          ))}
        </div>
      )}
    </section>
  );
}
