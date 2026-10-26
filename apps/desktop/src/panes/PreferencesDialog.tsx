import { useEffect, useState } from 'react';
import { isTauri, pickAudioFolder } from '../api/shell';
import { canSelectOutput, engine, listOutputDevices, selectOutput, type OutputDevice } from '../audio/engine';
import { Dialog } from '../components/Dialog';
import { NumberField } from '../components/NumberField';
import { selectQueueCounts, useStore, type PrefsTab } from '../store';
import { errorMessage } from '../util/format';

const TABS: { id: PrefsTab; label: string }[] = [
  { id: 'audio', label: 'Audio' },
  { id: 'library', label: 'Library' },
  { id: 'analysis', label: 'Analysis' },
  { id: 'appearance', label: 'Appearance' },
];

const LOG_POLL_MS = 500;

function ms(v: number | null | undefined): string {
  return v === null || v === undefined ? '—' : `${(v * 1000).toFixed(1)} ms`;
}

/** Output device, context readouts, clock health and the three test buttons
 * that bisect "no sound": direct → bus → clock. */
export function AudioPrefs() {
  const transport = useStore((s) => s.transport);
  const sinkPref = useStore((s) => s.ui.audioSinkId);
  const setUi = useStore((s) => s.setUi);
  const [devices, setDevices] = useState<OutputDevice[] | null>(null);
  const [clicks, setClicks] = useState<string>('');
  const [busy, setBusy] = useState<string | null>(null);
  const [log, setLog] = useState<readonly string[]>(engine.getLog());
  const [copied, setCopied] = useState(false);
  const selectable = canSelectOutput();

  useEffect(() => {
    void listOutputDevices().then(setDevices);
    // The engine only emits while playing or loading; keep the readouts live here.
    const poll = () => {
      setLog([...engine.getLog()]);
      useStore.getState().setTransport(engine.snapshot());
    };
    poll();
    const t = setInterval(poll, LOG_POLL_MS);
    return () => clearInterval(t);
  }, []);

  const run = async (name: string, fn: () => Promise<void>) => {
    setBusy(name);
    try {
      await fn();
    } catch (e) {
      useStore.setState({ lastError: errorMessage(e) });
    } finally {
      setBusy(null);
    }
  };

  const clockText = transport.clockSource
    ? `${transport.clockSource}${transport.playing ? (transport.ticking ? ` · ticking (${transport.ticksPerSecond ?? 0}/s)` : ' · NOT TICKING') : ''}`
    : '—';
  const clockBad = transport.playing && transport.ticking === false;

  const diagnostics = () =>
    [
      `context ${transport.contextState ?? 'idle'} · ${transport.sampleRate ?? '—'} Hz · lookahead ${ms(transport.lookAhead)}`,
      `latency base ${ms(transport.baseLatency)} · output ${ms(transport.outputLatency)}`,
      `clock ${clockText}${transport.clockNote ? ` · ${transport.clockNote}` : ''}`,
      `buffers ${transport.loadedSlots ?? 0}/${transport.totalSlots ?? 0} · sink ${transport.sinkId || 'system default'}`,
      `devices ${devices?.map((d) => d.label).join(', ') || 'none reported'} · selectable ${selectable}`,
      `userAgent ${navigator.userAgent}`,
      '',
      ...engine.getLog(),
    ].join('\n');

  return (
    <div className="prefs-section">
      <h3>Output</h3>
      <div className="row">
        {selectable && devices && devices.length > 0 ? (
          <label className="field">
            <span className="field-label">device</span>
            <select
              value={sinkPref ?? 'default'}
              onChange={(e) => {
                const id = e.target.value;
                void run('sink', async () => {
                  await selectOutput(id);
                  setUi({ audioSinkId: id === 'default' ? null : id });
                });
              }}
            >
              <option value="default">system default</option>
              {devices.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.label}
                </option>
              ))}
            </select>
          </label>
        ) : (
          <span className="readout">
            system default output
            {devices && devices.length > 0 ? ` (${devices.map((d) => d.label).join(', ')})` : ''}
          </span>
        )}
        {!selectable && (
          <span className="muted">this webview routes audio to the system output; change the device in the OS sound settings</span>
        )}
      </div>

      <h3>Context</h3>
      <table className="readouts">
        <tbody>
          <tr>
            <th>state</th>
            <td className={transport.contextState === 'running' ? '' : 'warn'}>
              {transport.contextState ?? 'idle'}
              {transport.contextState !== 'running' ? ' (Play or a test button starts it)' : ''}
            </td>
          </tr>
          <tr>
            <th>sample rate</th>
            <td>{transport.sampleRate ? `${Math.round(transport.sampleRate)} Hz` : '—'}</td>
          </tr>
          <tr>
            <th>latency</th>
            <td>
              base {ms(transport.baseLatency)} · output {ms(transport.outputLatency)} · lookahead {ms(transport.lookAhead)}
            </td>
          </tr>
          <tr>
            <th>clock</th>
            <td className={clockBad ? 'warn' : ''}>
              {clockText}
              {transport.clockNote ? <div className="muted">{transport.clockNote}</div> : null}
            </td>
          </tr>
          <tr>
            <th>buffers</th>
            <td>
              {transport.loadedSlots ?? 0}/{transport.totalSlots ?? 0} slot renders decoded
            </td>
          </tr>
        </tbody>
      </table>

      <h3>Test</h3>
      <div className="row">
        <button type="button" disabled={busy !== null} onClick={() => void run('direct', () => engine.testToneDirect())} title="440 Hz straight to the output">
          tone → direct
        </button>
        <button type="button" disabled={busy !== null} onClick={() => void run('bus', () => engine.testToneBus())} title="660 Hz through the slot bus and density compressor">
          tone → bus
        </button>
        <button
          type="button"
          disabled={busy !== null}
          onClick={() =>
            void run('clock', async () => {
              setClicks('…');
              const n = await engine.testClockClicks();
              setClicks(`${n}/4 fired`);
            })
          }
          title="four clicks scheduled on the audio clock the transport uses"
        >
          clicks → clock
        </button>
        {clicks && <span className={clicks.startsWith('4/') ? 'readout' : 'warn'}>{clicks}</span>}
        <span className="muted">silence on “direct” is the device; on “bus” the graph; on “clock” the scheduler</span>
      </div>

      <h3>Log</h3>
      <div className="row">
        <button
          type="button"
          onClick={() =>
            void navigator.clipboard
              .writeText(diagnostics())
              .then(() => setCopied(true))
              .catch((e: unknown) => useStore.setState({ lastError: errorMessage(e) }))
          }
        >
          copy diagnostics
        </button>
        {copied && <span className="muted">copied</span>}
      </div>
      <pre className="log">{log.length ? log.slice(-20).join('\n') : 'no engine events yet'}</pre>
    </div>
  );
}

/** Scanned folders, their state, and the analysis queue with its pause. */
export function LibraryPrefs() {
  const roots = useStore((s) => s.roots);
  const samples = useStore((s) => s.samples);
  const settings = useStore((s) => s.settings);
  const jobs = useStore((s) => s.jobs);
  const addRoot = useStore((s) => s.addRoot);
  const scanRoot = useStore((s) => s.scanRoot);
  const scanAllRoots = useStore((s) => s.scanAllRoots);
  const removeRoot = useStore((s) => s.removeRoot);
  const patchSettings = useStore((s) => s.patchSettings);
  const [manualPath, setManualPath] = useState('');
  const counts = selectQueueCounts(jobs);
  const perRoot = new Map<string, { total: number; analysed: number; missing: number; busy: number }>();
  for (const s of samples) {
    if (!s.root_id) continue;
    const c = perRoot.get(s.root_id) ?? { total: 0, analysed: 0, missing: 0, busy: 0 };
    c.total++;
    if (s.analysis_status === 'done') c.analysed++;
    else if (s.analysis_status === 'missing') c.missing++;
    else if (s.analysis_status === 'pending' || s.analysis_status === 'running') c.busy++;
    perRoot.set(s.root_id, c);
  }
  const loose = samples.filter((s) => !s.root_id).length;

  const pick = () =>
    void pickAudioFolder()
      .then((dir) => (dir ? addRoot(dir) : undefined))
      .catch((e: unknown) => useStore.setState({ lastError: errorMessage(e) }));

  const remove = (id: string, name: string) => {
    const del = window.confirm(`Remove “${name}” from the library?\n\nOK removes the folder and its samples.\nCancel keeps everything.`);
    if (!del) return;
    const keep = window.confirm('Keep the samples found under it as loose files?\n\nOK keeps them, Cancel deletes them from the library.');
    void removeRoot(id, !keep);
  };

  return (
    <div className="prefs-section">
      <h3>Library folders</h3>
      <p className="muted">
        Each folder is scanned recursively; everything found is analysed in the background. Rescan picks up new, changed and
        removed files. Hidden files and folders are skipped. Nothing is ever deleted from disk.
      </p>
      <div className="row">
        {isTauri() ? (
          <button type="button" onClick={pick}>
            add folder…
          </button>
        ) : (
          <>
            <input type="text" placeholder="/abs/path/to/Splice" value={manualPath} style={{ width: 260 }} onChange={(e) => setManualPath(e.target.value)} />
            <button type="button" disabled={!manualPath.trim()} onClick={() => void addRoot(manualPath.trim()).then(() => setManualPath(''))}>
              add folder
            </button>
          </>
        )}
        <button type="button" disabled={roots.length === 0} onClick={() => void scanAllRoots()}>
          rescan all
        </button>
        {counts.scanActive > 0 && <span className="badge badge-busy">{counts.scanActive} scanning</span>}
      </div>
      {roots.length === 0 ? (
        <p className="muted">No folders yet. Add your Splice folder or any sample pack folder.</p>
      ) : (
        <table className="readouts roots">
          <thead>
            <tr>
              <th>folder</th>
              <th>files</th>
              <th>analysed</th>
              <th>state</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {roots.map((r) => {
              const c = perRoot.get(r.id) ?? { total: 0, analysed: 0, missing: 0, busy: 0 };
              return (
                <tr key={r.id}>
                  <td title={r.path}>
                    <strong>{r.name}</strong>
                    <div className="muted path">{r.path}</div>
                  </td>
                  <td>
                    {c.total}
                    {c.missing > 0 && <span className="warn"> · {c.missing} missing</span>}
                  </td>
                  <td>
                    {c.analysed}/{c.total}
                    {c.busy > 0 && <span className="muted"> · {c.busy} queued</span>}
                  </td>
                  <td>
                    {r.status === 'scanning' ? (
                      <span className="badge badge-busy">scanning</span>
                    ) : r.status === 'error' ? (
                      <span className="badge badge-bad" title={r.error ?? ''}>
                        error
                      </span>
                    ) : (
                      <span className="muted">{r.last_scan_at ? `scanned ${r.last_scan_at.slice(0, 16).replace('T', ' ')}` : 'not scanned'}</span>
                    )}
                  </td>
                  <td className="hear">
                    <button type="button" disabled={r.status === 'scanning'} onClick={() => void scanRoot(r.id)}>
                      rescan
                    </button>
                    <button type="button" onClick={() => remove(r.id, r.name)} title="remove this folder from the library">
                      remove…
                    </button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
      {loose > 0 && <p className="muted">{loose} file(s) were added one by one and belong to no folder.</p>}

      <h3>Analysis queue</h3>
      <div className="row">
        <span className="readout">
          {counts.analyzeRunning} running · {counts.analyzeQueued} queued
        </span>
        <label className="check" title="park analysis jobs; renders and exports keep running">
          <input
            type="checkbox"
            checked={settings?.analysis_paused ?? false}
            disabled={!settings}
            onChange={(e) => void patchSettings({ analysis_paused: e.target.checked })}
          />
          pause analysis
        </label>
        <label className="check" title="rescan every folder when the engine starts">
          <input
            type="checkbox"
            checked={settings?.scan_on_startup ?? true}
            disabled={!settings}
            onChange={(e) => void patchSettings({ scan_on_startup: e.target.checked })}
          />
          rescan on startup
        </label>
      </div>
    </div>
  );
}

/** Chord-detection options and a way to re-run everything. */
export function AnalysisPrefs() {
  const settings = useStore((s) => s.settings);
  const samples = useStore((s) => s.samples);
  const info = useStore((s) => s.connection.info);
  const patchSettings = useStore((s) => s.patchSettings);
  const reanalyzeAll = useStore((s) => s.reanalyzeAll);
  const stale = info ? samples.filter((s) => s.analysis_status === 'done' && (s.analysis_version ?? 0) < info.analysis_version).length : 0;
  if (!settings) return <p className="muted">loading settings…</p>;
  return (
    <div className="prefs-section">
      <h3>Chord detection</h3>
      <p className="muted">These apply to the next analysis of a file; use “re-analyse everything” to apply them to the library.</p>
      <div className="row-col">
        <label className="check" title="fold the chord chroma from the harmonic part of the spectrum, so drums and noise do not read as “no chord”">
          <input type="checkbox" checked={settings.analysis_hpss} onChange={(e) => void patchSettings({ analysis_hpss: e.target.checked })} />
          harmonic chroma (ignore drums and noise under the chords)
        </label>
        <label className="check" title="label the longest tonal span of a file that would otherwise come back with no chord, at low confidence">
          <input type="checkbox" checked={settings.analysis_fallback} onChange={(e) => void patchSettings({ analysis_fallback: e.target.checked })} />
          whole-file fallback (never drop a tonal sample)
        </label>
        <NumberField
          label="observation window (beats)"
          value={settings.analysis_window_beats}
          min={0}
          max={4}
          step={0.5}
          width={64}
          onCommit={(v) => void patchSettings({ analysis_window_beats: Math.max(0, Math.min(4, v ?? 1)) })}
        />
        <span className="muted">how long the chord estimator listens before deciding; 1 beat lets arpeggios and melodies read as their chord, 0 decides per frame</span>
      </div>
      <h3>Re-analysis</h3>
      <div className="row">
        <button type="button" disabled={samples.length === 0} onClick={() => void reanalyzeAll()}>
          re-analyse everything
        </button>
        <span className="muted">
          queues every sample again at low priority; regions keep their ids, so sets and slots survive
          {stale > 0 ? ` · ${stale} sample(s) still carry an older analysis` : ''}
        </span>
      </div>
    </div>
  );
}

function Placeholder({ what }: { what: string }) {
  return <p className="muted">{what} settings arrive in a later update.</p>;
}

export function PreferencesDialog() {
  const tab = useStore((s) => s.ui.prefsTab);
  const setUi = useStore((s) => s.setUi);
  const close = () => setUi({ dialog: null });
  return (
    <Dialog title="Preferences" onClose={close}>
      <div className="prefs">
        <nav className="prefs-tabs">
          {TABS.map((t) => (
            <button key={t.id} type="button" className={t.id === tab ? 'active' : ''} onClick={() => setUi({ prefsTab: t.id })}>
              {t.label}
            </button>
          ))}
        </nav>
        <div className="prefs-page">
          {tab === 'audio' && <AudioPrefs />}
          {tab === 'library' && <LibraryPrefs />}
          {tab === 'analysis' && <AnalysisPrefs />}
          {tab === 'appearance' && <Placeholder what="Appearance" />}
        </div>
      </div>
    </Dialog>
  );
}
