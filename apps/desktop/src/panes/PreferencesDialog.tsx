import { useEffect, useState } from 'react';
import { canSelectOutput, engine, listOutputDevices, selectOutput, type OutputDevice } from '../audio/engine';
import { Dialog } from '../components/Dialog';
import { useStore, type PrefsTab } from '../store';
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
          {tab === 'library' && <Placeholder what="Library" />}
          {tab === 'analysis' && <Placeholder what="Analysis" />}
          {tab === 'appearance' && <Placeholder what="Appearance" />}
        </div>
      </div>
    </Dialog>
  );
}
