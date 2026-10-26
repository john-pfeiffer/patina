import { useEffect, useState } from 'react';
import { api } from '../api/client';
import type { ParseResponse } from '../api/types';
import { slotStarts } from '../audio/schedule';
import { Knob } from '../components/Knob';
import { TextField } from '../components/TextField';
import { regionsById, samplesById, selectCurrentProgression, selectCurrentSet, selectSelectedSlot, useStore } from '../store';
import { errorMessage } from '../util/format';
import { CandidateDrawer } from './CandidateDrawer';
import { SlotRow } from './SlotRow';

const PARSE_DEBOUNCE_MS = 250;

function ProgressionPicker() {
  const progressions = useStore((s) => s.progressions);
  const currentId = useStore((s) => s.currentProgressionId);
  const hasSet = useStore((s) => s.currentSetId !== null);
  const selectProgression = useStore((s) => s.selectProgression);
  const createProgression = useStore((s) => s.createProgression);
  const removeProgression = useStore((s) => s.removeProgression);
  const [name, setName] = useState('');
  const create = () => void createProgression({ name: name.trim() || 'Progression' }).then(() => setName(''));
  return (
    <div className="row">
      <select value={currentId ?? ''} onChange={(e) => void selectProgression(e.target.value || null)} disabled={!hasSet}>
        {progressions.length === 0 && <option value="">(none)</option>}
        {progressions.map((p) => (
          <option key={p.id} value={p.id}>
            {p.name}
          </option>
        ))}
      </select>
      <input
        type="text"
        placeholder="new progression"
        value={name}
        style={{ width: 110 }}
        disabled={!hasSet}
        onChange={(e) => setName(e.target.value)}
        onKeyDown={(e) => e.key === 'Enter' && create()}
      />
      <button type="button" disabled={!hasSet} onClick={create}>
        new
      </button>
      <button type="button" disabled={!currentId} onClick={() => currentId && void removeProgression(currentId)} title="delete progression">
        ×
      </button>
    </div>
  );
}

/** Chord text entry with live parse feedback from the sidecar (never parsed locally). */
function ChordEntry({ progressionKey, onApply }: { progressionKey: string | null; onApply: (text: string) => void }) {
  const [text, setText] = useState('');
  const [parsed, setParsed] = useState<ParseResponse | null>(null);
  useEffect(() => {
    if (!text.trim()) {
      setParsed(null);
      return;
    }
    const t = setTimeout(() => {
      api.theory
        .parse({ text, ...(progressionKey ? { key: progressionKey } : {}) })
        .then(setParsed)
        .catch((e: unknown) => setParsed({ ok: false, key: null, chords: [], error: errorMessage(e) }));
    }, PARSE_DEBOUNCE_MS);
    return () => clearTimeout(t);
  }, [text, progressionKey]);
  const ok = parsed?.ok === true && parsed.chords.length > 0;
  return (
    <div className="chord-entry">
      <div className="row">
        <input
          type="text"
          className="grow"
          placeholder="Cm Fm G   or   i iv V   (Fm:2 sets beats)"
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && ok) {
              onApply(text);
              setText('');
            }
          }}
        />
        <button
          type="button"
          disabled={!ok}
          onClick={() => {
            onApply(text);
            setText('');
          }}
          title="replace slots (assignments survive where the chord is unchanged)"
        >
          apply
        </button>
      </div>
      {parsed && (
        <div className={`parse ${parsed.ok ? 'parse-ok' : 'parse-bad'}`}>
          {parsed.ok ? (
            <>
              {parsed.key && <span className="readout">key {parsed.key}</span>}
              {parsed.chords.map((c, i) => (
                <span key={i} className="chip" title={`pcs ${c.pitch_classes.join(' ')}`}>
                  {c.chord_symbol}
                  {c.roman ? ` (${c.roman})` : ''} · {c.length_beats}b
                </span>
              ))}
            </>
          ) : (
            <span>{parsed.error ?? 'could not parse'}</span>
          )}
        </div>
      )}
    </div>
  );
}

function CsvImport() {
  const importCsv = useStore((s) => s.importCsv);
  const hasSet = useStore((s) => s.currentSetId !== null);
  const [open, setOpen] = useState(false);
  const [csv, setCsv] = useState('');
  return (
    <div className="block">
      <button type="button" className={open ? 'active' : ''} onClick={() => setOpen((v) => !v)} disabled={!hasSet}>
        import CSV…
      </button>
      {open && (
        <div className="row-col">
          <textarea
            rows={4}
            placeholder={'chords,roman,key,bpm\n"Cm Fm G","i iv V",Cm,92'}
            value={csv}
            onChange={(e) => setCsv(e.target.value)}
          />
          <button type="button" disabled={!csv.trim()} onClick={() => void importCsv(csv).then(() => setCsv(''))}>
            import
          </button>
        </div>
      )}
    </div>
  );
}

export function ProgressionPane() {
  const progression = useStore(selectCurrentProgression);
  const set = useStore(selectCurrentSet);
  const samples = useStore((s) => s.samples);
  const slotRenders = useStore((s) => s.slotRenders);
  const slotLastGood = useStore((s) => s.slotLastGood);
  const selectedSlot = useStore(selectSelectedSlot);
  const rerollAmount = useStore((s) => s.ui.rerollAmount);
  const strictness = useStore((s) => s.ui.strictness);
  const uniqueSamples = useStore((s) => s.ui.uniqueSamples);
  const setUi = useStore((s) => s.setUi);
  const patchProgression = useStore((s) => s.patchProgression);
  const createProgression = useStore((s) => s.createProgression);
  const reroll = useStore((s) => s.reroll);
  const playing = useStore((s) => s.transport.playing);
  const positionBeats = useStore((s) => s.transport.positionBeats);
  /** Key typed before any progression exists; used when the first one is created. */
  const [draftKey, setDraftKey] = useState('');

  const regions = regionsById(samples);
  if (set) for (const m of set.members) if (!regions.has(m.region.id)) regions.set(m.region.id, m.region);
  const names = samplesById(samples);
  const starts = progression ? slotStarts(progression.slots) : [];
  const playingIdx = progression && playing ? starts.findIndex((s, i) => positionBeats >= s && positionBeats < s + (progression.slots[i]?.length_beats ?? 0)) : -1;

  const key = progression ? (progression.key ?? '') : draftKey;
  const apply = (text: string) => {
    if (progression) void patchProgression({ text });
    else void createProgression({ name: 'Progression', text, ...(draftKey ? { key: draftKey } : {}) });
  };

  return (
    <section className="pane pane-progression">
      <header className="pane-head">
        <h2>Progression</h2>
        <ProgressionPicker />
      </header>
      <div className={`pane-body prog-body ${selectedSlot ? 'prog-split' : ''}`}>
        <div className="prog-main">
        {!set && <p className="muted">Select or create a set first.</p>}
        {set && (
          <div className="block">
            <div className="row">
              <TextField
                label="key"
                value={key}
                placeholder="Cm, F#, Bb"
                width={70}
                onCommit={(v) => {
                  if (!progression) setDraftKey(v);
                  else if (v.trim()) void patchProgression({ key: v.trim() });
                }}
              />
              <span className="readout">{progression ? `${progression.slots.length} slots` : 'no progression'}</span>
              <span className="spacer" />
              <Knob
                label="reroll amount"
                value={rerollAmount}
                min={0}
                max={100}
                step={5}
                unit="%"
                onChange={(v) => setUi({ rerollAmount: v })}
              />
              <label className="check" title="never put two slots on regions of the same source sample; also hides such candidates in the drawer">
                <input type="checkbox" checked={uniqueSamples} onChange={(e) => setUi({ uniqueSamples: e.target.checked })} />
                distinct samples
              </label>
              <button
                type="button"
                disabled={!progression || progression.slots.length === 0 || rerollAmount <= 0}
                onClick={() => void reroll()}
                title={
                  rerollAmount <= 0
                    ? 'turn Reroll Amount above 0 % first'
                    : `reassign ${rerollAmount}% of unlocked slots from their candidate pools at “${strictness}” strictness`
                }
              >
                reroll
              </button>
            </div>
            <ChordEntry progressionKey={key || null} onApply={apply} />
          </div>
        )}
        {progression && (
          <div className="block slots">
            {progression.slots.length === 0 && <p className="muted">Enter chords above.</p>}
            {progression.slots.map((slot, i) => {
              const region = slot.region_id ? (regions.get(slot.region_id) ?? null) : null;
              return (
                <SlotRow
                  key={slot.id}
                  slot={slot}
                  region={region}
                  sampleName={region ? (names.get(region.sample_id)?.name ?? null) : null}
                  render={slotRenders[slot.id]}
                  lastGood={slotLastGood[slot.id]}
                  selected={selectedSlot?.id === slot.id}
                  playing={i === playingIdx}
                />
              );
            })}
          </div>
        )}
        <CsvImport />
        </div>
        {selectedSlot && (
          <aside className="prog-side">
            <CandidateDrawer slot={selectedSlot} />
          </aside>
        )}
      </div>
    </section>
  );
}
