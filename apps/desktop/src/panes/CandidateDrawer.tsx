import { useEffect } from 'react';
import { api } from '../api/client';
import { STRICTNESS_ORDER, type Slot, type Strictness } from '../api/types';
import { engine } from '../audio/engine';
import { Knob } from '../components/Knob';
import { useStore } from '../store';
import { candidateKey } from '../util/candidates';
import { chordSymbol, transposeLabel } from '../util/chords';
import { errorMessage, num } from '../util/format';

const STRICTNESS_LABEL: Record<Strictness, string> = {
  exact: 'exact',
  function: 'same function',
  two_tones: 'shares two tones',
};

/** Ranked candidates for the selected slot (§5.2–5.3). */
/** Region length in beats: declared when the user set it, else the tracked beat count. */
function regionBeats(r: { declared_length_beats: number | null; beats: number[] }): number | null {
  if (r.declared_length_beats !== null && r.declared_length_beats !== undefined) return r.declared_length_beats;
  return r.beats.length > 0 ? r.beats.length : null;
}

export function CandidateDrawer({ slot }: { slot: Slot }) {
  const strictness = useStore((s) => s.ui.strictness);
  const uniqueSamples = useStore((s) => s.ui.uniqueSamples);
  const setUi = useStore((s) => s.setUi);
  const loadCandidates = useStore((s) => s.loadCandidates);
  const assignCandidate = useStore((s) => s.assignCandidate);
  const previewInContext = useStore((s) => s.previewInContext);
  const list = useStore((s) => s.candidates[candidateKey(slot.id, strictness, uniqueSamples)]);

  useEffect(() => {
    if (!list) void loadCandidates(slot.id, strictness);
  }, [slot.id, strictness, uniqueSamples, list, loadCandidates]);

  const solo = (regionId: string) =>
    void engine.auditionRegion(api.regions.audioPath(regionId)).catch((e: unknown) => useStore.setState({ lastError: errorMessage(e) }));

  return (
    <div className="drawer">
      <div className="row">
        <h3>
          Candidates for slot {slot.idx + 1} · {slot.chord_symbol}
        </h3>
        <span className="spacer" />
        <Knob
          label="strictness"
          value={STRICTNESS_ORDER.indexOf(strictness)}
          min={0}
          max={2}
          step={1}
          onChange={(v) => setUi({ strictness: STRICTNESS_ORDER[v] ?? 'function' })}
        />
        <span className="readout">{STRICTNESS_LABEL[strictness]}</span>
        <button type="button" onClick={() => void loadCandidates(slot.id, strictness)}>
          refresh
        </button>
      </div>
      {!list ? (
        <p className="muted">ranking…</p>
      ) : list.length === 0 ? (
        <p className="muted">{uniqueSamples ? 'no candidates from samples not already in use (distinct samples is on)' : 'no candidates at this strictness'}</p>
      ) : (
        <table className="candidates">
          <thead>
            <tr>
              <th>sample</th>
              <th>chord</th>
              <th title="region length: beats (declared or tracked) · seconds; red when shorter than the slot">len</th>
              <th>label</th>
              <th title="transposition + timbral distance + (1 − confidence)">score</th>
              <th>transp</th>
              <th>timbre</th>
              <th>conf</th>
              <th title="audition: solo plays the raw region; in ctx renders it conformed at this slot and plays the progression">hear</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {list.map((c) => {
              const r = c.region;
              const current = slot.region_id === r.id && slot.transpose_semitones === c.transpose_semitones;
              const used = c.used_by_slot_idx ?? [];
              return (
                <tr key={`${r.id}:${c.transpose_semitones}`} className={`${current ? 'current' : ''} ${used.length ? 'used' : ''}`.trim()}>
                  <td>
                    {c.sample_name}
                    {used.length > 0 && (
                      <span className="badge badge-muted" title="this sample already plays in another slot">
                        in slot {used.join(', ')}
                      </span>
                    )}
                  </td>
                  <td className="chord">{chordSymbol(r.chord_root, r.chord_quality, r.bass_note) ?? '?'}</td>
                  <td className={regionBeats(r) !== null && (regionBeats(r) as number) < slot.length_beats ? 'warn' : ''}>
                    {regionBeats(r) !== null ? `${num(regionBeats(r) as number, 1)} b · ` : ''}
                    {num(r.end_s - r.start_s, 2)} s
                  </td>
                  <td>
                    {c.label}
                    {c.transpose_semitones !== 0 && ` (${transposeLabel(c.transpose_semitones)})`}
                  </td>
                  <td>{num(c.score, 3)}</td>
                  <td>{num(c.transposition_cost, 2)}</td>
                  <td>{num(c.timbral_distance, 2)}</td>
                  <td>{num(c.confidence_cost, 2)}</td>
                  <td className="hear">
                    <button type="button" onClick={() => solo(r.id)} title="audition raw region solo">
                      solo
                    </button>
                    <button type="button" onClick={() => void previewInContext(r, slot.id)} title="audition conformed, in context at this slot">
                      in ctx
                    </button>
                  </td>
                  <td>
                    <button type="button" disabled={current} onClick={() => void assignCandidate(slot.id, c)}>
                      {current ? 'assigned' : 'assign'}
                    </button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
    </div>
  );
}
