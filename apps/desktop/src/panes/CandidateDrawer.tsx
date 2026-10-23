import { useEffect } from 'react';
import { STRICTNESS_ORDER, type Slot, type Strictness } from '../api/types';
import { Knob } from '../components/Knob';
import { useStore } from '../store';
import { candidateKey } from '../util/candidates';
import { chordSymbol, transposeLabel } from '../util/chords';
import { num } from '../util/format';

const STRICTNESS_LABEL: Record<Strictness, string> = {
  exact: 'exact',
  function: 'same function',
  two_tones: 'shares two tones',
};

/** Ranked candidates for the selected slot (§5.2–5.3). */
export function CandidateDrawer({ slot }: { slot: Slot }) {
  const strictness = useStore((s) => s.ui.strictness);
  const setUi = useStore((s) => s.setUi);
  const loadCandidates = useStore((s) => s.loadCandidates);
  const assignCandidate = useStore((s) => s.assignCandidate);
  const list = useStore((s) => s.candidates[candidateKey(slot.id, strictness)]);

  useEffect(() => {
    if (!list) void loadCandidates(slot.id, strictness);
  }, [slot.id, strictness, list, loadCandidates]);

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
        <p className="muted">no candidates at this strictness</p>
      ) : (
        <table className="candidates">
          <thead>
            <tr>
              <th>sample</th>
              <th>chord</th>
              <th>label</th>
              <th title="transposition + timbral distance + (1 − confidence)">score</th>
              <th>transp</th>
              <th>timbre</th>
              <th>conf</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {list.map((c) => {
              const r = c.region;
              const current = slot.region_id === r.id && slot.transpose_semitones === c.transpose_semitones;
              return (
                <tr key={`${r.id}:${c.transpose_semitones}`} className={current ? 'current' : ''}>
                  <td>{c.sample_name}</td>
                  <td className="chord">{chordSymbol(r.chord_root, r.chord_quality, r.bass_note) ?? '?'}</td>
                  <td>
                    {c.label}
                    {c.transpose_semitones !== 0 && ` (${transposeLabel(c.transpose_semitones)})`}
                  </td>
                  <td>{num(c.score, 3)}</td>
                  <td>{num(c.transposition_cost, 2)}</td>
                  <td>{num(c.timbral_distance, 2)}</td>
                  <td>{num(c.confidence_cost, 2)}</td>
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
