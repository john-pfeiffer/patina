import type { Division, Pattern, PatternMode, Slot, Step } from '../api/types';
import { DEFAULT_STEP, resizeSteps, stepCount } from '../audio/schedule';
import { NumberField } from '../components/NumberField';
import { useStore } from '../store';

const DIVISIONS: readonly Division[] = ['1/4', '1/8', '1/16'];

/** Per-slot pattern (§8.2): hold, or retrigger on a grid with per-step
 * gate / velocity / probability / beat offset. Same step model as Refonte. */
export function PatternEditor({ slot }: { slot: Slot }) {
  const patchSlot = useStore((s) => s.patchSlot);
  const p = slot.pattern;
  const commit = (pattern: Pattern) => void patchSlot(slot.id, { pattern });
  const n = stepCount(slot.length_beats, p.division);
  const steps = p.mode === 'retrigger' ? resizeSteps(p.steps, n) : [];
  const setStep = (i: number, patch: Partial<Step>) => {
    const next = steps.slice();
    next[i] = { ...(next[i] ?? DEFAULT_STEP), ...patch };
    commit({ ...p, steps: next });
  };

  return (
    <div className="pattern">
      <div className="row">
        <label className="field">
          <span className="field-label">pattern</span>
          <select value={p.mode} onChange={(e) => commit({ ...p, mode: e.target.value as PatternMode })}>
            <option value="hold">hold</option>
            <option value="retrigger">retrigger</option>
          </select>
        </label>
        {p.mode === 'retrigger' && (
          <label className="field">
            <span className="field-label">division</span>
            <select value={p.division} onChange={(e) => commit({ ...p, division: e.target.value as Division })}>
              {DIVISIONS.map((d) => (
                <option key={d} value={d}>
                  {d}
                </option>
              ))}
            </select>
          </label>
        )}
        {p.mode === 'retrigger' && <span className="readout">{n} steps</span>}
        {p.mode === 'retrigger' && p.steps.length > 0 && (
          <button type="button" onClick={() => commit({ ...p, steps: [] })} title="reset every step to the default">
            reset
          </button>
        )}
      </div>
      {p.mode === 'retrigger' && n > 0 && (
        <table className="steps">
          <thead>
            <tr>
              <th />
              {steps.map((_, i) => (
                <th key={i}>{i + 1}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            <tr>
              <th>on</th>
              {steps.map((s, i) => (
                <td key={i}>
                  <input type="checkbox" checked={s.active} onChange={(e) => setStep(i, { active: e.target.checked })} />
                </td>
              ))}
            </tr>
            <tr>
              <th title="fraction of the division the hit stays open">gate</th>
              {steps.map((s, i) => (
                <td key={i}>
                  <NumberField value={s.gate} min={0} max={1} step={0.05} width={44} onCommit={(v) => setStep(i, { gate: v ?? 1 })} />
                </td>
              ))}
            </tr>
            <tr>
              <th>vel</th>
              {steps.map((s, i) => (
                <td key={i}>
                  <NumberField value={s.velocity} min={0} max={1} step={0.05} width={44} onCommit={(v) => setStep(i, { velocity: v ?? 1 })} />
                </td>
              ))}
            </tr>
            <tr>
              <th>prob</th>
              {steps.map((s, i) => (
                <td key={i}>
                  <NumberField
                    value={s.probability}
                    min={0}
                    max={1}
                    step={0.05}
                    width={44}
                    onCommit={(v) => setStep(i, { probability: v ?? 1 })}
                  />
                </td>
              ))}
            </tr>
            <tr>
              <th title="beat offset into the region">off</th>
              {steps.map((s, i) => (
                <td key={i}>
                  <NumberField value={s.offset_beats} min={0} step={0.25} width={44} onCommit={(v) => setStep(i, { offset_beats: v ?? 0 })} />
                </td>
              ))}
            </tr>
          </tbody>
        </table>
      )}
    </div>
  );
}
