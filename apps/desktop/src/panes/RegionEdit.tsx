import { REGION_MODULES, type Region, type StretchMethod, type TempoFeel, type UserOverrides } from '../api/types';
import { NumberField } from '../components/NumberField';
import { useStore } from '../store';

const FEELS: readonly TempoFeel[] = ['quarter', 'eighth', 'half'];

/** Per-region user state (§6 bypass, §7 temporal). Every change is one PATCH. */
export function RegionEdit({ region }: { region: Region }) {
  const patchRegion = useStore((s) => s.patchRegion);
  const o = region.user_overrides;
  const setOverrides = (patch: Partial<UserOverrides>) => void patchRegion(region.id, { user_overrides: { ...o, ...patch } });
  const nonUniform = o.warp_markers.length > 0 || region.beats.length > 1;

  return (
    <div className="region-edit">
      <div className="row">
        <NumberField
          label="declared length"
          value={region.declared_length_beats}
          min={0.25}
          step={0.25}
          unit="beats"
          width={64}
          allowEmpty
          placeholder="auto"
          onCommit={(v) =>
            void patchRegion(region.id, v === null ? { clear_declared_length: true } : { declared_length_beats: v })
          }
        />
        <span className="readout" title="tracker tempo helper (§7.1)">
          tempo ×{o.tempo_multiplier}
        </span>
        <button type="button" onClick={() => setOverrides({ tempo_multiplier: o.tempo_multiplier / 2 })}>
          ½×
        </button>
        <button type="button" onClick={() => setOverrides({ tempo_multiplier: o.tempo_multiplier * 2 })}>
          2×
        </button>
        <label className="field">
          <span className="field-label">feel</span>
          <select value={o.tempo_feel} onChange={(e) => setOverrides({ tempo_feel: e.target.value as TempoFeel })}>
            {FEELS.map((f) => (
              <option key={f} value={f}>
                {f}
              </option>
            ))}
          </select>
        </label>
      </div>
      <div className="row">
        <label className="check">
          <input type="checkbox" checked={o.keep_preroll} onChange={(e) => setOverrides({ keep_preroll: e.target.checked })} />
          keep pre-roll
        </label>
        <NumberField label="nudge" value={o.nudge_ms} step={1} unit="ms" width={56} onCommit={(v) => setOverrides({ nudge_ms: v ?? 0 })} />
        <NumberField
          label="warp"
          value={o.warp_strength}
          min={0}
          max={100}
          step={5}
          unit="%"
          width={52}
          allowEmpty
          placeholder="set"
          onCommit={(v) => setOverrides({ warp_strength: v })}
        />
        <label className="field">
          <span className="field-label">stretch</span>
          <select
            value={o.stretch_method ?? ''}
            onChange={(e) => setOverrides({ stretch_method: (e.target.value || null) as StretchMethod | null })}
          >
            <option value="">set default</option>
            <option value="elastic">elastic</option>
            <option value="repitch">repitch</option>
          </select>
        </label>
        <label className="check" title="repitch under a non-uniform warp produces beat-to-beat pitch wobble; off forces elastic (§7.5)">
          <input type="checkbox" checked={o.allow_wobble} onChange={(e) => setOverrides({ allow_wobble: e.target.checked })} />
          allow pitch wobble{nonUniform ? '' : ' (uniform warp)'}
        </label>
      </div>
      <div className="row">
        <span className="field-label">bypass</span>
        {REGION_MODULES.map((name) => (
          <label key={name} className="check">
            <input
              type="checkbox"
              checked={o.bypass[name] === true}
              onChange={(e) => setOverrides({ bypass: { ...o.bypass, [name]: e.target.checked } })}
            />
            {name}
          </label>
        ))}
      </div>
    </div>
  );
}
