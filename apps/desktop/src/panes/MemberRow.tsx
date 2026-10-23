import type { ConformDeltas, RenderStatus, SetMember } from '../api/types';
import { RenderBadge } from '../components/RenderBadge';
import { Sparkline } from '../components/Sparkline';
import { useStore } from '../store';
import { chordSymbol } from '../util/chords';
import { db, hz, ratio, signed } from '../util/format';

interface MemberRowProps {
  member: SetMember;
  sampleName: string;
  isAnchor: boolean;
  deltas: ConformDeltas | undefined;
  render: RenderStatus | undefined;
}

function tiltSummary(tilt: readonly number[]): string {
  if (tilt.length === 0) return '—';
  const lo = Math.min(...tilt);
  const hi = Math.max(...tilt);
  return `${signed(lo, 1)}…${signed(hi, 1)} dB`;
}

/** One set member with its delta readouts (§6: how far it moved per module). */
export function MemberRow({ member, sampleName, isAnchor, deltas, render }: MemberRowProps) {
  const removeMember = useStore((s) => s.removeMember);
  const patchSet = useStore((s) => s.patchSet);
  const r = member.region;
  const symbol = chordSymbol(r.chord_root, r.chord_quality, r.bass_note) ?? '?';
  return (
    <div className={`member ${isAnchor ? 'member-anchor' : ''}`}>
      <div className="member-head">
        <span className="chord">{symbol}</span>
        <span className="sample-name" title={r.id}>
          {sampleName}
        </span>
        {isAnchor ? (
          <span className="badge badge-anchor" title="identity through the tonal layer">
            anchor
          </span>
        ) : (
          <button type="button" onClick={() => void patchSet({ anchor_region_id: r.id })} title="make this the anchor">
            set anchor
          </button>
        )}
        <RenderBadge status={render} />
        <span className="spacer" />
        <button type="button" onClick={() => void removeMember(r.id)} title="remove from set">
          ×
        </button>
      </div>
      {deltas ? (
        <div className="member-deltas">
          <span title="tuning correction">tune {signed(deltas.tuning_cents, 1)} ct</span>
          <span title="loudness gain">loud {db(deltas.loudness_db)}</span>
          <span className="spark" title={deltas.tilt_db.map((v) => db(v, 1)).join(' ')}>
            tilt {tiltSummary(deltas.tilt_db)} <Sparkline values={deltas.tilt_db} min={-6} max={6} />
          </span>
          <span title="low-end floor: high-pass / mono below">
            hpf {deltas.lowend_hpf_hz === null ? '—' : hz(deltas.lowend_hpf_hz)} · mono{' '}
            {deltas.lowend_mono_hz === null ? '—' : hz(deltas.lowend_mono_hz)}
          </span>
          <span title="noise floor bedded under the region">
            floor {deltas.noise_floor_db === null ? '—' : `${signed(deltas.noise_floor_db, 1)} dB`}
          </span>
          <span title="side scale per band">width {deltas.width_ratio.map((w) => ratio(w)).join(' ')}</span>
          <span title="average stretch ratio" className={Math.abs(deltas.stretch_ratio - 1) > 0.25 ? 'warn' : ''}>
            stretch {ratio(deltas.stretch_ratio, 3)}
          </span>
          {deltas.transpose_semitones !== 0 && <span>{signed(deltas.transpose_semitones, 0)} st</span>}
          {deltas.bypassed.length > 0 && <span className="muted">bypass: {deltas.bypassed.join(', ')}</span>}
        </div>
      ) : (
        <div className="member-deltas muted">deltas pending…</div>
      )}
    </div>
  );
}
