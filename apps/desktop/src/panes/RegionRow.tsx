import { useState } from 'react';
import { api } from '../api/client';
import { TILT_BAND_CENTRES_HZ, type Region, type Slot } from '../api/types';
import { engine } from '../audio/engine';
import { Sparkline } from '../components/Sparkline';
import { useStore } from '../store';
import { chordSymbol } from '../util/chords';
import { db, hz, num, pct, secs } from '../util/format';
import { RegionEdit } from './RegionEdit';

interface RegionRowProps {
  region: Region;
  isMember: boolean;
  hasSet: boolean;
  selectedSlot: Slot | null;
}

export function RegionRow({ region, isMember, hasSet, selectedSlot }: RegionRowProps) {
  const [editing, setEditing] = useState(false);
  const addMember = useStore((s) => s.addMember);
  const assignRegionToSlot = useStore((s) => s.assignRegionToSlot);
  const previewInContext = useStore((s) => s.previewInContext);
  const pending = useStore((s) => s.pendingAudition);
  const m = region.metrics;
  const symbol = chordSymbol(region.chord_root, region.chord_quality, region.bass_note) ?? '?';
  const beats = region.declared_length_beats ?? region.beats.length;

  return (
    <div className={`region ${isMember ? 'region-member' : ''}`}>
      <div className="region-head">
        <span className="chord" title={`confidence ${pct(region.chord_confidence)}`}>
          {symbol}
        </span>
        <span className="readout" title="bass note">
          bass {region.bass_note ?? '—'}
        </span>
        <span className="readout" title="key context">
          key {region.key_context ?? '—'}
        </span>
        <span className="readout" title={region.declared_length_beats !== null ? 'declared length' : 'tracked beats'}>
          {region.declared_length_beats !== null ? `${beats} b (decl)` : `${beats} b`}
        </span>
        <span className="readout" title="region bounds">
          {secs(region.start_s)}–{secs(region.end_s)}
        </span>
        {region.source === 'user' && <span className="badge badge-muted">user</span>}
        {isMember && <span className="badge badge-ok">in set</span>}
        <span className="spacer" />
        <button type="button" onClick={() => void engine.auditionRegion(api.regions.audioPath(region.id))} title="audition raw region solo">
          solo
        </button>
        <button
          type="button"
          disabled={!selectedSlot || !hasSet}
          title={selectedSlot ? `conform and audition at slot ${selectedSlot.idx + 1} (${selectedSlot.chord_symbol})` : 'select a slot first'}
          onClick={() => selectedSlot && void previewInContext(region, selectedSlot.id)}
        >
          {pending && pending.slotId === selectedSlot?.id ? 'ctx…' : 'in ctx'}
        </button>
        <button type="button" disabled={!hasSet || isMember} onClick={() => void addMember(region.id)} title="add to current set">
          + set
        </button>
        <button
          type="button"
          disabled={!selectedSlot}
          onClick={() => selectedSlot && void assignRegionToSlot(selectedSlot.id, region)}
          title={selectedSlot ? `assign to slot ${selectedSlot.idx + 1}` : 'select a slot first'}
        >
          assign
        </button>
        <button type="button" className={editing ? 'active' : ''} onClick={() => setEditing((v) => !v)}>
          edit
        </button>
      </div>
      <div className="region-metrics">
        <span title="busyness (melodic activity)">busy {num(region.busyness, 2)}</span>
        <span title="register centroid">reg {hz(region.register_centroid_hz)}</span>
        <span title="integrated loudness">{num(m?.loudness_lufs)} LUFS</span>
        <span title="crest factor">crest {num(m?.crest_db)} dB</span>
        <span title="noise floor">floor {num(m?.noise_floor_db)} dB</span>
        <span title="side/mid per band (<200, 200–3k, >3k)">
          width {m ? m.width_bands.map((w) => w.toFixed(2)).join(' ') : '—'}
        </span>
        <span title="wetness (late-energy ratio)">wet {num(m?.wetness, 2)}</span>
        <span title="low rolloff / bass mono">
          lo {hz(m?.low_rolloff_hz)} / mono {hz(m?.bass_mono_hz)}
        </span>
        <span title="sub ratio">sub {pct(m?.sub_ratio)}</span>
        <span className="spark" title="tilt: 10 octave bands 31.5 Hz–16 kHz (dB)">
          tilt{' '}
          {m ? (
            <Sparkline values={m.tilt_bands} title={m.tilt_bands.map((v, i) => `${TILT_BAND_CENTRES_HZ[i] ?? '?'}:${db(v, 0)}`).join(' ')} />
          ) : (
            '—'
          )}
        </span>
      </div>
      {editing && <RegionEdit region={region} />}
    </div>
  );
}
