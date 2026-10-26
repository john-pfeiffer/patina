import type { Region, RenderStatus, Slot } from '../api/types';
import { NumberField } from '../components/NumberField';
import { RenderBadge } from '../components/RenderBadge';
import { useStore } from '../store';
import { chordSymbol, transposeLabel } from '../util/chords';
import { PatternEditor } from './PatternEditor';

interface SlotRowProps {
  slot: Slot;
  region: Region | null;
  sampleName: string | null;
  render: RenderStatus | undefined;
  lastGood: RenderStatus | undefined;
  selected: boolean;
  playing: boolean;
}

export function SlotRow({ slot, region, sampleName, render, lastGood, selected, playing }: SlotRowProps) {
  const patchSlot = useStore((s) => s.patchSlot);
  const setUi = useStore((s) => s.setUi);
  const openEditor = useStore((s) => s.openEditor);
  const regionSymbol = region ? (chordSymbol(region.chord_root, region.chord_quality, region.bass_note) ?? '?') : null;
  const select = () => setUi({ selectedSlotId: selected ? null : slot.id });
  return (
    <div className={`slot ${selected ? 'slot-selected' : ''} ${playing ? 'slot-playing' : ''}`}>
      <div className="slot-head">
        <button type="button" className={`slot-pick ${selected ? 'active' : ''}`} onClick={select} title="select slot (candidates, assign, in-context audition)">
          {slot.idx + 1}
        </button>
        <span className="chord">{slot.chord_symbol}</span>
        <span className="readout roman">{slot.roman ?? ''}</span>
        <NumberField
          value={slot.length_beats}
          min={0.25}
          step={0.25}
          unit="b"
          width={48}
          onCommit={(v) => v !== null && void patchSlot(slot.id, { length_beats: v })}
        />
        <span className="slot-region" title={region?.id ?? 'unassigned'}>
          {region ? (
            <>
              {sampleName ?? region.sample_id} · {regionSymbol}
              {slot.transpose_semitones !== 0 && <span className="badge badge-stale"> {transposeLabel(slot.transpose_semitones)}</span>}
            </>
          ) : (
            <span className="muted">unassigned</span>
          )}
        </span>
        <RenderBadge status={slot.region_id ? render : undefined} previous={lastGood} empty="—" />
        <button
          type="button"
          className={selected ? 'active' : ''}
          onClick={select}
          title={selected ? 'close the candidate drawer' : 'open the candidate drawer for this slot'}
        >
          {selected ? 'close' : region ? 'change…' : 'candidates…'}
        </button>
        <button type="button" disabled={!region} onClick={() => region && openEditor(region.id, slot.id)} title="edit this slot's region: trim, move the transient, audition in context">
          edit
        </button>
        <span className="spacer" />
        <label className="check" title="locked slots are untouched by Reroll">
          <input type="checkbox" checked={slot.locked} onChange={(e) => void patchSlot(slot.id, { locked: e.target.checked })} />
          lock
        </label>
        <button type="button" disabled={!slot.region_id} onClick={() => void patchSlot(slot.id, { clear_region: true })} title="clear assignment">
          clear
        </button>
      </div>
      <PatternEditor slot={slot} />
    </div>
  );
}
