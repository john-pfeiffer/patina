import { describe, expect, it } from 'vitest';
import type { Candidate, Region } from '../api/types';
import { candidateKey, orderCandidates } from './candidates';

const region = (id: string): Region => ({
  id,
  sample_id: 's',
  position: 0,
  start_s: 0,
  end_s: 1,
  chord_root: 'C',
  chord_quality: 'maj',
  bass_note: null,
  chord_confidence: 1,
  key_context: null,
  beats: [],
  first_onset_s: null,
  declared_length_beats: null,
  busyness: null,
  register_centroid_hz: null,
  metrics: null,
  user_overrides: {
    keep_preroll: false,
    snap_s: null,
    nudge_ms: 0,
    warp_markers: [],
    warp_strength: null,
    stretch_method: null,
    allow_wobble: false,
    tempo_multiplier: 1,
    tempo_feel: 'quarter',
    bypass: {},
  },
  source: 'analysis',
});

const cand = (id: string, score: number, transpose: number): Candidate => ({
  region: region(id),
  sample_name: id,
  sample_id: 's',
  used_by_slot_idx: [],
  score,
  transposition_cost: transpose === 0 ? 0 : 1,
  timbral_distance: 0,
  confidence_cost: 0,
  transpose_semitones: transpose,
  label: transpose === 0 ? 'exact' : `${transpose} st`,
});

describe('orderCandidates', () => {
  it('puts substitutions before transpositions regardless of score', () => {
    const out = orderCandidates([cand('t', 0.1, 2), cand('s', 0.9, 0)]);
    expect(out.map((c) => c.region.id)).toEqual(['s', 't']);
  });
  it('sorts by score within each group and keeps ties stable', () => {
    const out = orderCandidates([cand('b', 0.5, 0), cand('a', 0.2, 0), cand('c', 0.5, 0), cand('y', 0.3, -1), cand('x', 0.1, 3)]);
    expect(out.map((c) => c.region.id)).toEqual(['a', 'b', 'c', 'x', 'y']);
  });
  it('does not mutate the input', () => {
    const input = [cand('b', 0.5, 0), cand('a', 0.2, 0)];
    orderCandidates(input);
    expect(input.map((c) => c.region.id)).toEqual(['b', 'a']);
  });
});

describe('candidateKey', () => {
  it('is distinct per strictness', () => {
    expect(candidateKey('slot', 'exact')).not.toBe(candidateKey('slot', 'function'));
    expect(candidateKey('slot', 'exact', true)).not.toBe(candidateKey('slot', 'exact'));
    expect(candidateKey('slot', 'exact', false)).toBe(candidateKey('slot', 'exact'));
  });
});
