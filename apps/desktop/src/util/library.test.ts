import { describe, expect, it } from 'vitest';
import type { LibraryRoot, Region, Sample } from '../api/types';
import { groupLibrary, sampleMatches } from './library';

const region = (root: string, quality: string): Region => ({
  id: `${root}${quality}`,
  sample_id: 's',
  position: 0,
  start_s: 0,
  end_s: 1,
  chord_root: root,
  chord_quality: quality,
  bass_note: root,
  chord_confidence: 0.9,
  key_context: 'C major',
  beats: [],
  first_onset_s: 0,
  declared_length_beats: null,
  busyness: 0,
  register_centroid_hz: 200,
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

const sample = (id: string, name: string, rootId: string | null, relPath: string | null, extra: Partial<Sample> = {}): Sample => ({
  id,
  path: `/x/${name}`,
  hash: 'h',
  name,
  root_id: rootId,
  rel_path: relPath,
  mtime: 1,
  size: 1,
  duration: 1,
  sample_rate: 44100,
  channels: 2,
  added_at: 'now',
  tuning_cents: 0,
  detected_bpm: 120,
  bpm_confidence: 1,
  beats: [],
  downbeats: [],
  onsets: [],
  key_estimate: 'A minor',
  key_confidence: 1,
  metrics: null,
  analysis_status: 'done',
  analysis_error: null,
  analysis_version: 2,
  analysis_progress: 1,
  regions: [region('A', 'min')],
  ...extra,
});

const roots: LibraryRoot[] = [
  { id: 'r1', path: '/splice', name: 'Splice', added_at: 'now', last_scan_at: null, status: 'idle', file_count: 3, error: null },
  { id: 'r2', path: '/packs', name: 'Packs', added_at: 'now', last_scan_at: null, status: 'scanning', file_count: 0, error: null },
];

describe('groupLibrary', () => {
  it('groups by root then folder, loose files last, keeps empty roots', () => {
    const samples = [
      sample('a', 'kick.wav', 'r1', 'Drums/kick.wav'),
      sample('b', 'pad.wav', 'r1', 'Keys/Pads/pad.wav'),
      sample('c', 'top.wav', 'r1', 'top.wav', { analysis_status: 'missing' }),
      sample('d', 'loose.wav', null, null),
      sample('e', 'gone.wav', 'r-deleted', 'x/gone.wav'),
    ];
    const groups = groupLibrary(samples, roots);
    expect(groups.map((g) => g.title)).toEqual(['Splice', 'Packs', 'Added files']);
    expect(groups[0]!.folders.map((f) => f.folder)).toEqual(['', 'Drums', 'Keys/Pads']);
    expect(groups[0]!.count).toBe(3);
    expect(groups[0]!.missing).toBe(1);
    expect(groups[0]!.analysed).toBe(2);
    expect(groups[1]!.folders).toEqual([]);
    expect(groups[2]!.folders[0]!.samples.map((s) => s.id)).toEqual(['d', 'e']);
  });

  it('filters by name, folder, chord and key and drops empty roots while filtering', () => {
    const samples = [
      sample('a', 'kick.wav', 'r1', 'Drums/kick.wav', { regions: [] }),
      sample('b', 'pad.wav', 'r1', 'Keys/Pads/pad.wav', { regions: [region('F', 'maj'), region('G', '7')] }),
    ];
    expect(sampleMatches(samples[1]!, 'F')).toBe(true);
    expect(sampleMatches(samples[1]!, 'g7')).toBe(true);
    expect(sampleMatches(samples[1]!, 'pads')).toBe(true);
    expect(sampleMatches(samples[1]!, 'a minor')).toBe(true);
    expect(sampleMatches(samples[0]!, 'pad')).toBe(false);
    const groups = groupLibrary(samples, roots, 'pad');
    expect(groups.map((g) => g.title)).toEqual(['Splice']);
    expect(groups[0]!.folders.map((f) => f.folder)).toEqual(['Keys/Pads']);
  });
});
