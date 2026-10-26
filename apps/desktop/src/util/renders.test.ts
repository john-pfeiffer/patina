import { describe, expect, it } from 'vitest';
import type { RenderStatus } from '../api/types';
import { renderPathFor } from './renders';

const base: RenderStatus = {
  cache_key: 'new',
  region_id: 'r1',
  set_id: 's',
  status: 'ready',
  url: '/renders/new.wav',
  error: null,
  deltas: null,
  stale: false,
};

describe('renderPathFor', () => {
  it('uses a ready render', () => {
    expect(renderPathFor(base, 'r1')).toBe('/renders/new.wav');
  });

  it('keeps playing the last good render while a newer key renders', () => {
    const pending: RenderStatus = { ...base, status: 'rendering', url: '/renders/old.wav', stale: true };
    expect(renderPathFor(pending, 'r1')).toBe('/renders/old.wav');
  });

  it('has nothing to play while the first render is queued', () => {
    expect(renderPathFor({ ...base, status: 'queued', url: null, stale: true }, 'r1')).toBeNull();
  });

  it('never plays a render for another region or an unassigned slot', () => {
    expect(renderPathFor(base, 'r2')).toBeNull();
    expect(renderPathFor(base, null)).toBeNull();
    expect(renderPathFor(undefined, 'r1')).toBeNull();
  });
});
