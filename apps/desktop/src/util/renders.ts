import type { RenderStatus } from '../api/types';

/** The render URL the engine should hold for a slot. A render that is queued
 * or rendering carries the URL of the last good render (`stale`), and the
 * handoff (§2) says to keep playing that until the new one lands; a render
 * for a different region is never used. */
export function renderPathFor(render: RenderStatus | undefined, regionId: string | null): string | null {
  if (!render || !regionId || render.region_id !== regionId || !render.url) return null;
  return render.url;
}
