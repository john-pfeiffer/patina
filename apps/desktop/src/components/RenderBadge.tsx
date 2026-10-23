import type { RenderStatus } from '../api/types';

interface RenderBadgeProps {
  status: RenderStatus | undefined;
  /** Shown when the slot/member has nothing to render. */
  empty?: string;
}

/** Render state readout. A stale render is marked, never silent (§9). */
export function RenderBadge({ status, empty = 'no render' }: RenderBadgeProps) {
  if (!status) return <span className="badge badge-muted">{empty}</span>;
  const cls =
    status.status === 'ready' ? (status.stale ? 'badge-stale' : 'badge-ok') : status.status === 'failed' ? 'badge-bad' : 'badge-busy';
  const text = status.status === 'ready' && status.stale ? 'ready · STALE' : status.status === 'ready' ? 'ready' : status.status;
  return (
    <span className={`badge ${cls}`} title={status.error ?? status.cache_key}>
      {text}
      {status.stale && status.status !== 'ready' ? ' · stale' : ''}
    </span>
  );
}
