// Sidecar discovery (docs/ARCHITECTURE.md "Process model"). Under Tauri the
// shell spawns the sidecar and hands us {url, token} once it has printed its
// PATINE_READY line; we poll `sidecar_info` every 500 ms and also listen for
// the `sidecar://ready` event. In a plain browser we use the --dev defaults.

import type { SidecarConnection } from './client';
import { isTauri, onSidecarReady, sidecarInfo } from './shell';

export const DEV_CONNECTION: SidecarConnection = { url: 'http://127.0.0.1:47621', token: 'dev' };
export const POLL_INTERVAL_MS = 500;

export interface DiscoveryHandlers {
  onFound: (conn: SidecarConnection) => void;
  onError?: (message: string) => void;
}

/** Starts discovery; returns a function that stops it. `onFound` fires once. */
export function discoverSidecar({ onFound, onError }: DiscoveryHandlers): () => void {
  if (!isTauri()) {
    queueMicrotask(() => onFound(DEV_CONNECTION));
    return () => undefined;
  }

  let done = false;
  let timer: ReturnType<typeof setTimeout> | null = null;
  let unlisten: (() => void) | null = null;

  const found = (conn: SidecarConnection) => {
    if (done) return;
    done = true;
    if (timer) clearTimeout(timer);
    if (unlisten) unlisten();
    onFound(conn);
  };

  const poll = async () => {
    if (done) return;
    try {
      const info = await sidecarInfo();
      if (info) {
        found(info);
        return;
      }
    } catch (e) {
      onError?.(e instanceof Error ? e.message : String(e));
    }
    if (!done) timer = setTimeout(poll, POLL_INTERVAL_MS);
  };

  onSidecarReady(found)
    .then((fn) => {
      if (done) fn();
      else unlisten = fn;
    })
    .catch((e: unknown) => onError?.(e instanceof Error ? e.message : String(e)));
  void poll();

  return () => {
    done = true;
    if (timer) clearTimeout(timer);
    if (unlisten) unlisten();
  };
}
