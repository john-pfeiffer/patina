// Server-Sent Events from GET /events. EventSource cannot set headers, so the
// token travels as `?token=` (the sidecar accepts that on /events).

import { tokenUrl } from './client';
import { SIDECAR_EVENT_NAMES, type SidecarEventMap, type SidecarEventName } from './types';

export type SseState = 'connecting' | 'open' | 'reconnecting' | 'closed';

export interface EventSubscriberHandlers {
  onEvent: <K extends SidecarEventName>(name: K, data: SidecarEventMap[K]) => void;
  onState: (state: SseState) => void;
}

export function subscribeEvents({ onEvent, onState }: EventSubscriberHandlers): () => void {
  const es = new EventSource(tokenUrl('/events'));
  onState('connecting');

  es.onopen = () => onState('open');
  // EventSource reconnects by itself; surface that so the store can resync.
  es.onerror = () => onState(es.readyState === EventSource.CLOSED ? 'closed' : 'reconnecting');

  for (const name of SIDECAR_EVENT_NAMES) {
    es.addEventListener(name, (ev: MessageEvent<string>) => {
      let data: unknown;
      try {
        data = JSON.parse(ev.data);
      } catch {
        return;
      }
      // JSON boundary: the sidecar's event payloads are the documented models.
      onEvent(name, data as SidecarEventMap[typeof name]);
    });
  }

  return () => {
    es.close();
    onState('closed');
  };
}
