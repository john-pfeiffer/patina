// Wires discovery → API connection → initial load → SSE → playback engine.
// Called once from main.tsx.

import { api, setConnection, type SidecarConnection } from './api/client';
import { subscribeEvents, type SseState } from './api/events';
import { discoverSidecar } from './api/sidecar';
import { engine, type EngineProgram } from './audio/engine';
import {
  regionsById,
  selectAnchorRegion,
  selectCurrentProgression,
  selectCurrentSet,
  sessionBpm,
  useStore,
  type PatineState,
} from './store';
import { errorMessage } from './util/format';

const HEALTH_RETRY_MS = 1000;

export function buildProgram(s: PatineState): EngineProgram | null {
  const set = selectCurrentSet(s);
  const progression = selectCurrentProgression(s);
  if (!set || !progression) return null;
  const regions = regionsById(s.samples);
  for (const m of set.members) if (!regions.has(m.region.id)) regions.set(m.region.id, m.region);
  const anchor = selectAnchorRegion(s);
  return {
    bpm: sessionBpm(set),
    loop: progression.loop,
    slots: progression.slots.map((slot) => {
      const render = s.slotRenders[slot.id];
      return {
        id: slot.id,
        regionId: slot.region_id,
        length_beats: slot.length_beats,
        pattern: slot.pattern,
        wetness: slot.region_id ? (regions.get(slot.region_id)?.metrics?.wetness ?? null) : null,
        renderPath: render && render.status === 'ready' && render.url && render.region_id === slot.region_id ? render.url : null,
      };
    }),
    density: { enabled: set.tonal_settings.density.enabled, referenceLufs: anchor?.metrics?.loudness_lufs ?? null },
    space: {
      enabled: set.tonal_settings.space.enabled,
      amount: set.tonal_settings.space.amount ?? 100,
      anchorWetness: anchor?.metrics?.wetness ?? null,
    },
  };
}

function programInputsChanged(a: PatineState, b: PatineState): boolean {
  return (
    a.progressions !== b.progressions ||
    a.currentProgressionId !== b.currentProgressionId ||
    a.sets !== b.sets ||
    a.currentSetId !== b.currentSetId ||
    a.slotRenders !== b.slotRenders ||
    a.samples !== b.samples
  );
}

async function connect(conn: SidecarConnection): Promise<void> {
  const store = useStore.getState();
  setConnection(conn);
  store.setConnection({ status: 'connecting', url: conn.url, error: null });
  for (;;) {
    try {
      const info = await api.health();
      store.setConnection({ status: 'ready', info, error: null });
      break;
    } catch (e) {
      store.setConnection({ status: 'connecting', error: errorMessage(e) });
      await new Promise((r) => setTimeout(r, HEALTH_RETRY_MS));
    }
  }
  await store.loadAll();
}

export function bootstrap(): () => void {
  let unsubscribeSse: (() => void) | null = null;
  let lastSse: SseState = 'closed';

  const stopDiscovery = discoverSidecar({
    onFound: (conn) => {
      void connect(conn).then(() => {
        unsubscribeSse = subscribeEvents({
          onEvent: (name, data) => useStore.getState().applyEvent(name, data),
          onState: (state) => {
            useStore.getState().setConnection({ sse: state });
            // Events may have been lost while the stream was down: resync.
            if (state === 'open' && lastSse === 'reconnecting') void useStore.getState().loadAll();
            lastSse = state;
          },
        });
      });
    },
    onError: (message) => useStore.getState().setConnection({ error: message }),
  });

  let lastSerialized = '';
  const unsubStore = useStore.subscribe((s, prev) => {
    if (programInputsChanged(s, prev)) {
      const program = buildProgram(s);
      const serialized = JSON.stringify(program);
      if (serialized !== lastSerialized) {
        lastSerialized = serialized;
        engine.sync(program);
      }
    }
    if (s.pendingAudition && (s.pendingAudition !== prev.pendingAudition || s.previewRenders !== prev.previewRenders)) {
      const pending = s.pendingAudition;
      const status = s.previewRenders[pending.cacheKey];
      if (!status) return;
      if (status.status === 'ready' && status.url) {
        s.clearPendingAudition();
        void engine.auditionInContext(pending.slotId, status.url).catch((e: unknown) => engine.onError(errorMessage(e)));
      } else if (status.status === 'failed') {
        s.clearPendingAudition();
        useStore.setState({ lastError: `preview render failed: ${status.error ?? 'unknown error'}` });
      }
    }
  });

  const unsubEngine = engine.subscribe((t) => useStore.getState().setTransport(t));
  engine.onError = (message) => useStore.setState({ lastError: message });

  return () => {
    stopDiscovery();
    unsubscribeSse?.();
    unsubStore();
    unsubEngine();
  };
}
