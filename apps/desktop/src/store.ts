// Application state. One zustand store: library, sets, progressions, renders,
// jobs, transport and UI state. SSE events (src/api/events.ts) are applied
// here via `applyEvent`; the playback engine is driven from src/bootstrap.ts
// by subscribing to this store, so the store itself never imports Tone.

import { create } from 'zustand';
import { api } from './api/client';
import type { SseState } from './api/events';
import type {
  Candidate,
  ConformDeltas,
  ConformSet,
  ExportKind,
  Job,
  LibraryRoot,
  Progression,
  ProgressionCreate,
  ProgressionPatch,
  Region,
  RegionPatch,
  RenderStatus,
  Sample,
  SetPatch,
  Settings,
  SettingsPatch,
  SidecarEventMap,
  SidecarEventName,
  SidecarInfo,
  Slot,
  SlotPatch,
  Strictness,
} from './api/types';
import { candidateKey, orderCandidates } from './util/candidates';
import { transposeBetween } from './util/chords';
import { errorMessage } from './util/format';

export type ConnectionStatus = 'starting' | 'connecting' | 'ready' | 'error';

export interface ConnectionState {
  status: ConnectionStatus;
  url: string | null;
  info: SidecarInfo | null;
  error: string | null;
  sse: SseState;
}

export interface TransportState {
  playing: boolean;
  positionBeats: number;
  contextState?: string;
  sampleRate?: number;
  loadedSlots?: number;
  totalSlots?: number;
  /** Diagnostics mirrored from the engine snapshot (src/audio/engine.ts). */
  clockSource?: string;
  ticking?: boolean;
  ticksPerSecond?: number;
  clockNote?: string | null;
  baseLatency?: number | null;
  outputLatency?: number | null;
  lookAhead?: number;
  sinkId?: string | null;
}

export type DialogName = 'preferences';
export type PrefsTab = 'audio' | 'library' | 'analysis' | 'appearance';

export interface UiState {
  selectedSlotId: string | null;
  strictness: Strictness;
  rerollAmount: number;
  paneFractions: [number, number, number];
  collapsedSamples: Record<string, true>;
  outDir: string;
  exportLoops: number;
  /** Which modal is open, if any. */
  dialog: DialogName | null;
  prefsTab: PrefsTab;
  /** Chosen output device (Chromium webviews only); null is the system default. */
  audioSinkId: string | null;
  /** Reroll and the candidate drawer never put two slots on the same source sample. */
  uniqueSamples: boolean;
  /** Library pane filter: name, folder, chord, key. */
  libraryFilter: string;
  /** Collapsed folders of the library tree, by `LibraryFolder.key`. */
  collapsedFolders: Record<string, true>;
}

export interface PendingAudition {
  slotId: string;
  cacheKey: string;
}

export interface PatineState {
  connection: ConnectionState;
  samples: Sample[];
  /** Folders the library scans (Preferences › Library). */
  roots: LibraryRoot[];
  /** Sidecar settings; null until loaded. */
  settings: Settings | null;
  sets: ConformSet[];
  currentSetId: string | null;
  progressions: Progression[];
  currentProgressionId: string | null;
  /** Renders for the current progression, by slot id. */
  slotRenders: Record<string, RenderStatus>;
  /** Last render that was ready per slot: what the engine keeps playing while a newer key renders. */
  slotLastGood: Record<string, RenderStatus>;
  /** Renders for the current set's members at transpose 0, by region id. */
  memberRenders: Record<string, RenderStatus>;
  /** In-context preview renders, by cache key. */
  previewRenders: Record<string, RenderStatus>;
  pendingAudition: PendingAudition | null;
  deltas: Record<string, ConformDeltas>;
  candidates: Record<string, Candidate[]>;
  jobs: Record<string, Job>;
  transport: TransportState;
  ui: UiState;
  lastError: string | null;

  setConnection: (patch: Partial<ConnectionState>) => void;
  setTransport: (patch: Partial<TransportState>) => void;
  setUi: (patch: Partial<UiState>) => void;
  clearError: () => void;

  loadAll: () => Promise<void>;

  refreshSample: (id: string) => Promise<void>;
  refreshRegion: (id: string) => Promise<void>;
  ingest: (paths: string[]) => Promise<void>;
  removeSample: (id: string) => Promise<void>;
  reanalyze: (id: string) => Promise<void>;
  reanalyzeAll: () => Promise<void>;
  patchRegion: (id: string, patch: RegionPatch) => Promise<void>;

  refreshRoots: () => Promise<void>;
  addRoot: (path: string, name?: string) => Promise<void>;
  scanRoot: (id: string) => Promise<void>;
  scanAllRoots: () => Promise<void>;
  removeRoot: (id: string, deleteSamples: boolean) => Promise<void>;
  refreshSettings: () => Promise<void>;
  patchSettings: (patch: SettingsPatch) => Promise<void>;

  refreshSets: () => Promise<void>;
  refreshSet: (id: string) => Promise<void>;
  selectSet: (id: string | null) => Promise<void>;
  createSet: (name: string) => Promise<void>;
  patchSet: (patch: SetPatch) => Promise<void>;
  removeSet: (id: string) => Promise<void>;
  addMember: (regionId: string) => Promise<void>;
  removeMember: (regionId: string) => Promise<void>;
  refreshDeltas: () => Promise<void>;
  refreshMemberRenders: () => Promise<void>;

  refreshProgressions: () => Promise<void>;
  selectProgression: (id: string | null) => Promise<void>;
  createProgression: (body: Omit<ProgressionCreate, 'set_id'>) => Promise<void>;
  patchProgression: (patch: ProgressionPatch) => Promise<void>;
  removeProgression: (id: string) => Promise<void>;
  refreshSlotRenders: () => Promise<void>;
  patchSlot: (id: string, patch: SlotPatch) => Promise<void>;
  assignCandidate: (slotId: string, candidate: Candidate) => Promise<void>;
  assignRegionToSlot: (slotId: string, region: Region) => Promise<void>;
  loadCandidates: (slotId: string, strictness: Strictness) => Promise<void>;
  reroll: () => Promise<void>;
  importCsv: (csvText: string) => Promise<void>;

  previewInContext: (region: Region, slotId: string) => Promise<void>;
  clearPendingAudition: () => void;

  cancelJob: (id: string) => Promise<void>;
  startExport: (kind: ExportKind) => Promise<void>;

  applyEvent: <K extends SidecarEventName>(name: K, data: SidecarEventMap[K]) => void;
}

// --- selectors ------------------------------------------------------------------

export const selectCurrentSet = (s: PatineState): ConformSet | null =>
  s.sets.find((x) => x.id === s.currentSetId) ?? null;

export const selectCurrentProgression = (s: PatineState): Progression | null =>
  s.progressions.find((x) => x.id === s.currentProgressionId) ?? null;

export const selectSelectedSlot = (s: PatineState): Slot | null => {
  const p = selectCurrentProgression(s);
  return p?.slots.find((x) => x.id === s.ui.selectedSlotId) ?? null;
};

export function regionsById(samples: readonly Sample[]): Map<string, Region> {
  const m = new Map<string, Region>();
  for (const sample of samples) for (const r of sample.regions) m.set(r.id, r);
  return m;
}

export function samplesById(samples: readonly Sample[]): Map<string, Sample> {
  return new Map(samples.map((s) => [s.id, s]));
}

export const selectAnchorRegion = (s: PatineState): Region | null => {
  const set = selectCurrentSet(s);
  if (!set?.anchor_region_id) return null;
  return set.members.find((m) => m.region.id === set.anchor_region_id)?.region ?? null;
};

export interface QueueCounts {
  scanActive: number;
  analyzeQueued: number;
  analyzeRunning: number;
  renderQueued: number;
  renderRunning: number;
  exportActive: number;
}

export const selectQueueCounts = (jobs: Record<string, Job>): QueueCounts => {
  const c: QueueCounts = { scanActive: 0, analyzeQueued: 0, analyzeRunning: 0, renderQueued: 0, renderRunning: 0, exportActive: 0 };
  for (const j of Object.values(jobs)) {
    if (j.kind === 'scan' && (j.status === 'queued' || j.status === 'running')) c.scanActive++;
    else if (j.kind === 'analyze') {
      if (j.status === 'queued') c.analyzeQueued++;
      else if (j.status === 'running') c.analyzeRunning++;
    } else if (j.kind === 'render') {
      if (j.status === 'queued') c.renderQueued++;
      else if (j.status === 'running') c.renderRunning++;
    } else if (j.kind === 'export' && (j.status === 'queued' || j.status === 'running')) c.exportActive++;
  }
  return c;
};

export const DEFAULT_BPM = 120;

export const sessionBpm = (set: ConformSet | null): number => set?.session_bpm ?? DEFAULT_BPM;

// --- persistence of small UI prefs -------------------------------------------------

const PREFS_KEY = 'patine.ui';

interface Prefs {
  currentSetId: string | null;
  currentProgressionId: string | null;
  paneFractions: [number, number, number];
  outDir: string;
  audioSinkId: string | null;
  uniqueSamples: boolean;
}

function loadPrefs(): Partial<Prefs> {
  try {
    const raw = localStorage.getItem(PREFS_KEY);
    return raw ? (JSON.parse(raw) as Partial<Prefs>) : {};
  } catch {
    return {};
  }
}

function savePrefs(p: Prefs): void {
  try {
    localStorage.setItem(PREFS_KEY, JSON.stringify(p));
  } catch {
    // storage unavailable: prefs are a convenience only
  }
}

// --- debounced refetches (SSE bursts) ------------------------------------------------

const timers = new Map<string, ReturnType<typeof setTimeout>>();

function debounced(key: string, ms: number, fn: () => void): void {
  const t = timers.get(key);
  if (t) clearTimeout(t);
  timers.set(
    key,
    setTimeout(() => {
      timers.delete(key);
      fn();
    }, ms),
  );
}

function replaceById<T extends { id: string }>(list: readonly T[], item: T): T[] {
  const i = list.findIndex((x) => x.id === item.id);
  if (i < 0) return [...list, item];
  const out = list.slice();
  out[i] = item;
  return out;
}

function replaceRegion(samples: readonly Sample[], region: Region): Sample[] {
  return samples.map((s) =>
    s.id === region.sample_id ? { ...s, regions: replaceById(s.regions, region).sort((a, b) => a.position - b.position) } : s,
  );
}

/** Remember the last ready render per slot so the UI can say "playing stale". */
function rememberLastGood(prev: Record<string, RenderStatus>, renders: Record<string, RenderStatus>): Record<string, RenderStatus> {
  let out = prev;
  for (const [slotId, r] of Object.entries(renders)) {
    if (r.status === 'ready' && r.url && out[slotId]?.cache_key !== r.cache_key) {
      if (out === prev) out = { ...prev };
      out[slotId] = r;
    }
  }
  return out;
}

const MAX_JOBS = 200;

/** Keep the job list bounded: drop the oldest finished jobs past MAX_JOBS. */
function pruneJobs(jobs: Record<string, Job>): Record<string, Job> {
  const all = Object.values(jobs);
  if (all.length <= MAX_JOBS) return jobs;
  const finished = all
    .filter((j) => j.status !== 'queued' && j.status !== 'running')
    .sort((a, b) => (a.updated_at < b.updated_at ? -1 : 1));
  const drop = new Set(finished.slice(0, all.length - MAX_JOBS).map((j) => j.id));
  return Object.fromEntries(all.filter((j) => !drop.has(j.id)).map((j) => [j.id, j]));
}

const prefs = loadPrefs();

export const useStore = create<PatineState>()((set, get) => {
  /** Runs an API call, routing failures to `lastError` instead of throwing. */
  const run = async (fn: () => Promise<void>): Promise<void> => {
    try {
      await fn();
    } catch (e) {
      set({ lastError: errorMessage(e) });
    }
  };

  const persist = () => {
    const s = get();
    savePrefs({
      currentSetId: s.currentSetId,
      currentProgressionId: s.currentProgressionId,
      paneFractions: s.ui.paneFractions,
      outDir: s.ui.outDir,
      audioSinkId: s.ui.audioSinkId,
      uniqueSamples: s.ui.uniqueSamples,
    });
  };

  const isMember = (regionId: string): boolean =>
    selectCurrentSet(get())?.members.some((m) => m.region.id === regionId) ?? false;

  return {
    connection: { status: 'starting', url: null, info: null, error: null, sse: 'closed' },
    samples: [],
    roots: [],
    settings: null,
    sets: [],
    currentSetId: prefs.currentSetId ?? null,
    progressions: [],
    currentProgressionId: prefs.currentProgressionId ?? null,
    slotRenders: {},
    slotLastGood: {},
    memberRenders: {},
    previewRenders: {},
    pendingAudition: null,
    deltas: {},
    candidates: {},
    jobs: {},
    transport: { playing: false, positionBeats: 0 },
    ui: {
      selectedSlotId: null,
      strictness: 'function',
      rerollAmount: 100,
      paneFractions: prefs.paneFractions ?? [1.2, 1, 1.4],
      collapsedSamples: {},
      outDir: prefs.outDir ?? '',
      exportLoops: 1,
      dialog: null,
      prefsTab: 'audio',
      audioSinkId: prefs.audioSinkId ?? null,
      uniqueSamples: prefs.uniqueSamples ?? false,
      libraryFilter: '',
      collapsedFolders: {},
    },
    lastError: null,

    setConnection: (patch) => set((s) => ({ connection: { ...s.connection, ...patch } })),
    setTransport: (patch) => set((s) => ({ transport: { ...s.transport, ...patch } })),
    setUi: (patch) => {
      set((s) => ({ ui: { ...s.ui, ...patch } }));
      if (patch.paneFractions || patch.outDir !== undefined || patch.audioSinkId !== undefined || patch.uniqueSamples !== undefined) persist();
    },
    clearError: () => set({ lastError: null }),

    loadAll: () =>
      run(async () => {
        const [samples, sets, jobs, roots, settings] = await Promise.all([
          api.samples.list(),
          api.sets.list(),
          api.jobs.list(),
          api.roots.list(),
          api.settings.get(),
        ]);
        set({ samples, sets, roots, settings, jobs: Object.fromEntries(jobs.map((j) => [j.id, j])) });
        const wanted = get().currentSetId;
        const setId = sets.some((x) => x.id === wanted) ? wanted : (sets[0]?.id ?? null);
        await get().selectSet(setId);
      }),

    // --- library -------------------------------------------------------------------

    refreshSample: (id) =>
      run(async () => {
        const sample = await api.samples.get(id);
        set((s) => ({ samples: replaceById(s.samples, sample), candidates: {} }));
      }),

    refreshRegion: (id) =>
      run(async () => {
        const region = await api.regions.get(id);
        set((s) => ({ samples: replaceRegion(s.samples, region), candidates: {} }));
      }),

    ingest: (paths) =>
      run(async () => {
        const res = await api.samples.ingest(paths);
        set((s) => ({
          samples: res.samples.reduce((acc, sample) => replaceById(acc, sample), s.samples),
          jobs: { ...s.jobs, ...Object.fromEntries(res.jobs.map((j) => [j.id, j])) },
        }));
      }),

    removeSample: (id) =>
      run(async () => {
        await api.samples.remove(id);
        set((s) => ({ samples: s.samples.filter((x) => x.id !== id), candidates: {} }));
        await get().refreshSets();
      }),

    reanalyze: (id) =>
      run(async () => {
        const job = await api.samples.reanalyze(id);
        set((s) => ({ jobs: { ...s.jobs, [job.id]: job } }));
      }),

    reanalyzeAll: () =>
      run(async () => {
        const jobs = await api.samples.reanalyzeAll();
        set((s) => ({ jobs: pruneJobs({ ...s.jobs, ...Object.fromEntries(jobs.map((j) => [j.id, j])) }) }));
      }),

    // --- library roots & settings --------------------------------------------------

    refreshRoots: () =>
      run(async () => {
        set({ roots: await api.roots.list() });
      }),

    addRoot: (path, name) =>
      run(async () => {
        const { root, job } = await api.roots.add({ path, name });
        set((s) => ({ roots: replaceById(s.roots, root), jobs: { ...s.jobs, [job.id]: job } }));
      }),

    scanRoot: (id) =>
      run(async () => {
        const job = await api.roots.scan(id);
        set((s) => ({ jobs: { ...s.jobs, [job.id]: job } }));
      }),

    scanAllRoots: () =>
      run(async () => {
        const jobs = await api.roots.scanAll();
        set((s) => ({ jobs: { ...s.jobs, ...Object.fromEntries(jobs.map((j) => [j.id, j])) } }));
      }),

    removeRoot: (id, deleteSamples) =>
      run(async () => {
        await api.roots.remove(id, deleteSamples);
        set((s) => ({ roots: s.roots.filter((r) => r.id !== id) }));
        await get().loadAll();
      }),

    refreshSettings: () =>
      run(async () => {
        set({ settings: await api.settings.get() });
      }),

    patchSettings: (patch) =>
      run(async () => {
        set({ settings: await api.settings.patch(patch) });
      }),

    patchRegion: (id, patch) =>
      run(async () => {
        const region = await api.regions.patch(id, patch);
        set((s) => ({ samples: replaceRegion(s.samples, region), candidates: {} }));
        if (isMember(id)) {
          await Promise.all([get().refreshSet(get().currentSetId ?? ''), get().refreshDeltas()]);
        }
      }),

    // --- sets ------------------------------------------------------------------------

    refreshSets: () =>
      run(async () => {
        const sets = await api.sets.list();
        set({ sets });
        if (get().currentSetId && !sets.some((x) => x.id === get().currentSetId)) await get().selectSet(sets[0]?.id ?? null);
      }),

    refreshSet: (id) =>
      run(async () => {
        if (!id) return;
        const one = await api.sets.get(id);
        set((s) => ({ sets: replaceById(s.sets, one) }));
      }),

    selectSet: async (id) => {
      if (id !== get().currentSetId) {
        set((s) => ({
          currentSetId: id,
          progressions: [],
          currentProgressionId: null,
          slotRenders: {},
          slotLastGood: {},
          memberRenders: {},
          deltas: {},
          candidates: {},
          ui: { ...s.ui, selectedSlotId: null },
        }));
        persist();
      }
      if (!id) return;
      await Promise.all([get().refreshSet(id), get().refreshDeltas(), get().refreshMemberRenders(), get().refreshProgressions()]);
    },

    createSet: (name) =>
      run(async () => {
        const created = await api.sets.create({ name });
        set((s) => ({ sets: [...s.sets, created] }));
        await get().selectSet(created.id);
      }),

    patchSet: (patch) =>
      run(async () => {
        const id = get().currentSetId;
        if (!id) return;
        const updated = await api.sets.patch(id, patch);
        set((s) => ({ sets: replaceById(s.sets, updated), candidates: {} }));
        await Promise.all([get().refreshDeltas(), get().refreshMemberRenders(), get().refreshSlotRenders()]);
      }),

    removeSet: (id) =>
      run(async () => {
        await api.sets.remove(id);
        const sets = get().sets.filter((x) => x.id !== id);
        set({ sets });
        if (get().currentSetId === id) await get().selectSet(sets[0]?.id ?? null);
      }),

    addMember: (regionId) =>
      run(async () => {
        const id = get().currentSetId;
        if (!id) return;
        const updated = await api.sets.addMember(id, { region_id: regionId });
        set((s) => ({ sets: replaceById(s.sets, updated), candidates: {} }));
        await Promise.all([get().refreshDeltas(), get().refreshMemberRenders()]);
      }),

    removeMember: (regionId) =>
      run(async () => {
        const id = get().currentSetId;
        if (!id) return;
        const updated = await api.sets.removeMember(id, regionId);
        set((s) => {
          const memberRenders = { ...s.memberRenders };
          const deltas = { ...s.deltas };
          delete memberRenders[regionId];
          delete deltas[regionId];
          return { sets: replaceById(s.sets, updated), memberRenders, deltas, candidates: {} };
        });
        await Promise.all([get().refreshDeltas(), get().refreshMemberRenders()]);
      }),

    refreshDeltas: () =>
      run(async () => {
        const id = get().currentSetId;
        if (!id) return;
        const deltas = await api.sets.deltas(id);
        if (get().currentSetId === id) set({ deltas });
      }),

    refreshMemberRenders: () =>
      run(async () => {
        const id = get().currentSetId;
        if (!id) return;
        const list = await api.sets.renders(id);
        if (get().currentSetId === id) set({ memberRenders: Object.fromEntries(list.map((r) => [r.region_id, r])) });
      }),

    // --- progressions ------------------------------------------------------------------

    refreshProgressions: () =>
      run(async () => {
        const setId = get().currentSetId;
        if (!setId) return;
        const progressions = await api.progressions.list(setId);
        if (get().currentSetId !== setId) return;
        set({ progressions });
        const wanted = get().currentProgressionId;
        const next = progressions.some((p) => p.id === wanted) ? wanted : (progressions[0]?.id ?? null);
        await get().selectProgression(next);
      }),

    selectProgression: async (id) => {
      if (id !== get().currentProgressionId) {
        set((s) => ({
          currentProgressionId: id,
          slotRenders: {},
          slotLastGood: {},
          candidates: {},
          ui: { ...s.ui, selectedSlotId: null },
        }));
        persist();
      }
      if (id) await get().refreshSlotRenders();
    },

    createProgression: (body) =>
      run(async () => {
        const setId = get().currentSetId;
        if (!setId) return;
        const created = await api.progressions.create({ set_id: setId, ...body });
        set((s) => ({ progressions: [...s.progressions, created] }));
        await get().selectProgression(created.id);
      }),

    patchProgression: (patch) =>
      run(async () => {
        const id = get().currentProgressionId;
        if (!id) return;
        const updated = await api.progressions.patch(id, patch);
        set((s) => ({ progressions: replaceById(s.progressions, updated), candidates: {} }));
        if (patch.text !== undefined) await get().refreshSlotRenders();
      }),

    removeProgression: (id) =>
      run(async () => {
        await api.progressions.remove(id);
        const progressions = get().progressions.filter((p) => p.id !== id);
        set({ progressions });
        if (get().currentProgressionId === id) await get().selectProgression(progressions[0]?.id ?? null);
      }),

    refreshSlotRenders: () =>
      run(async () => {
        const id = get().currentProgressionId;
        if (!id) return;
        const slotRenders = await api.progressions.renders(id);
        if (get().currentProgressionId === id) set((s) => ({ slotRenders, slotLastGood: rememberLastGood(s.slotLastGood, slotRenders) }));
      }),

    patchSlot: (id, patch) =>
      run(async () => {
        const slot = await api.slots.patch(id, patch);
        set((s) => ({
          progressions: s.progressions.map((p) =>
            p.id === slot.progression_id ? { ...p, slots: replaceById(p.slots, slot).sort((a, b) => a.idx - b.idx) } : p,
          ),
          // A chord change changes the pools; an assignment changes which samples are in use.
          candidates:
            patch.chord_symbol !== undefined || patch.roman !== undefined || patch.region_id !== undefined || patch.clear_region
              ? {}
              : s.candidates,
        }));
        if (patch.region_id !== undefined || patch.clear_region || patch.transpose_semitones !== undefined) {
          await get().refreshSlotRenders();
        }
      }),

    assignCandidate: (slotId, candidate) =>
      get().patchSlot(slotId, { region_id: candidate.region.id, transpose_semitones: candidate.transpose_semitones }),

    assignRegionToSlot: async (slotId, region) => {
      const slot = selectCurrentProgression(get())?.slots.find((x) => x.id === slotId);
      if (!slot) return;
      await get().patchSlot(slotId, {
        region_id: region.id,
        transpose_semitones: transposeBetween(region.chord_root, slot.chord_root),
      });
    },

    loadCandidates: (slotId, strictness) =>
      run(async () => {
        const unique = get().ui.uniqueSamples;
        const list = await api.slots.candidates(slotId, strictness, unique);
        set((s) => ({ candidates: { ...s.candidates, [candidateKey(slotId, strictness, unique)]: orderCandidates(list) } }));
      }),

    reroll: () =>
      run(async () => {
        const id = get().currentProgressionId;
        if (!id) return;
        const { rerollAmount, strictness, uniqueSamples } = get().ui;
        const before = get().progressions.find((p) => p.id === id);
        const { progression: updated, warnings } = await api.progressions.reroll(id, {
          amount: rerollAmount,
          strictness,
          unique_samples: uniqueSamples,
        });
        set((s) => ({ progressions: replaceById(s.progressions, updated), candidates: {} }));
        const changed = updated.slots.some((sl, i) => sl.region_id !== before?.slots[i]?.region_id);
        if (warnings.length > 0) set({ lastError: `reroll: ${warnings.join('; ')}` });
        if (!changed) {
          const unassigned = updated.slots.filter((sl) => !sl.region_id && !sl.locked).length;
          set({
            lastError:
              unassigned > 0
                ? `reroll changed nothing: no candidates at “${strictness}” strictness for ${unassigned} unlocked slot(s) — loosen the strictness knob (shares two tones) or assign from the library`
                : 'reroll changed nothing (all slots locked, or amount too low)',
          });
        }
        await get().refreshSlotRenders();
      }),

    importCsv: (csvText) =>
      run(async () => {
        const setId = get().currentSetId;
        if (!setId) return;
        const imported = await api.progressions.importCsv({ set_id: setId, csv_text: csvText });
        set((s) => ({ progressions: imported.reduce((acc, p) => replaceById(acc, p), s.progressions) }));
        if (imported[0] && !get().currentProgressionId) await get().selectProgression(imported[0].id);
      }),

    // --- audition ------------------------------------------------------------------------

    previewInContext: (region, slotId) =>
      run(async () => {
        const setId = get().currentSetId;
        const slot = selectCurrentProgression(get())?.slots.find((x) => x.id === slotId);
        if (!setId || !slot) return;
        const status = await api.sets.previewRender(setId, {
          region_id: region.id,
          transpose_semitones: transposeBetween(region.chord_root, slot.chord_root),
        });
        set((s) => ({
          previewRenders: { ...s.previewRenders, [status.cache_key]: status },
          pendingAudition: { slotId, cacheKey: status.cache_key },
        }));
      }),

    clearPendingAudition: () => set({ pendingAudition: null }),

    // --- jobs & export -------------------------------------------------------------------

    cancelJob: (id) =>
      run(async () => {
        const job = await api.jobs.cancel(id);
        set((s) => ({ jobs: { ...s.jobs, [job.id]: job } }));
      }),

    startExport: (kind) =>
      run(async () => {
        const id = get().currentProgressionId;
        const { outDir, exportLoops } = get().ui;
        if (!id || !outDir) {
          set({ lastError: 'export: pick an output directory first' });
          return;
        }
        const job = await api.export({ progression_id: id, kind, out_dir: outDir, loops: exportLoops });
        set((s) => ({ jobs: { ...s.jobs, [job.id]: job } }));
      }),

    // --- SSE -----------------------------------------------------------------------------

    applyEvent: (name, data) => {
      switch (name) {
        case 'sample.updated': {
          const ev = data as SidecarEventMap['sample.updated'];
          set((s) => ({
            samples: s.samples.map((x) =>
              x.id === ev.sample_id
                ? { ...x, analysis_status: ev.analysis_status, analysis_progress: ev.analysis_progress }
                : x,
            ),
          }));
          const known = get().samples.some((x) => x.id === ev.sample_id);
          if (!known || ev.analysis_status === 'done' || ev.analysis_status === 'failed') {
            debounced(`sample:${ev.sample_id}`, 150, () => void get().refreshSample(ev.sample_id));
          }
          break;
        }
        case 'sample.removed': {
          const ev = data as SidecarEventMap['sample.removed'];
          set((s) => ({ samples: s.samples.filter((x) => x.id !== ev.sample_id), candidates: {} }));
          debounced('sets', 200, () => void get().refreshSets());
          break;
        }
        case 'region.updated': {
          const ev = data as SidecarEventMap['region.updated'];
          debounced(`region:${ev.region_id}`, 150, () => void get().refreshRegion(ev.region_id));
          if (isMember(ev.region_id)) {
            const setId = get().currentSetId ?? '';
            debounced(`set:${setId}`, 300, () => void Promise.all([get().refreshSet(setId), get().refreshDeltas()]));
          }
          break;
        }
        case 'set.updated': {
          const ev = data as SidecarEventMap['set.updated'];
          debounced(`set:${ev.set_id}`, 300, () => {
            void get().refreshSet(ev.set_id);
            if (get().currentSetId === ev.set_id) {
              void get().refreshDeltas();
              void get().refreshMemberRenders();
              void get().refreshSlotRenders();
              set({ candidates: {} });
            }
          });
          break;
        }
        case 'render.updated': {
          const rs = data as SidecarEventMap['render.updated'];
          set((s) => {
            const next: Partial<PatineState> = {};
            if (s.previewRenders[rs.cache_key]) next.previewRenders = { ...s.previewRenders, [rs.cache_key]: rs };
            const member = s.memberRenders[rs.region_id];
            if (member && member.cache_key === rs.cache_key) next.memberRenders = { ...s.memberRenders, [rs.region_id]: rs };
            const slotRenders = { ...s.slotRenders };
            let touched = false;
            for (const [slotId, existing] of Object.entries(slotRenders)) {
              if (existing.cache_key === rs.cache_key) {
                slotRenders[slotId] = rs;
                touched = true;
              }
            }
            if (touched) {
              next.slotRenders = slotRenders;
              next.slotLastGood = rememberLastGood(s.slotLastGood, slotRenders);
            }
            return next;
          });
          if (rs.set_id !== get().currentSetId) break;
          // `render.updated` carries no slot id and no transpose: when the key
          // is new to us (or a key we hold went stale) refetch the lists.
          const s = get();
          const memberMatch = s.memberRenders[rs.region_id]?.cache_key === rs.cache_key;
          if (isMember(rs.region_id) && (!memberMatch || rs.stale)) {
            debounced('member-renders', 250, () => void get().refreshMemberRenders());
          }
          const prog = selectCurrentProgression(s);
          const usesRegion = prog?.slots.some((sl) => sl.region_id === rs.region_id) ?? false;
          const slotMatch = Object.values(s.slotRenders).some((r) => r.cache_key === rs.cache_key);
          if (usesRegion && (!slotMatch || rs.stale)) {
            debounced('slot-renders', 250, () => void get().refreshSlotRenders());
          }
          break;
        }
        case 'job.updated': {
          const job = data as SidecarEventMap['job.updated'];
          set((s) => ({ jobs: pruneJobs({ ...s.jobs, [job.id]: job }) }));
          break;
        }
        case 'root.updated': {
          const ev = data as SidecarEventMap['root.updated'];
          if (ev.status === 'removed') {
            set((s) => ({ roots: s.roots.filter((r) => r.id !== ev.root_id) }));
            debounced('load-all', 300, () => void get().loadAll());
            break;
          }
          set((s) => ({
            roots: s.roots.map((r) =>
              r.id === ev.root_id
                ? { ...r, status: ev.status as LibraryRoot['status'], file_count: ev.file_count ?? r.file_count, error: ev.error ?? null, last_scan_at: ev.last_scan_at ?? r.last_scan_at }
                : r,
            ),
          }));
          if (!get().roots.some((r) => r.id === ev.root_id)) debounced('roots', 300, () => void get().refreshRoots());
          break;
        }
        case 'settings.updated': {
          set({ settings: data as SidecarEventMap['settings.updated'] });
          break;
        }
      }
    },
  };
});
