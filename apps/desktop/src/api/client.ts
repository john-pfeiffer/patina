// Typed fetch wrapper over docs/API.md. One module-level connection, set once
// discovery (src/api/sidecar.ts) has found the sidecar.

import type {
  Candidate,
  ConformDeltas,
  ConformSet,
  CsvImportRequest,
  ExportRequest,
  IngestRequest,
  IngestResponse,
  Job,
  JobStatus,
  MemberAdd,
  ParseRequest,
  ParseResponse,
  Peaks,
  PreviewRenderRequest,
  Progression,
  ProgressionCreate,
  ProgressionPatch,
  Region,
  RegionCreate,
  RegionPatch,
  RenderStatus,
  RerollRequest,
  Sample,
  SetCreate,
  SetPatch,
  SidecarInfo,
  Slot,
  SlotPatch,
  Strictness,
} from './types';

export interface SidecarConnection {
  url: string;
  token: string;
}

export const TOKEN_HEADER = 'X-Patine-Token';

export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly detail: string,
    readonly path: string,
  ) {
    super(`${status} ${path}: ${detail}`);
    this.name = 'ApiError';
  }
}

let connection: SidecarConnection | null = null;

export function setConnection(conn: SidecarConnection | null): void {
  connection = conn;
}

function requireConnection(): SidecarConnection {
  if (!connection) throw new ApiError(0, 'sidecar not connected', '');
  return connection;
}

/** Absolute URL for a sidecar path. */
export function absoluteUrl(path: string): string {
  const { url } = requireConnection();
  return path.startsWith('http') ? path : `${url}${path}`;
}

/** URL with the token as a query parameter, for consumers that cannot set
 * headers (EventSource, <audio>). The sidecar accepts `?token=` on /events,
 * /renders/*.wav, /samples/{id}/audio and /regions/{id}/audio. */
export function tokenUrl(path: string): string {
  const { token } = requireConnection();
  const u = new URL(absoluteUrl(path));
  u.searchParams.set('token', token);
  return u.toString();
}

async function readDetail(res: Response): Promise<string> {
  try {
    const body = (await res.json()) as { detail?: unknown };
    if (typeof body.detail === 'string') return body.detail;
    return JSON.stringify(body);
  } catch {
    return res.statusText || `HTTP ${res.status}`;
  }
}

async function request<T>(method: string, path: string, body?: unknown, init?: RequestInit): Promise<T> {
  const { url, token } = requireConnection();
  const headers: Record<string, string> = { [TOKEN_HEADER]: token };
  if (body !== undefined) headers['Content-Type'] = 'application/json';
  const res = await fetch(`${url}${path}`, {
    ...init,
    method,
    headers,
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!res.ok) throw new ApiError(res.status, await readDetail(res), path);
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

/** Fetch binary audio with the token header (used by the playback engine). */
export async function fetchAudio(path: string): Promise<ArrayBuffer> {
  const { token } = requireConnection();
  const res = await fetch(absoluteUrl(path), { headers: { [TOKEN_HEADER]: token } });
  if (!res.ok) throw new ApiError(res.status, await readDetail(res), path);
  return res.arrayBuffer();
}

const q = (params: Record<string, string | number | undefined>): string => {
  const sp = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) if (v !== undefined) sp.set(k, String(v));
  const s = sp.toString();
  return s ? `?${s}` : '';
};

export const api = {
  health: () => request<SidecarInfo>('GET', '/health'),

  samples: {
    ingest: (paths: string[]) => request<IngestResponse>('POST', '/samples', { paths } satisfies IngestRequest),
    list: () => request<Sample[]>('GET', '/samples'),
    get: (id: string) => request<Sample>('GET', `/samples/${id}`),
    remove: (id: string) => request<void>('DELETE', `/samples/${id}`),
    reanalyze: (id: string) => request<Job>('POST', `/samples/${id}/reanalyze`),
    peaks: (id: string, bins = 800) => request<Peaks>('GET', `/samples/${id}/peaks${q({ bins })}`),
    audioPath: (id: string) => `/samples/${id}/audio`,
    createRegion: (id: string, body: RegionCreate) => request<Region>('POST', `/samples/${id}/regions`, body),
  },

  regions: {
    get: (id: string) => request<Region>('GET', `/regions/${id}`),
    patch: (id: string, body: RegionPatch) => request<Region>('PATCH', `/regions/${id}`, body),
    remove: (id: string) => request<void>('DELETE', `/regions/${id}`),
    audioPath: (id: string) => `/regions/${id}/audio`,
  },

  sets: {
    create: (body: SetCreate) => request<ConformSet>('POST', '/sets', body),
    list: () => request<ConformSet[]>('GET', '/sets'),
    get: (id: string) => request<ConformSet>('GET', `/sets/${id}`),
    patch: (id: string, body: SetPatch) => request<ConformSet>('PATCH', `/sets/${id}`, body),
    remove: (id: string) => request<void>('DELETE', `/sets/${id}`),
    addMember: (id: string, body: MemberAdd) => request<ConformSet>('POST', `/sets/${id}/members`, body),
    removeMember: (id: string, regionId: string) => request<ConformSet>('DELETE', `/sets/${id}/members/${regionId}`),
    deltas: (id: string) => request<Record<string, ConformDeltas>>('GET', `/sets/${id}/deltas`),
    renders: (id: string) => request<RenderStatus[]>('GET', `/sets/${id}/renders`),
    previewRender: (id: string, body: PreviewRenderRequest) =>
      request<RenderStatus>('POST', `/sets/${id}/renders/preview`, body),
  },

  progressions: {
    create: (body: ProgressionCreate) => request<Progression>('POST', '/progressions', body),
    list: (setId?: string) => request<Progression[]>('GET', `/progressions${q({ set_id: setId })}`),
    get: (id: string) => request<Progression>('GET', `/progressions/${id}`),
    patch: (id: string, body: ProgressionPatch) => request<Progression>('PATCH', `/progressions/${id}`, body),
    remove: (id: string) => request<void>('DELETE', `/progressions/${id}`),
    renders: (id: string) => request<Record<string, RenderStatus>>('GET', `/progressions/${id}/renders`),
    reroll: (id: string, body: RerollRequest) => request<Progression>('POST', `/progressions/${id}/reroll`, body),
    importCsv: (body: CsvImportRequest) => request<Progression[]>('POST', '/progressions/import-csv', body),
  },

  slots: {
    patch: (id: string, body: SlotPatch) => request<Slot>('PATCH', `/slots/${id}`, body),
    candidates: (id: string, strictness: Strictness, limit = 50) =>
      request<Candidate[]>('GET', `/slots/${id}/candidates${q({ strictness, limit })}`),
  },

  theory: {
    parse: (body: ParseRequest) => request<ParseResponse>('POST', '/theory/parse', body),
  },

  renders: {
    wavPath: (cacheKey: string) => `/renders/${cacheKey}.wav`,
  },

  export: (body: ExportRequest) => request<Job>('POST', '/export', body),

  jobs: {
    list: (status?: JobStatus) => request<Job[]>('GET', `/jobs${q({ status })}`),
    get: (id: string) => request<Job>('GET', `/jobs/${id}`),
    cancel: (id: string) => request<Job>('POST', `/jobs/${id}/cancel`),
  },
};
