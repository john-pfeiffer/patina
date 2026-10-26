// TypeScript mirror of sidecar/src/patine/models.py. Field names are identical
// to the JSON the sidecar emits; keep the two files in step.

export const TILT_BAND_CENTRES_HZ = [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000] as const;

export interface TonalMetrics {
  loudness_lufs: number | null;
  tilt_bands: number[];
  low_rolloff_hz: number | null;
  bass_mono_hz: number | null;
  sub_ratio: number | null;
  noise_floor_db: number | null;
  noise_spectrum: number[];
  width_bands: number[];
  crest_db: number | null;
  wetness: number | null;
}

export interface WarpMarker {
  source_s: number;
  beat: number;
}

export type StretchMethod = 'elastic' | 'repitch';
export type TempoFeel = 'quarter' | 'eighth' | 'half';

export interface UserOverrides {
  keep_preroll: boolean;
  nudge_ms: number;
  warp_markers: WarpMarker[];
  warp_strength: number | null;
  stretch_method: StretchMethod | null;
  allow_wobble: boolean;
  tempo_multiplier: number;
  tempo_feel: TempoFeel;
  bypass: Record<string, boolean>;
}

export type ModuleName = 'tuning' | 'loudness' | 'tilt' | 'lowend' | 'noise' | 'width' | 'density' | 'space';
export const MODULE_NAMES: readonly ModuleName[] = ['tuning', 'loudness', 'tilt', 'lowend', 'noise', 'width', 'density', 'space'];
export const REGION_MODULES: readonly ModuleName[] = ['tuning', 'loudness', 'tilt', 'lowend', 'noise', 'width'];

export interface ModuleSetting {
  enabled: boolean;
  amount: number | null;
}

export type TonalSettings = Record<ModuleName, ModuleSetting>;

export type RepitchMode = 'plain' | 'elastic';

export interface TemporalSettings {
  stretch_method: StretchMethod;
  warp_strength: number;
  transient_snap: boolean;
  repitch_mode: RepitchMode;
  declick_ms: number;
}

export interface Step {
  active: boolean;
  gate: number;
  velocity: number;
  probability: number;
  offset_beats: number;
}

export type PatternMode = 'hold' | 'retrigger';
export type Division = '1/4' | '1/8' | '1/16';

export interface Pattern {
  mode: PatternMode;
  division: Division;
  steps: Step[];
}

export type RegionSource = 'analysis' | 'user';

export interface Region {
  id: string;
  sample_id: string;
  position: number;
  start_s: number;
  end_s: number;
  chord_root: string | null;
  chord_quality: string | null;
  bass_note: string | null;
  chord_confidence: number | null;
  key_context: string | null;
  beats: number[];
  first_onset_s: number | null;
  declared_length_beats: number | null;
  busyness: number | null;
  register_centroid_hz: number | null;
  metrics: TonalMetrics | null;
  user_overrides: UserOverrides;
  source: RegionSource;
}

export type AnalysisStatus = 'pending' | 'running' | 'done' | 'failed';

export interface Sample {
  id: string;
  path: string;
  hash: string;
  name: string;
  duration: number | null;
  sample_rate: number | null;
  channels: number | null;
  added_at: string;
  tuning_cents: number | null;
  detected_bpm: number | null;
  bpm_confidence: number | null;
  beats: number[];
  downbeats: number[];
  /** Whole-file transients (§4.4), seconds. */
  onsets: number[];
  key_estimate: string | null;
  key_confidence: number | null;
  metrics: TonalMetrics | null;
  analysis_status: AnalysisStatus;
  analysis_error: string | null;
  analysis_version: number | null;
  analysis_progress: number;
  regions: Region[];
}

export interface SetMember {
  region: Region;
  position: number;
}

/** `Set` in models.py; renamed here to avoid shadowing the JS built-in. */
export interface ConformSet {
  id: string;
  name: string;
  anchor_region_id: string | null;
  session_bpm: number | null;
  bpm_pinned: boolean;
  tonal_settings: TonalSettings;
  temporal_settings: TemporalSettings;
  created_at: string;
  members: SetMember[];
}

export interface Slot {
  id: string;
  progression_id: string;
  idx: number;
  chord_symbol: string;
  roman: string | null;
  chord_root: string | null;
  chord_quality: string | null;
  bass_note: string | null;
  length_beats: number;
  region_id: string | null;
  transpose_semitones: number;
  locked: boolean;
  pattern: Pattern;
}

export interface Progression {
  id: string;
  set_id: string;
  name: string;
  key: string | null;
  loop: boolean;
  created_at: string;
  slots: Slot[];
}

export type JobStatus = 'queued' | 'running' | 'done' | 'failed' | 'cancelled';
export type JobKind = 'analyze' | 'render' | 'export';

export interface Job {
  id: string;
  kind: JobKind | string;
  status: JobStatus;
  progress: number;
  message: string | null;
  payload: Record<string, unknown>;
  result: Record<string, unknown> | null;
  error: string | null;
  created_at: string;
  updated_at: string;
}

export interface ConformDeltas {
  tuning_cents: number;
  loudness_db: number;
  tilt_db: number[];
  lowend_hpf_hz: number | null;
  lowend_mono_hz: number | null;
  noise_floor_db: number | null;
  width_ratio: number[];
  stretch_ratio: number;
  transpose_semitones: number;
  bypassed: string[];
}

export type RenderState = 'queued' | 'rendering' | 'ready' | 'failed';

export interface RenderStatus {
  cache_key: string;
  region_id: string;
  set_id: string;
  status: RenderState;
  url: string | null;
  error: string | null;
  deltas: ConformDeltas | null;
  stale: boolean;
}

export interface Candidate {
  region: Region;
  sample_name: string;
  sample_id: string;
  /** Other slots of this progression (1-based) whose region comes from the same sample. */
  used_by_slot_idx: number[];
  score: number;
  transposition_cost: number;
  timbral_distance: number;
  confidence_cost: number;
  transpose_semitones: number;
  label: string;
}

// --- request bodies ----------------------------------------------------------

export interface IngestRequest {
  paths: string[];
}

export interface IngestResponse {
  samples: Sample[];
  jobs: Job[];
}

export interface RegionPatch {
  start_s?: number;
  end_s?: number;
  chord_root?: string;
  chord_quality?: string;
  bass_note?: string;
  key_context?: string;
  declared_length_beats?: number;
  clear_declared_length?: boolean;
  user_overrides?: UserOverrides;
}

export interface RegionCreate {
  start_s: number;
  end_s: number;
  chord_symbol?: string;
  declared_length_beats?: number;
}

export interface SetCreate {
  name: string;
}

export interface SetPatch {
  name?: string;
  anchor_region_id?: string;
  session_bpm?: number;
  bpm_pinned?: boolean;
  tonal_settings?: TonalSettings;
  temporal_settings?: TemporalSettings;
}

export interface MemberAdd {
  region_id: string;
}

export interface PreviewRenderRequest {
  region_id: string;
  transpose_semitones?: number;
}

export interface ProgressionCreate {
  set_id: string;
  name?: string;
  key?: string;
  text?: string;
  default_length_beats?: number;
}

export interface ProgressionPatch {
  name?: string;
  key?: string;
  loop?: boolean;
  text?: string;
}

export interface SlotPatch {
  chord_symbol?: string;
  roman?: string;
  length_beats?: number;
  region_id?: string;
  clear_region?: boolean;
  transpose_semitones?: number;
  locked?: boolean;
  pattern?: Pattern;
}

export type Strictness = 'exact' | 'function' | 'two_tones';
export const STRICTNESS_ORDER: readonly Strictness[] = ['exact', 'function', 'two_tones'];

export interface RerollRequest {
  amount: number;
  strictness: Strictness;
  seed?: number;
  /** No two slots on regions of the same source sample (relaxed with a warning when impossible). */
  unique_samples?: boolean;
}

export interface RerollResponse {
  progression: Progression;
  warnings: string[];
}

export interface ParseRequest {
  text: string;
  key?: string;
}

export interface ParsedChord {
  input: string;
  chord_symbol: string;
  roman: string | null;
  root: string;
  quality: string;
  bass: string | null;
  pitch_classes: number[];
  length_beats: number;
}

export interface ParseResponse {
  ok: boolean;
  key: string | null;
  chords: ParsedChord[];
  error: string | null;
}

export interface CsvImportRequest {
  set_id: string;
  csv_text: string;
  default_length_beats?: number;
}

export type ExportKind = 'full' | 'stems' | 'kit' | 'midi';
export const EXPORT_KINDS: readonly ExportKind[] = ['full', 'stems', 'kit', 'midi'];

export interface ExportRequest {
  progression_id: string;
  kind: ExportKind;
  out_dir: string;
  loops: number;
  /** Exports are numbered takes that never overwrite; true restores the fixed `<name>_full.wav` names. */
  overwrite?: boolean;
}

export interface SidecarInfo {
  name: string;
  version: string;
  analysis_version: number;
  render_version: number;
  data_dir: string;
  ml_available: boolean;
}

export interface Peaks {
  bins: number;
  duration: number;
  peaks: number[];
}

// --- SSE payloads ----------------------------------------------------------------

export interface SampleUpdatedEvent {
  sample_id: string;
  analysis_status: AnalysisStatus;
  analysis_progress: number;
  message: string | null;
}

export interface SampleRemovedEvent {
  sample_id: string;
}

export interface RegionUpdatedEvent {
  region_id: string;
  sample_id: string;
}

export interface SetUpdatedEvent {
  set_id: string;
}

export interface SidecarEventMap {
  'sample.updated': SampleUpdatedEvent;
  'sample.removed': SampleRemovedEvent;
  'region.updated': RegionUpdatedEvent;
  'set.updated': SetUpdatedEvent;
  'render.updated': RenderStatus;
  'job.updated': Job;
}

export type SidecarEventName = keyof SidecarEventMap;
export const SIDECAR_EVENT_NAMES: readonly SidecarEventName[] = [
  'sample.updated',
  'sample.removed',
  'region.updated',
  'set.updated',
  'render.updated',
  'job.updated',
];
