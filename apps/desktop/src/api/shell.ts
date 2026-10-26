// Commands the Tauri shell exposes to the UI. In a plain browser (vite dev
// without Tauri) `isTauri()` is false and callers fall back to text inputs.

import { invoke } from '@tauri-apps/api/core';
import { listen, type UnlistenFn } from '@tauri-apps/api/event';
import type { SidecarConnection } from './client';

export function isTauri(): boolean {
  return typeof window !== 'undefined' && '__TAURI_INTERNALS__' in window;
}

export function sidecarInfo(): Promise<SidecarConnection | null> {
  return invoke<SidecarConnection | null>('sidecar_info');
}

export function onSidecarReady(handler: (conn: SidecarConnection) => void): Promise<UnlistenFn> {
  return listen<SidecarConnection>('sidecar://ready', (e) => handler(e.payload));
}

/** Native multi-file picker; absolute paths. */
export function pickAudioFiles(): Promise<string[]> {
  return invoke<string[]>('pick_audio_files');
}

/** Native directory picker. */
export function pickDirectory(): Promise<string | null> {
  return invoke<string | null>('pick_directory');
}

/** Native folder picker for a library root (a Splice library, a pack folder). */
export function pickAudioFolder(): Promise<string | null> {
  return invoke<string | null>('pick_audio_folder');
}
