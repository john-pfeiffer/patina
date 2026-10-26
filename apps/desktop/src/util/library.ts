import type { LibraryRoot, Sample } from '../api/types';

/** One folder of the library tree: a root's subfolder (or its top level), or
 * the loose files added one by one. */
export interface LibraryFolder {
  /** Stable key for collapse state: `<root id>/<relative folder>` or `loose`. */
  key: string;
  rootId: string | null;
  /** Folder path relative to the root, '' for its top level. */
  folder: string;
  samples: Sample[];
}

export interface LibraryGroup {
  root: LibraryRoot | null;
  title: string;
  folders: LibraryFolder[];
  count: number;
  analysed: number;
  missing: number;
}

const LOOSE = 'loose';

function folderOf(sample: Sample): string {
  if (!sample.rel_path) return '';
  const i = Math.max(sample.rel_path.lastIndexOf('/'), sample.rel_path.lastIndexOf('\\'));
  return i < 0 ? '' : sample.rel_path.slice(0, i);
}

/** Case-insensitive match of the filter against name, folder, chord symbols,
 * key names and qualities of a sample. Empty filter matches everything. */
export function sampleMatches(sample: Sample, filter: string): boolean {
  const q = filter.trim().toLowerCase();
  if (!q) return true;
  const hay = [
    sample.name,
    sample.rel_path ?? '',
    sample.key_estimate ?? '',
    ...sample.regions.flatMap((r) => [
      r.chord_root && r.chord_quality ? `${r.chord_root}${r.chord_quality === 'maj' ? '' : r.chord_quality}` : '',
      r.chord_quality ?? '',
      r.key_context ?? '',
    ]),
  ]
    .join(' ')
    .toLowerCase();
  return q.split(/\s+/).every((term) => hay.includes(term));
}

/** Group samples by root and folder for the library tree. Roots come first
 * in their listed order; files added one by one form the last group. Roots
 * with no matching samples are kept (so an empty or still-scanning root is
 * visible) unless a filter is active. */
export function groupLibrary(samples: readonly Sample[], roots: readonly LibraryRoot[], filter = ''): LibraryGroup[] {
  const byRoot = new Map<string | null, Sample[]>();
  for (const s of samples) {
    if (!sampleMatches(s, filter)) continue;
    const key = s.root_id && roots.some((r) => r.id === s.root_id) ? s.root_id : null;
    const list = byRoot.get(key) ?? [];
    list.push(s);
    byRoot.set(key, list);
  }
  const groups: LibraryGroup[] = [];
  const build = (root: LibraryRoot | null, list: Sample[]): LibraryGroup => {
    const folders = new Map<string, Sample[]>();
    for (const s of list) {
      const f = root ? folderOf(s) : '';
      const arr = folders.get(f) ?? [];
      arr.push(s);
      folders.set(f, arr);
    }
    const sortedFolders = [...folders.entries()].sort(([a], [b]) => a.localeCompare(b));
    return {
      root,
      title: root ? root.name : 'Added files',
      folders: sortedFolders.map(([folder, arr]) => ({
        key: root ? `${root.id}/${folder}` : LOOSE,
        rootId: root?.id ?? null,
        folder,
        samples: arr.slice().sort((a, b) => (a.rel_path ?? a.name).localeCompare(b.rel_path ?? b.name)),
      })),
      count: list.length,
      analysed: list.filter((s) => s.analysis_status === 'done').length,
      missing: list.filter((s) => s.analysis_status === 'missing').length,
    };
  };
  for (const root of roots) {
    const list = byRoot.get(root.id) ?? [];
    if (list.length === 0 && filter.trim()) continue;
    groups.push(build(root, list));
  }
  const loose = byRoot.get(null) ?? [];
  if (loose.length > 0) groups.push(build(null, loose));
  return groups;
}
