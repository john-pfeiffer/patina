// Display-only chord helpers. Chord *parsing* is the sidecar's job
// (POST /theory/parse); these only format what the sidecar already classified
// and measure semitone distances between named roots.

const QUALITY_SUFFIX: Record<string, string> = {
  maj: '',
  min: 'm',
  '7': '7',
  m7: 'm7',
  maj7: 'maj7',
  dim: 'dim',
  sus2: 'sus2',
  sus4: 'sus4',
  aug: 'aug',
  dim7: 'dim7',
  m7b5: 'm7b5',
  '6': '6',
  m6: 'm6',
};

const LETTER_PC: Record<string, number> = { C: 0, D: 2, E: 4, F: 5, G: 7, A: 9, B: 11 };

/** Pitch class 0..11 for a note name (`C#`, `Db`, `F`), or null. */
export function notePc(name: string | null | undefined): number | null {
  if (!name) return null;
  const letter = name[0]?.toUpperCase();
  if (letter === undefined || !(letter in LETTER_PC)) return null;
  let pc = LETTER_PC[letter] ?? 0;
  for (const acc of name.slice(1)) {
    if (acc === '#') pc += 1;
    else if (acc === 'b') pc -= 1;
    else return null;
  }
  return ((pc % 12) + 12) % 12;
}

/** Smallest signed semitone move from `fromRoot` to `toRoot`, in -6..+5 (0 when unknown). */
export function transposeBetween(fromRoot: string | null, toRoot: string | null): number {
  const a = notePc(fromRoot);
  const b = notePc(toRoot);
  if (a === null || b === null) return 0;
  const d = (((b - a) % 12) + 12) % 12;
  return d > 6 ? d - 12 : d;
}

export function chordSymbol(root: string | null, quality: string | null, bass?: string | null): string | null {
  if (!root || !quality) return null;
  let s = root + (QUALITY_SUFFIX[quality] ?? quality);
  if (bass && bass !== root) s += '/' + bass;
  return s;
}

export function transposeLabel(semitones: number): string {
  if (semitones === 0) return '';
  return `${semitones > 0 ? '+' : ''}${semitones} st`;
}
