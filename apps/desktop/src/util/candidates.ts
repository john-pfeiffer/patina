import type { Candidate, Strictness } from '../api/types';

/** Cache key for the candidate drawer: one list per slot and strictness. */
export function candidateKey(slotId: string, strictness: Strictness): string {
  return `${slotId}:${strictness}`;
}

/** Handoff §5.3–5.4: substitutions (no transposition) rank before any
 * transposition, then by score ascending; ties keep the sidecar's order. */
export function orderCandidates(candidates: readonly Candidate[]): Candidate[] {
  return candidates
    .map((c, i) => ({ c, i }))
    .sort((a, b) => {
      const ta = a.c.transpose_semitones === 0 ? 0 : 1;
      const tb = b.c.transpose_semitones === 0 ? 0 : 1;
      if (ta !== tb) return ta - tb;
      if (a.c.score !== b.c.score) return a.c.score - b.c.score;
      return a.i - b.i;
    })
    .map(({ c }) => c);
}
