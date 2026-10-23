import { describe, expect, it } from 'vitest';
import { chordSymbol, notePc, transposeBetween, transposeLabel } from './chords';

describe('notePc', () => {
  it('reads sharps and flats', () => {
    expect(notePc('C')).toBe(0);
    expect(notePc('C#')).toBe(1);
    expect(notePc('Db')).toBe(1);
    expect(notePc('Cb')).toBe(11);
    expect(notePc('B#')).toBe(0);
    expect(notePc('H')).toBeNull();
    expect(notePc('')).toBeNull();
    expect(notePc(null)).toBeNull();
  });
});

describe('transposeBetween', () => {
  it('prefers the shorter direction', () => {
    expect(transposeBetween('C', 'D')).toBe(2);
    expect(transposeBetween('C', 'B')).toBe(-1);
    expect(transposeBetween('C', 'F#')).toBe(6);
    expect(transposeBetween('F#', 'C')).toBe(6);
    expect(transposeBetween('A', 'A')).toBe(0);
    expect(transposeBetween(null, 'A')).toBe(0);
  });
});

describe('chordSymbol', () => {
  it('formats like the sidecar', () => {
    expect(chordSymbol('C', 'maj')).toBe('C');
    expect(chordSymbol('F', 'min', 'Ab')).toBe('Fm/Ab');
    expect(chordSymbol('G', '7', 'G')).toBe('G7');
    expect(chordSymbol('G', 'm7b5')).toBe('Gm7b5');
    expect(chordSymbol(null, 'maj')).toBeNull();
  });
});

describe('transposeLabel', () => {
  it('signs non-zero shifts', () => {
    expect(transposeLabel(0)).toBe('');
    expect(transposeLabel(3)).toBe('+3 st');
    expect(transposeLabel(-2)).toBe('-2 st');
  });
});
