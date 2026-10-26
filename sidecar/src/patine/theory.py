"""Chords, keys, roman numerals and the relationships Patine's selection layer
needs (handoff §5.3–5.4, §8.1).

Pitch classes are ints 0..11 (C=0). Names are spelled with sharps internally;
`pc_name(pc, prefer_flats)` picks the spelling for display.

This module is the single source of truth for chord parsing. The UI never
parses chords itself; it asks the sidecar (POST /theory/parse)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

PC_NAMES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
PC_NAMES_FLAT = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
_LETTER_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

Quality = Literal["maj", "min", "7", "m7", "maj7", "dim", "sus2", "sus4", "5", "aug", "dim7", "m7b5", "6", "m6"]

# Intervals above the root. The first eight are the qualities the chord
# estimator decides between (handoff §4.5); "5" is reported for a bare root or
# root+fifth (a plucked note, a bass line, a power chord) where asserting a
# third would be a guess; the rest exist so typed progressions parse without
# surprises.
QUALITY_INTERVALS: dict[str, tuple[int, ...]] = {
    "maj": (0, 4, 7),
    "min": (0, 3, 7),
    "7": (0, 4, 7, 10),
    "m7": (0, 3, 7, 10),
    "maj7": (0, 4, 7, 11),
    "dim": (0, 3, 6),
    "sus2": (0, 2, 7),
    "sus4": (0, 5, 7),
    "5": (0, 7),
    "aug": (0, 4, 8),
    "dim7": (0, 3, 6, 9),
    "m7b5": (0, 3, 6, 10),
    "6": (0, 4, 7, 9),
    "m6": (0, 3, 7, 9),
}

ANALYSIS_QUALITIES: tuple[str, ...] = ("maj", "min", "7", "m7", "maj7", "dim", "sus2", "sus4")
# Everything a region can carry: the estimator's qualities plus the bare root.
REPORTED_QUALITIES: tuple[str, ...] = (*ANALYSIS_QUALITIES, "5")

QUALITY_SUFFIX: dict[str, str] = {
    "maj": "", "min": "m", "7": "7", "m7": "m7", "maj7": "maj7", "dim": "dim",
    "sus2": "sus2", "sus4": "sus4", "5": "5", "aug": "aug", "dim7": "dim7", "m7b5": "m7b5",
    "6": "6", "m6": "m6",
}

# Accepted spellings for each quality, longest first so "maj7" wins over "maj".
_QUALITY_TOKENS: list[tuple[str, str]] = sorted(
    [
        ("maj7", "maj7"), ("Maj7", "maj7"), ("M7", "maj7"), ("Δ7", "maj7"), ("Δ", "maj7"), ("ma7", "maj7"),
        ("maj9", "maj7"), ("M9", "maj7"),
        ("m7b5", "m7b5"), ("ø7", "m7b5"), ("ø", "m7b5"), ("min7b5", "m7b5"), ("-7b5", "m7b5"),
        ("dim7", "dim7"), ("°7", "dim7"), ("o7", "dim7"),
        ("dim", "dim"), ("°", "dim"), ("o", "dim"),
        ("min7", "m7"), ("m7", "m7"), ("-7", "m7"), ("m9", "m7"), ("min9", "m7"),
        ("min6", "m6"), ("m6", "m6"),
        ("min", "min"), ("m", "min"), ("-", "min"),
        ("sus2", "sus2"), ("sus4", "sus4"), ("sus", "sus4"),
        ("aug", "aug"), ("+", "aug"),
        ("dom7", "7"), ("7", "7"), ("9", "7"), ("11", "7"), ("13", "7"),
        ("maj", "maj"), ("Maj", "maj"), ("M", "maj"), ("major", "maj"), ("6", "6"),
        ("add9", "maj"), ("add2", "maj"), ("5", "5"),
    ],
    key=lambda kv: -len(kv[0]),
)

MAJOR_SCALE = (0, 2, 4, 5, 7, 9, 11)
MINOR_SCALE = (0, 2, 3, 5, 7, 8, 10)  # natural minor; harmonic 7th allowed in is_diatonic
ROMAN_NUMERALS = ("I", "II", "III", "IV", "V", "VI", "VII")

# Harmonic function groups by scale degree index (0-based). Same in both modes
# for the degrees Patine cares about (handoff §5.3 "same function").
FUNCTION_GROUPS: dict[str, tuple[int, ...]] = {
    "tonic": (0, 2, 5),
    "subdominant": (1, 3),
    "dominant": (4, 6),
}

Strictness = Literal["exact", "function", "two_tones"]


class TheoryError(ValueError):
    pass


@dataclass(frozen=True)
class Chord:
    root: int
    quality: str
    bass: int | None = None

    @property
    def pitch_classes(self) -> frozenset[int]:
        return frozenset((self.root + i) % 12 for i in QUALITY_INTERVALS[self.quality])

    @property
    def triad(self) -> frozenset[int]:
        return frozenset((self.root + i) % 12 for i in QUALITY_INTERVALS[self.quality][:3])

    @property
    def is_minor_like(self) -> bool:
        return self.quality in ("min", "m7", "dim", "dim7", "m7b5", "m6")

    def symbol(self, prefer_flats: bool = False) -> str:
        s = pc_name(self.root, prefer_flats) + QUALITY_SUFFIX[self.quality]
        if self.bass is not None and self.bass != self.root:
            s += "/" + pc_name(self.bass, prefer_flats)
        return s


@dataclass(frozen=True)
class Key:
    tonic: int
    mode: Literal["major", "minor"]

    @property
    def scale(self) -> tuple[int, ...]:
        base = MAJOR_SCALE if self.mode == "major" else MINOR_SCALE
        return tuple((self.tonic + d) % 12 for d in base)

    @property
    def prefer_flats(self) -> bool:
        # Keys on the flat side of the circle of fifths spell with flats.
        rel_major = self.tonic if self.mode == "major" else (self.tonic + 3) % 12
        return rel_major in (5, 10, 3, 8, 1, 6)  # F Bb Eb Ab Db Gb

    @property
    def relative(self) -> Key:
        if self.mode == "major":
            return Key((self.tonic + 9) % 12, "minor")
        return Key((self.tonic + 3) % 12, "major")

    def name(self) -> str:
        return f"{pc_name(self.tonic, self.prefer_flats)} {self.mode}"


# --- names -------------------------------------------------------------------

def pc_name(pc: int, prefer_flats: bool = False) -> str:
    return (PC_NAMES_FLAT if prefer_flats else PC_NAMES_SHARP)[pc % 12]


_NOTE_RE = re.compile(r"^([A-Ga-g])([#♯b♭]*)")


def parse_note(text: str) -> tuple[int, str]:
    """Parse a note name at the start of `text`. Returns (pitch class, rest)."""
    m = _NOTE_RE.match(text)
    if not m:
        raise TheoryError(f"not a note name: {text!r}")
    pc = _LETTER_PC[m.group(1).upper()]
    for acc in m.group(2):
        pc += 1 if acc in "#♯" else -1
    return pc % 12, text[m.end():]


# --- chord symbols -----------------------------------------------------------

def parse_chord_symbol(text: str) -> Chord:
    """`Cm`, `F#m7`, `Bbmaj7`, `Gsus4`, `Ddim`, `C/E`, `Am7/G`, `N` (no chord raises)."""
    s = text.strip()
    if not s:
        raise TheoryError("empty chord symbol")
    if s[0] not in "ABCDEFG":
        raise TheoryError(f"chord symbol must start with a note letter A–G: {text!r}")
    root, rest = parse_note(s)
    bass: int | None = None
    if "/" in rest:
        rest, bass_text = rest.split("/", 1)
        bass, tail = parse_note(bass_text.strip())
        if tail.strip():
            raise TheoryError(f"unexpected text after bass note: {text!r}")
    rest = rest.strip()
    quality = "maj"
    if rest:
        for token, q in _QUALITY_TOKENS:
            if rest.startswith(token):
                quality = q
                rest = rest[len(token):]
                break
        else:
            raise TheoryError(f"unknown chord quality {rest!r} in {text!r}")
        # Tolerate trailing extensions we fold away (e.g. "7#9", "7b9", "add9").
        if rest and not re.fullmatch(r"[\(\)\s#♯b♭0-9addsus+-]*", rest):
            raise TheoryError(f"unknown chord suffix {rest!r} in {text!r}")
    return Chord(root, quality, bass)


# --- keys ----------------------------------------------------------------------

_KEY_MODE_TOKENS = [
    ("minor", "minor"), ("min", "minor"), ("m", "minor"), ("aeolian", "minor"), ("-", "minor"),
    ("major", "major"), ("maj", "major"), ("M", "major"), ("ionian", "major"),
]


def parse_key(text: str) -> Key:
    """`C minor`, `Cm`, `Am`, `F# major`, `Bb`, `Ebmaj`."""
    s = text.strip()
    if not s:
        raise TheoryError("empty key")
    tonic, rest = parse_note(s)
    rest = rest.strip()
    mode: Literal["major", "minor"] = "major"
    if rest:
        for token, m in sorted(_KEY_MODE_TOKENS, key=lambda kv: -len(kv[0])):
            if rest == token or rest.lower() == token.lower() and token not in ("m", "M"):
                mode = m
                break
        else:
            raise TheoryError(f"unknown key mode {rest!r} in {text!r}")
    return Key(tonic, mode)


# --- roman numerals --------------------------------------------------------------

_ROMAN_RE = re.compile(r"^([b#♭♯]?)(vii|vi|iv|v|iii|ii|i)(.*)$", re.IGNORECASE)


def looks_like_roman(token: str) -> bool:
    t = token.strip()
    if not t:
        return False
    if t[0] in "ABCDEFG":
        return False
    return bool(_ROMAN_RE.match(t))


def parse_roman(text: str, key: Key) -> Chord:
    """`i`, `iv`, `V7`, `bVII`, `vii°`, `IVmaj7`, `ii7`, `Vsus4`. Case decides the
    triad quality unless a suffix says otherwise."""
    s = text.strip()
    m = _ROMAN_RE.match(s)
    if not m:
        raise TheoryError(f"not a roman numeral: {text!r}")
    acc, numeral, suffix = m.group(1), m.group(2), m.group(3).strip()
    degree = ROMAN_NUMERALS.index(numeral.upper())
    base = MAJOR_SCALE if key.mode == "major" else MINOR_SCALE
    root = (key.tonic + base[degree]) % 12
    if acc and not (key.mode == "minor" and acc in "b♭" and degree in (2, 5, 6)):
        # In minor, III/VI/VII already sit a semitone below their major-scale
        # positions, so the pop-harmony spelling "bVI bVII" is redundant, not a
        # further lowering.
        root = (root + (1 if acc in "#♯" else -1)) % 12
    upper = numeral.isupper()
    quality = "maj" if upper else "min"
    if key.mode == "minor" and degree == 6 and not acc and re.match(r"^(°|o|dim|ø)", suffix):
        root = (key.tonic + 11) % 12  # vii° in minor is the leading-tone chord
    if suffix:
        for token, q in _QUALITY_TOKENS:
            if suffix.startswith(token):
                quality = q
                suffix = suffix[len(token):]
                break
        else:
            raise TheoryError(f"unknown roman suffix {suffix!r} in {text!r}")
        if quality == "7" and not upper:
            quality = "m7"
        if quality == "maj7" and not upper:
            quality = "m7"  # "imaj7" is rare; keep the lowercase meaning minor
        if quality == "6" and not upper:
            quality = "m6"
        if suffix and not re.fullmatch(r"[\(\)\s#♯b♭0-9addsus+-]*", suffix):
            raise TheoryError(f"unknown roman suffix {suffix!r} in {text!r}")
    return Chord(root, quality)


def chord_to_roman(chord: Chord, key: Key) -> str:
    """Best-effort roman label for display. Chromatic roots get a b/# prefix."""
    base = MAJOR_SCALE if key.mode == "major" else MINOR_SCALE
    rel = (chord.root - key.tonic) % 12
    acc = ""
    if key.mode == "minor" and rel == 11:
        degree = 6
        acc = "" if chord.quality in ("dim", "dim7", "m7b5") else "#"
    elif rel in base:
        degree = base.index(rel)
    elif (rel + 1) % 12 in base:
        degree = base.index((rel + 1) % 12)
        acc = "b"
    else:
        degree = base.index((rel - 1) % 12)
        acc = "#"
    numeral = ROMAN_NUMERALS[degree]
    if chord.is_minor_like:
        numeral = numeral.lower()
    suffix = {
        "maj": "", "min": "", "7": "7", "m7": "7", "maj7": "maj7", "dim": "°", "dim7": "°7",
        "m7b5": "ø7", "sus2": "sus2", "sus4": "sus4", "5": "5", "aug": "+", "6": "6", "m6": "6",
    }[chord.quality]
    return f"{acc}{numeral}{suffix}"


# --- relationships -----------------------------------------------------------------

def shared_tone_count(a: Chord, b: Chord) -> int:
    return len(a.pitch_classes & b.pitch_classes)


def transposition_semitones(from_pc: int, to_pc: int) -> int:
    """Smallest signed shift taking `from_pc` to `to_pc`, in [-6, 6]."""
    d = (to_pc - from_pc) % 12
    return d - 12 if d > 6 else d


def degree_of(chord: Chord, key: Key) -> int | None:
    scale = key.scale
    return scale.index(chord.root) if chord.root in scale else None


def is_diatonic(chord: Chord, key: Key) -> bool:
    """Triad tones inside the key. Minor keys also accept the raised 7th so V and
    vii° count as diatonic."""
    allowed = set(key.scale)
    if key.mode == "minor":
        allowed.add((key.tonic + 11) % 12)
    return chord.triad <= allowed


def function_of(chord: Chord, key: Key) -> str | None:
    d = degree_of(chord, key)
    if d is None or not is_diatonic(chord, key):
        return None
    for name, degrees in FUNCTION_GROUPS.items():
        if d in degrees:
            return name
    return None


def is_relative(a: Chord, b: Chord) -> bool:
    """Relative major/minor pair: C maj ↔ A min (and their 7th-chord forms)."""
    if a.is_minor_like == b.is_minor_like:
        return False
    major, minor = (a, b) if not a.is_minor_like else (b, a)
    return (major.root + 9) % 12 == minor.root and major.quality in ("maj", "maj7", "6") and minor.quality in ("min", "m7", "m6")


def is_bare_root_match(candidate: Chord, target: Chord) -> bool:
    """A bare root or power chord ("5") on the same root fits any chord on that
    root: it asserts no third, so it is neither right nor wrong about one."""
    return candidate.root == target.root and "5" in (candidate.quality, target.quality)


def matches(candidate: Chord, target: Chord, key: Key | None, strictness: Strictness) -> bool:
    """Does `candidate` satisfy a slot asking for `target` at this strictness?"""
    if candidate.root == target.root and candidate.quality == target.quality:
        return True
    if strictness == "exact":
        return False
    if is_bare_root_match(candidate, target):
        return True
    if strictness == "function":
        if is_relative(candidate, target):
            return True
        if key is not None:
            ft, fc = function_of(target, key), function_of(candidate, key)
            return ft is not None and ft == fc
        return False
    return shared_tone_count(candidate, target) >= 2


def substitution_label(candidate: Chord, target: Chord, key: Key | None) -> str:
    if candidate.root == target.root and candidate.quality == target.quality:
        return "exact"
    if is_bare_root_match(candidate, target):
        return "bare root"
    if is_relative(candidate, target):
        return "relative"
    if key is not None:
        f = function_of(candidate, key)
        if f is not None and f == function_of(target, key):
            return f"same function ({f})"
    n = shared_tone_count(candidate, target)
    return f"shares {n} tone{'s' if n != 1 else ''}"


# --- progression text ------------------------------------------------------------

_SPLIT_RE = re.compile(r"[\s|,]+")


@dataclass(frozen=True)
class SlotSpec:
    text: str
    chord: Chord
    roman: str | None
    length_beats: float


def parse_progression_text(text: str, key: Key | None, default_length: float = 4.0) -> list[SlotSpec]:
    """`Cm Fm G` or `i iv V` (needs a key). An optional `:N` sets beats, e.g. `Fm:2`."""
    specs: list[SlotSpec] = []
    for raw in _SPLIT_RE.split(text.strip()):
        if not raw:
            continue
        length = default_length
        tok = raw
        if ":" in tok:
            tok, n = tok.rsplit(":", 1)
            try:
                length = float(n)
            except ValueError as e:
                raise TheoryError(f"bad slot length in {raw!r}") from e
        if looks_like_roman(tok):
            if key is None:
                raise TheoryError(f"roman numeral {tok!r} needs a key")
            chord = parse_roman(tok, key)
            specs.append(SlotSpec(raw, chord, tok, length))
        else:
            chord = parse_chord_symbol(tok)
            roman = chord_to_roman(chord, key) if key is not None else None
            specs.append(SlotSpec(raw, chord, roman, length))
    return specs


def chord_midi_notes(chord: Chord, octave: int = 4) -> list[int]:
    """Close voicing from the root in `octave`, bass note an octave below when set."""
    base = 12 * (octave + 1) + chord.root
    notes = [base + i for i in QUALITY_INTERVALS[chord.quality]]
    bass = chord.bass if chord.bass is not None else chord.root
    notes.insert(0, 12 * octave + bass)
    return notes
