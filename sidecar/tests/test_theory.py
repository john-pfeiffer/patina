"""Chord/key/roman parsing and the §5.3 strictness relationships."""

import pytest

from patine import theory as t


@pytest.mark.parametrize(
    "symbol,root,quality,bass",
    [
        ("Cm", 0, "min", None),
        ("F#m7", 6, "m7", None),
        ("Bbmaj7", 10, "maj7", None),
        ("Gsus4", 7, "sus4", None),
        ("Ddim", 2, "dim", None),
        ("C/E", 0, "maj", 4),
        ("Am7/G", 9, "m7", 7),
        ("E7", 4, "7", None),
        ("Bø7", 11, "m7b5", None),
        ("G+", 7, "aug", None),
        ("Dsus2", 2, "sus2", None),
        ("Ebm", 3, "min", None),
    ],
)
def test_parse_chord_symbol(symbol, root, quality, bass):
    c = t.parse_chord_symbol(symbol)
    assert (c.root, c.quality, c.bass) == (root, quality, bass)


def test_bad_chord_symbols():
    for s in ["Xm", "", "H7", "Cxyz"]:
        with pytest.raises(t.TheoryError):
            t.parse_chord_symbol(s)


@pytest.mark.parametrize(
    "text,tonic,mode",
    [("C minor", 0, "minor"), ("Cm", 0, "minor"), ("Am", 9, "minor"), ("F# major", 6, "major"), ("Bb", 10, "major"), ("Ebmaj", 3, "major")],
)
def test_parse_key(text, tonic, mode):
    k = t.parse_key(text)
    assert (k.tonic, k.mode) == (tonic, mode)


def test_roman_in_minor_follows_pop_convention():
    k = t.parse_key("C minor")
    cases = {"i": "Cm", "iv": "Fm", "V": "G", "V7": "G7", "bVII": "Bb", "VII": "Bb", "bVI": "Ab", "VI": "Ab",
             "vii°": "Bdim", "ii°": "Ddim", "III": "Eb", "v": "Gm"}
    for roman, symbol in cases.items():
        assert t.parse_roman(roman, k).symbol(True) == symbol, roman


def test_roman_in_major():
    k = t.parse_key("C major")
    cases = {"I": "C", "ii": "Dm", "iii": "Em", "IV": "F", "V7": "G7", "vi": "Am", "vii°": "Bdim", "bVII": "Bb", "IVmaj7": "Fmaj7"}
    for roman, symbol in cases.items():
        assert t.parse_roman(roman, k).symbol(True) == symbol, roman


def test_chord_to_roman_round_trip():
    k = t.parse_key("C minor")
    for roman in ["i", "iv", "V", "VI", "VII", "vii°", "ii°", "III"]:
        assert t.chord_to_roman(t.parse_roman(roman, k), k) == roman
    km = t.parse_key("C major")
    assert t.chord_to_roman(t.parse_chord_symbol("Ab"), km) == "bVI"
    assert t.chord_to_roman(t.parse_chord_symbol("G7"), km) == "V7"


def test_progression_text_mixed_and_lengths():
    k = t.parse_key("C minor")
    specs = t.parse_progression_text("i iv V:2 bVI | Cm, Fm", k)
    assert [s.chord.symbol(True) for s in specs] == ["Cm", "Fm", "G", "Ab", "Cm", "Fm"]
    assert [s.length_beats for s in specs] == [4, 4, 2, 4, 4, 4]
    assert specs[4].roman == "i"
    with pytest.raises(t.TheoryError):
        t.parse_progression_text("i iv V", None)


def test_strictness_levels():
    fm, ab, cm, bb, g, f = (t.parse_chord_symbol(s) for s in ["Fm", "Ab", "Cm", "Bb", "G", "F"])
    key = t.parse_key("F minor")
    assert t.matches(fm, fm, key, "exact")
    assert not t.matches(ab, fm, key, "exact")
    assert t.matches(ab, fm, key, "function")          # relative major
    assert t.substitution_label(ab, fm, key) == "relative"
    assert not t.matches(bb, fm, key, "function")      # iv is subdominant, tonic slot
    assert t.matches(ab, cm, None, "two_tones")        # Ab C Eb vs C Eb G
    assert not t.matches(g, fm, None, "two_tones")
    assert t.shared_tone_count(f, fm) == 2


def test_function_groups():
    k = t.parse_key("C major")
    assert t.function_of(t.parse_chord_symbol("G"), k) == "dominant"
    assert t.function_of(t.parse_chord_symbol("Bdim"), k) == "dominant"
    assert t.function_of(t.parse_chord_symbol("Dm"), k) == "subdominant"
    assert t.function_of(t.parse_chord_symbol("Am"), k) == "tonic"
    assert t.function_of(t.parse_chord_symbol("Ab"), k) is None
    km = t.parse_key("C minor")
    assert t.function_of(t.parse_chord_symbol("G"), km) == "dominant"
    assert t.function_of(t.parse_chord_symbol("Eb"), km) == "tonic"


def test_transposition_semitones():
    assert t.transposition_semitones(0, 7) == -5
    assert t.transposition_semitones(0, 2) == 2
    assert t.transposition_semitones(0, 11) == -1
    assert abs(t.transposition_semitones(0, 6)) == 6


def test_midi_notes_voicing():
    notes = t.chord_midi_notes(t.parse_chord_symbol("C/E"), octave=4)
    assert notes[0] == 52
    assert notes[1:] == [60, 64, 67]
