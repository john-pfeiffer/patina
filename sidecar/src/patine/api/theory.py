"""Chord parsing for the UI (never 4xx: bad input is `ok=false`)."""

from __future__ import annotations

from fastapi import APIRouter

from ..models import ParsedChord, ParseRequest, ParseResponse
from ..theory import PC_NAMES_SHARP, TheoryError, parse_key, parse_progression_text

router = APIRouter(tags=["theory"])


@router.post("/theory/parse", response_model=ParseResponse)
def parse(body: ParseRequest) -> ParseResponse:
    try:
        key = parse_key(body.key) if body.key else None
        specs = parse_progression_text(body.text, key)
    except TheoryError as e:
        return ParseResponse(ok=False, error=str(e))
    flats = key.prefer_flats if key else False
    chords = [
        ParsedChord(
            input=s.text,
            chord_symbol=s.chord.symbol(flats),
            roman=s.roman,
            root=PC_NAMES_SHARP[s.chord.root],
            quality=s.chord.quality,
            bass=PC_NAMES_SHARP[s.chord.bass] if s.chord.bass is not None else None,
            pitch_classes=sorted(s.chord.pitch_classes),
            length_beats=s.length_beats,
        )
        for s in specs
    ]
    return ParseResponse(ok=True, key=key.name() if key else None, chords=chords)
