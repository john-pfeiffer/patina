"""Export contract.

    files = export_progression(ExportJob(...))

`renders` maps region_id → path of that region's *ready* conformed WAV for the
progression's set. The API layer guarantees every assigned region has one
before calling. Full and stem renders reproduce the UI playback model (§8.2–8.3):
hard cuts with 1–2 ms declick ramps, hold = region stretched to the slot,
retrigger = restart from the snap point on the division with per-step gate,
velocity, probability and offset; the set bus (Density comp; Space send when
on) applies to the full mix only."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from ..models import Progression, Region, Set


@dataclass
class ExportJob:
    kind: Literal["full", "stems", "kit", "midi"]
    progression: Progression
    set_: Set
    regions: dict[str, Region]
    renders: dict[str, str]                   # region_id → wav path
    deltas: dict[str, dict] = field(default_factory=dict)  # region_id → ConformDeltas dict
    out_dir: Path = Path(".")
    loops: int = 1
    sample_rate: int = 48000
    seed: int | None = None                   # for probability steps


def export_progression(job: ExportJob) -> list[Path]:
    raise NotImplementedError
