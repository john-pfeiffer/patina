"""Candidate ranking, strictness and Reroll (handoff §5.2–5.5).

    score = transposition_cost + timbral_distance_from_anchor + (1 - chord_confidence)

Substitutions rank before transpositions: an untransposed region that
satisfies the strictness costs less than any repitched exact match, because a
full-mix region shifted 5 semitones is a vocal that's wrong."""

from __future__ import annotations

import random

from .models import Candidate, Region, Sample, TonalMetrics
from .theory import (
    PC_NAMES_SHARP,
    Chord,
    Key,
    Strictness,
    matches,
    substitution_label,
    transposition_semitones,
)

# §5.4 transposition policy: exact first, ±2 acceptable, beyond that ranked last.
SUBSTITUTION_COST = 0.25
TRANSPOSE_COST = {0: 0.0, 1: 0.6, 2: 0.8}
TRANSPOSE_COST_FAR = 2.0   # plus 0.5 per semitone beyond 2
MAX_TRANSPOSE = 6


def region_chord(region: Region) -> Chord | None:
    if region.chord_root is None or region.chord_quality is None:
        return None
    try:
        root = PC_NAMES_SHARP.index(region.chord_root)
    except ValueError:
        return None
    bass = PC_NAMES_SHARP.index(region.bass_note) if region.bass_note in PC_NAMES_SHARP else None
    return Chord(root, region.chord_quality, bass)


def transposition_cost(semitones: int) -> float:
    n = abs(semitones)
    if n in TRANSPOSE_COST:
        return TRANSPOSE_COST[n]
    return TRANSPOSE_COST_FAR + 0.5 * (n - 2)


def timbral_distance(a: TonalMetrics | None, b: TonalMetrics | None) -> float:
    """0 when identical, ~1 for clearly different recordings. Only the
    measurements the conform layer cannot fully fix count heavily: tilt shape,
    noise floor, crest and width; loudness is cheap to fix so it weighs little."""
    if a is None or b is None:
        return 0.0
    parts: list[float] = []
    if a.loudness_lufs is not None and b.loudness_lufs is not None:
        parts.append(min(1.0, abs(a.loudness_lufs - b.loudness_lufs) / 20.0) * 0.5)
    if a.tilt_bands and b.tilt_bands and len(a.tilt_bands) == len(b.tilt_bands):
        diff = [x - y for x, y in zip(a.tilt_bands, b.tilt_bands, strict=True)]
        mean = sum(diff) / len(diff)
        shape = sum(abs(d - mean) for d in diff) / len(diff)
        parts.append(min(1.0, shape / 6.0))
    if a.noise_floor_db is not None and b.noise_floor_db is not None:
        parts.append(min(1.0, abs(a.noise_floor_db - b.noise_floor_db) / 30.0))
    if a.crest_db is not None and b.crest_db is not None:
        parts.append(min(1.0, abs(a.crest_db - b.crest_db) / 12.0))
    if a.width_bands and b.width_bands:
        parts.append(min(1.0, sum(abs(x - y) for x, y in zip(a.width_bands, b.width_bands, strict=True)) / 1.5))
    if a.wetness is not None and b.wetness is not None:
        parts.append(min(1.0, abs(a.wetness - b.wetness)) * 0.5)
    return sum(parts) / max(1, len(parts))


def rank_candidates(
    target: Chord,
    key: Key | None,
    pool: list[tuple[Region, Sample]],
    anchor_metrics: TonalMetrics | None,
    strictness: Strictness = "function",
    *,
    limit: int = 50,
    allow_transposition: bool = True,
    exclude_samples: set[str] | None = None,
) -> list[Candidate]:
    """`exclude_samples`: source samples that may not be offered (the
    distinct-samples rule: they already play in another slot)."""
    out: list[Candidate] = []
    for region, sample in pool:
        if exclude_samples and sample.id in exclude_samples:
            continue
        chord = region_chord(region)
        if chord is None:
            continue
        conf = region.chord_confidence if region.chord_confidence is not None else 0.5
        confidence_cost = 1.0 - max(0.0, min(1.0, conf))
        timbre = timbral_distance(region.metrics, anchor_metrics)
        best: Candidate | None = None
        if matches(chord, target, key, strictness):
            exact = chord.root == target.root and chord.quality == target.quality
            t_cost = 0.0 if exact else SUBSTITUTION_COST
            best = Candidate(
                region=region, sample_name=sample.name, sample_id=sample.id,
                score=t_cost + timbre + confidence_cost,
                transposition_cost=t_cost, timbral_distance=timbre, confidence_cost=confidence_cost,
                transpose_semitones=0, label=substitution_label(chord, target, key),
            )
        if allow_transposition and chord.quality == target.quality and chord.root != target.root:
            n = transposition_semitones(chord.root, target.root)
            if abs(n) <= MAX_TRANSPOSE:
                t_cost = transposition_cost(n)
                cand = Candidate(
                    region=region, sample_name=sample.name, sample_id=sample.id,
                    score=t_cost + timbre + confidence_cost,
                    transposition_cost=t_cost, timbral_distance=timbre, confidence_cost=confidence_cost,
                    transpose_semitones=n, label=f"{n:+d} st",
                )
                if best is None or cand.score < best.score:
                    best = cand
        if best is not None:
            out.append(best)
    out.sort(key=lambda c: (c.score, c.region.start_s))
    return out[:limit]


class RerollChanges(dict[str, tuple[str, int] | None]):
    """slot_id → (region_id, transpose) for the slots that change, plus the
    warnings a relaxed distinct-samples rule produced."""

    def __init__(self) -> None:
        super().__init__()
        self.warnings: list[str] = []


def reroll(
    slots: list[tuple[str, Chord, str | None, bool]],
    key: Key | None,
    pool: list[tuple[Region, Sample]],
    anchor_metrics: TonalMetrics | None,
    strictness: Strictness,
    amount: float,
    *,
    seed: int | None = None,
    top_k: int = 8,
    unique_samples: bool = False,
    slot_labels: dict[str, str] | None = None,
) -> RerollChanges:
    """§5.5. `slots` = (slot_id, target chord, current region_id, locked), in
    progression order. Picks are weighted toward lower scores so a Reroll
    stays musical. With `unique_samples` no two slots end up on regions of
    the same source sample: slots that keep their region claim their sample
    first, then the changing slots are filled in order, each excluding the
    samples claimed so far; a slot with nothing left falls back to the full
    pool and a warning names it."""
    rng = random.Random(seed)
    changes = RerollChanges()
    unlocked = [s for s in slots if not s[3]]
    n_change = round(len(unlocked) * max(0.0, min(100.0, amount)) / 100.0)
    if n_change == 0:
        return changes
    chosen = rng.sample(unlocked, n_change) if n_change < len(unlocked) else unlocked
    order = {s[0]: i for i, s in enumerate(slots)}
    chosen.sort(key=lambda s: order[s[0]])
    sample_of = {r.id: smp.id for r, smp in pool}
    chosen_ids = {s[0] for s in chosen}
    used: set[str] = set()
    if unique_samples:
        for slot_id, _, current, _ in slots:
            if slot_id not in chosen_ids and current and current in sample_of:
                used.add(sample_of[current])
    labels = slot_labels or {}
    for slot_id, target, current, _ in chosen:
        exclude = used if unique_samples else None
        cands = rank_candidates(target, key, pool, anchor_metrics, strictness, limit=top_k + 1, exclude_samples=exclude)
        cands = [c for c in cands if c.region.id != current] or cands
        if not cands and unique_samples:
            cands = rank_candidates(target, key, pool, anchor_metrics, strictness, limit=top_k + 1)
            cands = [c for c in cands if c.region.id != current] or cands
            if cands:
                changes.warnings.append(
                    f"slot {labels.get(slot_id, slot_id)}: every candidate's sample already plays in another slot; reused one"
                )
        if not cands:
            continue
        weights = [1.0 / (1.0 + c.score) for c in cands]
        pick = rng.choices(cands, weights=weights, k=1)[0]
        changes[slot_id] = (pick.region.id, pick.transpose_semitones)
        if unique_samples:
            used.add(pick.sample_id or sample_of.get(pick.region.id, ""))
    return changes
