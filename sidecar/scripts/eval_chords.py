#!/usr/bin/env python
"""Measure chord recall over a folder of real samples.

    uv run --directory sidecar python scripts/eval_chords.py ~/Splice/sounds --out after.csv
    uv run --directory sidecar python scripts/eval_chords.py ~/Splice/sounds --legacy --out before.csv
    uv run --directory sidecar python scripts/eval_chords.py ~/Splice/sounds --out after.csv --baseline before.csv

Walks every audio file under the folder (recursively), runs the analysis
pipeline and writes one CSV row per file: path, seconds, number of regions,
first chord, mean confidence, whether the whole-span fallback produced the
region, and analysis time. The summary prints the share of files with at
least one region overall and by duration bucket, the share recovered by the
fallback, confidence quartiles and timing. `--legacy` approximates the
pipeline before the recall work (no harmonic chroma, 0.55 no-chord floor, no
observation window, no fallback). `--baseline` diffs against an earlier CSV
and lists the files that gained or lost regions and the chords that changed.
`--dump-frames FILE` prints per-frame gate values for one file, which is how
the gate constants are calibrated on real material."""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from patine.analysis import chords  # noqa: E402
from patine.analysis.chroma import compute_chroma  # noqa: E402
from patine.analysis.decode import decode  # noqa: E402
from patine.analysis.pipeline import AnalysisOptions, analyze_decoded  # noqa: E402
from patine.audio import is_audio_file  # noqa: E402

BUCKETS = (("<2 s", 0.0, 2.0), ("2-8 s", 2.0, 8.0), (">8 s", 8.0, float("inf")))


def options(args: argparse.Namespace) -> AnalysisOptions:
    if args.legacy:
        return AnalysisOptions(beat_tracker="librosa", hpss=False, no_chord_floor=0.55, window_beats=0.0,
                               expected_chord_beats=0.0, fallback=False)
    return AnalysisOptions(
        beat_tracker="librosa",
        hpss=not args.no_hpss,
        no_chord_floor=args.floor,
        window_beats=args.window_beats,
        fallback=not args.no_fallback,
    )


def walk(folder: Path, limit: int | None) -> list[Path]:
    files = sorted(p for p in folder.rglob("*") if p.is_file() and is_audio_file(p) and not p.name.startswith("._"))
    return files[:limit] if limit else files


def analyse(path: Path, opts: AnalysisOptions) -> dict:
    t0 = time.perf_counter()
    try:
        audio = decode(path)
        result = analyze_decoded(audio, options=opts, name=path.name)
    except Exception as e:  # noqa: BLE001 - one bad file must not stop the run
        return {"path": str(path), "seconds": "", "regions": "", "first_chord": "", "mean_confidence": "",
                "recovered": "", "analysis_s": round(time.perf_counter() - t0, 3), "error": f"{type(e).__name__}: {e}"}
    regions = result.regions
    first = regions[0] if regions else None
    return {
        "path": str(path),
        "seconds": round(result.duration, 3),
        "regions": len(regions),
        "first_chord": f"{first.chord_root}{'' if first.chord_quality == 'maj' else first.chord_quality}" if first and first.chord_root else "",
        "mean_confidence": round(statistics.fmean(r.chord_confidence for r in regions), 3) if regions else "",
        "recovered": int(any(r.recovered for r in regions)),
        "analysis_s": round(time.perf_counter() - t0, 3),
        "error": "",
    }


def summarise(rows: list[dict]) -> None:
    ok = [r for r in rows if r["error"] == "" and r["seconds"] != ""]
    if not ok:
        print("no files analysed")
        return
    with_regions = [r for r in ok if int(r["regions"]) > 0]
    print(f"files: {len(rows)}  analysed: {len(ok)}  errors: {len(rows) - len(ok)}")
    print(f"with >= 1 region: {len(with_regions)}/{len(ok)} ({100.0 * len(with_regions) / len(ok):.1f} %)")
    for name, lo, hi in BUCKETS:
        b = [r for r in ok if lo <= float(r["seconds"]) < hi]
        if b:
            n = sum(1 for r in b if int(r["regions"]) > 0)
            print(f"  {name:6s} {n}/{len(b)} ({100.0 * n / len(b):.1f} %)")
    rec = sum(1 for r in with_regions if int(r["recovered"]) == 1)
    print(f"recovered by the whole-span fallback: {rec}/{len(with_regions)}")
    confs = sorted(float(r["mean_confidence"]) for r in with_regions if r["mean_confidence"] != "")
    if confs:
        q = [confs[int(len(confs) * f)] for f in (0.25, 0.5, 0.75)]
        print(f"confidence quartiles: {q[0]:.2f} / {q[1]:.2f} / {q[2]:.2f}")
    times = sorted(float(r["analysis_s"]) for r in ok)
    print(f"analysis seconds per file: p50 {times[len(times) // 2]:.2f}  p95 {times[int(len(times) * 0.95)]:.2f}")


def diff(rows: list[dict], baseline_path: Path) -> None:
    with open(baseline_path, newline="", encoding="utf-8") as f:
        base = {r["path"]: r for r in csv.DictReader(f)}
    gained, lost, changed = [], [], []
    for r in rows:
        b = base.get(r["path"])
        if b is None or r["regions"] == "" or b["regions"] == "":
            continue
        before, after = int(b["regions"]), int(r["regions"])
        if before == 0 and after > 0:
            gained.append(r)
        elif before > 0 and after == 0:
            lost.append(r)
        elif before and after and b["first_chord"] != r["first_chord"]:
            changed.append((b["first_chord"], r))
    print(f"\nagainst {baseline_path.name}: gained {len(gained)}, lost {len(lost)}, first chord changed {len(changed)}")
    for r in gained:
        print(f"  + {r['path']}  →  {r['first_chord']} ({r['regions']} region(s){', fallback' if r['recovered'] == 1 else ''})")
    for r in lost:
        print(f"  - {r['path']}")
    for before, r in changed:
        print(f"  ~ {r['path']}  {before} → {r['first_chord']}")


def dump_frames(path: Path, opts: AnalysisOptions) -> None:
    audio = decode(path)
    feats = compute_chroma(audio.mono_a, audio.sr_a, hpss=opts.hpss)
    backend = chords.TemplateBackend(no_chord_floor=opts.no_chord_floor)
    sims = backend.similarities(feats)
    silence, no_structure = backend.gates(feats)
    print("frame   time  best_sim  no_chord  silence  no_structure  best_state")
    for i in range(feats.n_frames):
        best = int(sims[:chords.N_CHORD_STATES, i].argmax())
        root, q = chords.state_label(best)
        print(f"{i:5d} {feats.times[i]:6.2f}  {sims[best, i]:.3f}     {sims[chords.NO_CHORD, i]:.3f}    "
              f"{silence[i]:.2f}     {no_structure[i]:.2f}         {root}{q}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path)
    ap.add_argument("--out", type=Path, help="CSV to write (one row per file)")
    ap.add_argument("--baseline", type=Path, help="earlier CSV to diff against")
    ap.add_argument("--limit", type=int, help="only the first N files")
    ap.add_argument("--legacy", action="store_true", help="approximate the pipeline before the recall work")
    ap.add_argument("--no-hpss", action="store_true")
    ap.add_argument("--no-fallback", action="store_true")
    ap.add_argument("--floor", type=float, default=chords.NO_CHORD_FLOOR)
    ap.add_argument("--window-beats", type=float, default=1.0)
    ap.add_argument("--dump-frames", type=Path, help="print per-frame gate values for this one file and exit")
    args = ap.parse_args()
    opts = options(args)
    if args.dump_frames:
        dump_frames(args.dump_frames, opts)
        return 0
    files = walk(args.folder, args.limit)
    if not files:
        print(f"no audio files under {args.folder}")
        return 1
    rows: list[dict] = []
    for i, path in enumerate(files, 1):
        row = analyse(path, opts)
        rows.append(row)
        status = row["error"] or f"{row['regions']} region(s) {row['first_chord']}{' (fallback)' if row['recovered'] == 1 else ''}"
        print(f"[{i}/{len(files)}] {path.name}: {status}", flush=True)
    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.out}")
    print()
    summarise(rows)
    if args.baseline:
        diff(rows, args.baseline)
    return 0


if __name__ == "__main__":
    sys.exit(main())
