"""§8.4 export: full mix, stems, kit and MIDI from synthetic renders."""

import json

import mido
import numpy as np
import pytest
import soundfile as sf

from patine.export.exporter import ExportJob, export_progression, render_slot
from patine.models import (
    ConformDeltas,
    ModuleSetting,
    Pattern,
    Progression,
    Region,
    Set,
    Slot,
    Step,
    TonalMetrics,
    TonalSettings,
)
from patine.theory import Chord, chord_midi_notes

SR = 48000
BPM = 120.0


def write_render(path, seconds: float, freq: float, sr: int = SR, channels: int = 2):
    t = np.arange(int(seconds * sr)) / sr
    y = np.stack([np.sin(2 * np.pi * freq * t)] * channels) * 0.3
    sf.write(str(path), y.T.astype(np.float32), sr, subtype="FLOAT")
    return str(path)


def make_regions() -> dict[str, Region]:
    return {
        "ra": Region(id="ra", sample_id="s", start_s=0.0, end_s=2.0, chord_root="C", chord_quality="min",
                     beats=[0.0, 0.5, 1.0, 1.5], metrics=TonalMetrics(wetness=0.5)),
        "rb": Region(id="rb", sample_id="s", start_s=2.0, end_s=4.0, chord_root="F", chord_quality="min",
                     metrics=TonalMetrics(wetness=0.1)),
    }


def make_set(**kw) -> Set:
    base = dict(id="set1", name="My set", anchor_region_id="ra", session_bpm=BPM, created_at="now")
    base.update(kw)
    return Set(**base)


def make_progression(pattern_b: Pattern | None = None) -> Progression:
    slots = [
        Slot(id="s0", progression_id="p", idx=0, chord_symbol="Cm", length_beats=4.0, region_id="ra"),
        Slot(id="s1", progression_id="p", idx=1, chord_symbol="Fm", length_beats=4.0, region_id="rb",
             pattern=pattern_b or Pattern(mode="retrigger", division="1/8",
                                          steps=[Step(gate=0.5, velocity=1.0), Step(gate=0.5, velocity=0.5)])),
        Slot(id="s2", progression_id="p", idx=2, chord_symbol="G7", length_beats=2.0, region_id=None),
    ]
    return Progression(id="p", set_id="set1", name="Test Prog", key="C minor", created_at="now", slots=slots)


@pytest.fixture
def job_parts(tmp_path):
    ra = write_render(tmp_path / "ra.wav", 2.0, 220.0)              # exactly 4 beats at 120
    rb = write_render(tmp_path / "rb.wav", 1.0, 330.0, sr=44100)    # short + other rate
    renders = {"s0": ra, "s1": rb}
    deltas = {"s0": ConformDeltas().model_dump(), "s1": ConformDeltas(loudness_db=1.5).model_dump()}
    return renders, deltas, tmp_path / "out"


def test_full_mix_length_and_content(job_parts):
    renders, deltas, out_dir = job_parts
    job = ExportJob(kind="full", progression=make_progression(), set_=make_set(), regions=make_regions(),
                    renders=renders, deltas=deltas, out_dir=out_dir, sample_rate=SR, seed=1)
    files = export_progression(job)
    assert len(files) == 1 and files[0].exists()
    y, sr = sf.read(str(files[0]), dtype="float32", always_2d=True)
    assert sr == SR
    assert y.shape == (int(10 * 0.5 * SR), 2)          # 4 + 4 + 2 beats at 120
    beat = int(0.5 * SR)
    assert np.abs(y[: 4 * beat]).max() > 0.1          # slot 0 plays
    assert np.abs(y[8 * beat:]).max() == 0.0            # unassigned slot is silence
    # retrigger slot: 1/8 steps, gate 0.5 → sound in the first half of each step, silence after
    step = beat // 2
    s1 = y[4 * beat: 8 * beat]
    for k in range(8):
        on = s1[k * step + 100: k * step + step // 2 - 100]
        off = s1[k * step + step // 2 + 100: (k + 1) * step - 10]
        assert np.abs(on).max() > 0.05
        assert np.abs(off).max() == 0.0
    # velocity 0.5 on every second step
    assert np.abs(s1[step + 200: step + 2000]).max() == pytest.approx(0.5 * np.abs(s1[200:2000]).max(), rel=0.2)


def test_full_mix_loops_and_bus(job_parts):
    renders, deltas, out_dir = job_parts
    tonal = TonalSettings(space=ModuleSetting(enabled=True, amount=100))
    job = ExportJob(kind="full", progression=make_progression(), set_=make_set(tonal_settings=tonal),
                    regions=make_regions(), renders=renders, deltas=deltas, out_dir=out_dir,
                    sample_rate=SR, loops=2, seed=1)
    files = export_progression(job)
    y, sr = sf.read(str(files[0]), dtype="float32", always_2d=True)
    assert y.shape[0] == 2 * int(10 * 0.5 * SR)
    beat = int(0.5 * SR)
    # Space: only the drier region (rb, slot 1) gets a send, so the empty slot 2 now carries its tail
    assert np.abs(y[8 * beat: 9 * beat]).max() > 1e-4
    # Density off → no tail into the silence, unchanged lengths
    tonal2 = TonalSettings(density=ModuleSetting(enabled=False))
    job2 = ExportJob(kind="full", progression=make_progression(), set_=make_set(tonal_settings=tonal2),
                     regions=make_regions(), renders=renders, deltas=deltas, out_dir=out_dir,
                     sample_rate=SR, seed=1)
    y2, _ = sf.read(str(export_progression(job2)[0]), dtype="float32", always_2d=True)
    assert np.abs(y2[8 * beat:]).max() == 0.0


def test_hold_truncates_and_never_loops():
    rng = np.random.default_rng(0)
    render = (rng.standard_normal((2, SR)) * 0.2).astype(np.float32)   # 1 s = 2 beats at 120
    slot = Slot(id="s", progression_id="p", idx=0, chord_symbol="C", length_beats=4.0, region_id="r")
    out = render_slot(render, slot, 0.5, SR, rng, 2)
    assert out.shape == (2, 2 * SR)
    assert np.abs(out[:, SR:]).max() == 0.0              # silence follows, no loop
    assert np.allclose(out[:, 200:SR - 200], render[:, 200:SR - 200])
    assert out[0, 0] == 0.0                              # declick ramp
    short = Slot(id="s", progression_id="p", idx=0, chord_symbol="C", length_beats=1.0, region_id="r")
    out2 = render_slot(render, short, 0.5, SR, rng, 2)
    assert out2.shape == (2, SR // 2)
    assert out2[0, -1] == pytest.approx(0.0, abs=1e-6) or abs(out2[0, -1]) < abs(render[0, SR // 2 - 1])


def test_retrigger_probability_offset_and_inactive():
    render = np.zeros((1, SR), dtype=np.float32)
    render[0, : SR // 4] = 1.0            # the first half beat is "hot", the rest silent
    render[0, SR // 2:] = 0.5             # from beat 1 on: 0.5
    pat = Pattern(mode="retrigger", division="1/4",
                  steps=[Step(), Step(active=False), Step(probability=0.0), Step(offset_beats=1.0)])
    slot = Slot(id="s", progression_id="p", idx=0, chord_symbol="C", length_beats=4.0, region_id="r", pattern=pat)
    rng = np.random.default_rng(0)
    out = render_slot(render, slot, 0.5, SR, rng, 1)
    beat = SR // 2
    assert np.abs(out[0, 100: beat // 2 - 100]).max() == pytest.approx(1.0)
    assert np.abs(out[0, beat: 2 * beat]).max() == 0.0            # inactive
    assert np.abs(out[0, 2 * beat: 3 * beat]).max() == 0.0        # probability 0
    assert np.abs(out[0, 3 * beat + 100: 4 * beat - 100]).max() == pytest.approx(0.5)  # offset 1 beat
    # seeded probability is reproducible
    pat2 = Pattern(mode="retrigger", division="1/16", steps=[Step(probability=0.5)])
    slot2 = Slot(id="s", progression_id="p", idx=0, chord_symbol="C", length_beats=4.0, region_id="r", pattern=pat2)
    a = render_slot(render, slot2, 0.5, SR, np.random.default_rng(7), 1)
    b = render_slot(render, slot2, 0.5, SR, np.random.default_rng(7), 1)
    assert np.array_equal(a, b)
    hits = [np.abs(a[0, k * (beat // 4) + 100: k * (beat // 4) + 400]).max() > 0.1 for k in range(16)]
    assert 0 < sum(hits) < 16


def test_stems_are_full_length_and_aligned(job_parts):
    renders, deltas, out_dir = job_parts
    job = ExportJob(kind="stems", progression=make_progression(), set_=make_set(), regions=make_regions(),
                    renders=renders, deltas=deltas, out_dir=out_dir, sample_rate=SR, seed=1)
    files = export_progression(job)
    assert len(files) == 3
    total = int(10 * 0.5 * SR)
    beat = int(0.5 * SR)
    for i, f in enumerate(files):
        y, sr = sf.read(str(f), dtype="float32", always_2d=True)
        assert y.shape[0] == total and sr == SR
        if i == 0:
            assert np.abs(y[: 4 * beat]).max() > 0.1 and np.abs(y[4 * beat:]).max() == 0.0
        if i == 1:
            assert np.abs(y[: 4 * beat]).max() == 0.0 and np.abs(y[4 * beat: 8 * beat]).max() > 0.05
        if i == 2:
            assert np.abs(y).max() == 0.0
    assert "slot00_Cm" in files[0].name and "slot01_Fm" in files[1].name


def test_kit_copies_renders_and_writes_json(job_parts):
    renders, deltas, out_dir = job_parts
    job = ExportJob(kind="kit", progression=make_progression(), set_=make_set(), regions=make_regions(),
                    renders=renders, deltas=deltas, out_dir=out_dir, sample_rate=SR)
    files = export_progression(job)
    names = sorted(f.name for f in files)
    assert names == ["00_Cm.wav", "01_Fm.wav", "patine.json"]
    # copies are byte-identical to the conformed renders (source rate kept)
    assert (out_dir / "Test_Prog_kit" / "01_Fm.wav").read_bytes() == open(renders["s1"], "rb").read()
    meta = json.loads((out_dir / "Test_Prog_kit" / "patine.json").read_text())
    assert meta["tempo"] == BPM and meta["key"] == "C minor"
    assert [s["chord"] for s in meta["slots"]] == ["Cm", "Fm", "G7"]
    s0 = meta["slots"][0]
    assert s0["file"] == "00_Cm.wav" and s0["deltas"]["loudness_db"] == 0.0
    assert s0["beats"] == [{"beat": k, "time_s": k * 0.5} for k in range(5)]
    assert meta["slots"][1]["deltas"]["loudness_db"] == 1.5
    assert meta["slots"][1]["beats"][-1]["time_s"] == 1.0           # 1 s render → beats 0..2
    assert meta["slots"][2]["file"] is None and meta["slots"][2]["beats"] == []


def test_midi_notes_lengths_and_tempo(job_parts):
    renders, deltas, out_dir = job_parts
    job = ExportJob(kind="midi", progression=make_progression(), set_=make_set(), regions=make_regions(),
                    renders=renders, deltas=deltas, out_dir=out_dir, loops=2)
    files = export_progression(job)
    assert files[0].suffix == ".mid"
    mid = mido.MidiFile(str(files[0]))
    assert len(mid.tracks) == 1
    tempos = [m for m in mid.tracks[0] if m.type == "set_tempo"]
    assert mido.tempo2bpm(tempos[0].tempo) == pytest.approx(BPM)
    tpb = mid.ticks_per_beat
    ons: list[tuple[int, int]] = []   # (abs ticks, note)
    offs: list[tuple[int, int]] = []
    t = 0
    for m in mid.tracks[0]:
        t += m.time
        if m.type == "note_on" and m.velocity > 0:
            ons.append((t, m.note))
        elif m.type in ("note_off", "note_on"):
            offs.append((t, m.note))
    cm = chord_midi_notes(Chord(0, "min"))
    fm = chord_midi_notes(Chord(5, "min"))
    g7 = chord_midi_notes(Chord(7, "7"))
    expected_pass = [(0, n) for n in cm] + [(4 * tpb, n) for n in fm] + [(8 * tpb, n) for n in g7]
    pass_len = 10 * tpb
    expected = expected_pass + [(t + pass_len, n) for t, n in expected_pass]
    assert ons == expected
    assert sorted(offs)[: len(cm)] == sorted((4 * tpb, n) for n in cm)
    assert max(t for t, _ in offs) == 2 * pass_len
    assert cm == [48, 60, 63, 67]


def test_unknown_chord_rests_in_midi(tmp_path):
    prog = make_progression()
    prog.slots[0].chord_symbol = "???"
    job = ExportJob(kind="midi", progression=prog, set_=make_set(), regions={}, renders={}, out_dir=tmp_path)
    mid = mido.MidiFile(str(export_progression(job)[0]))
    first_on = next(m for m in mid.tracks[0] if m.type == "note_on")
    assert first_on.time == 4 * mid.ticks_per_beat


def test_missing_session_bpm_falls_back(tmp_path):
    ra = write_render(tmp_path / "ra.wav", 2.0, 220.0)
    job = ExportJob(kind="full", progression=make_progression(), set_=make_set(session_bpm=None),
                    regions=make_regions(), renders={"s0": ra}, out_dir=tmp_path, sample_rate=22050)
    y, sr = sf.read(str(export_progression(job)[0]), dtype="float32", always_2d=True)
    assert sr == 22050 and y.shape[0] == int(10 * 0.5 * 22050)
