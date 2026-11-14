"""ml-stems bar features and engine wiring, with the models replaced by fakes."""
from __future__ import annotations

import argparse
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from autohotcue import _allin1, _raveform, _stems, analysis, cli
from autohotcue.backends import BeatAnalysis, Segment, StructureAnalysis
from autohotcue.gridlock import GridFit
from autohotcue.stempolicy import BarFeatures

SR = analysis.SR


def _fit(ok=True, bpm=120.0, render=120.0, anchor=0.5) -> GridFit:
    return GridFit(bpm=bpm, render_bpm=render, anchor_s=anchor, beat_fit=0.0,
                   bar_resid_std=0.0, splice_jump=0.0, ok=ok, reason="" if ok else "no fit")


def test_bar_edges_follow_the_grid():
    edges = _stems.bar_edges(np.zeros(0), np.zeros(0), _fit(), 10.6)
    assert edges == pytest.approx([0.5, 2.5, 4.5, 6.5, 8.5, 10.5])


def test_bar_edges_use_djay_tempo_when_the_fit_runs_at_double():
    edges = _stems.bar_edges(np.zeros(0), np.zeros(0), _fit(bpm=160.0, render=80.0, anchor=0.0), 9.5)
    assert edges == pytest.approx([0.0, 3.0, 6.0, 9.0])


def test_bar_edges_keep_a_first_bar_that_starts_just_before_the_audio():
    edges = _stems.bar_edges(np.zeros(0), np.zeros(0), _fit(anchor=1.997), 8.5)
    assert edges == pytest.approx([0.0, 1.997, 3.997, 5.997, 7.997])


def test_bar_edges_drop_a_first_bar_that_is_cut_short():
    edges = _stems.bar_edges(np.zeros(0), np.zeros(0), _fit(anchor=1.9), 8.5)
    assert edges == pytest.approx([1.9, 3.9, 5.9, 7.9])


def test_bars_survive_snapping_when_the_fit_is_slightly_off_djay_tempo():
    """Eight bars must stay 32 beats once apply has snapped both ends to the fitted grid."""
    from autohotcue.gridlock import snap_cues

    fit = _fit(bpm=124.09, render=124.0, anchor=0.1)
    edges = _stems.bar_edges(np.zeros(0), np.zeros(0), fit, 420.0)
    beat = 60.0 / fit.bpm
    assert len(edges) > 200
    for bar in range(8, len(edges) - 1):
        snapped = snap_cues({"C": edges[bar - 8], "D": edges[bar]}, fit)
        assert (snapped["D"] - snapped["C"]) / beat == pytest.approx(32.0, abs=1e-6)
        assert snapped["D"] == pytest.approx(edges[bar], abs=1e-6)


def test_bar_edges_without_a_grid_take_four_beats_per_bar():
    beats = 0.25 + 0.5 * np.arange(40)
    # A downbeat every two beats, as beat_this sometimes reports, and most of them on beat 2.
    downbeats = np.concatenate([beats[2::4], beats[4:12:4]])
    edges = _stems.bar_edges(beats, np.sort(downbeats), _fit(ok=False), 20.5)
    assert edges == pytest.approx(beats[2::4])


def test_bar_edges_without_a_grid_keep_the_last_whole_bar():
    beats = np.arange(96) * 0.5
    edges = _stems.bar_edges(beats, beats[::4], None, 48.0)
    assert len(edges) - 1 == 24
    assert edges[-1] == pytest.approx(48.0)


def test_bar_edges_without_a_grid_drop_a_bar_the_audio_cuts_short():
    beats = np.arange(96) * 0.5
    edges = _stems.bar_edges(beats, beats[::4], None, 47.0)
    assert len(edges) - 1 == 23


def _drifting_beats(count=80):
    """Beats that slow down all along, so that no straight lattice holds them."""
    return np.cumsum(0.5 * 1.004 ** np.arange(count))


def test_bar_edges_of_a_steady_track_ignore_doubled_and_missing_beats():
    lattice = 0.25 + 0.5 * np.arange(120)
    # The tracker ran at double tempo for a while, then lost the beat for 5 beats.
    doubled = lattice[20:30] + 0.25
    beats = np.sort(np.concatenate([np.delete(lattice, np.arange(60, 65)), doubled]))
    edges = _stems.bar_edges(beats, lattice[1::4], _fit(ok=False, bpm=97.3), 60.25)
    assert edges == pytest.approx(0.75 + 2.0 * np.arange(30))


def _rushed(lattice: np.ndarray, start: int, beats: int, spacing: float) -> np.ndarray:
    """The tracker left the lattice for ``beats`` beats and reported one every ``spacing`` s."""
    span = lattice[start + beats] - lattice[start]
    stray = lattice[start] + spacing * np.arange(1, round(span / spacing))
    return np.concatenate([lattice[:start + 1], stray, lattice[start + beats:]])


def test_bar_edges_of_a_steady_track_ignore_a_stretch_of_stray_beats():
    lattice = 0.25 + 0.5 * np.arange(200)
    # 20 beats where the music has 14: bars counted on tracked beats would slip by 6 beats.
    beats = _rushed(lattice, 80, 14, 0.35)
    edges = _stems.bar_edges(beats, lattice[::4], _fit(ok=False), 100.5)
    assert edges == pytest.approx(0.25 + 2.0 * np.arange(51), abs=0.02)


def test_bar_edges_of_a_steady_track_find_the_tempo_despite_slightly_fast_beats():
    lattice = 0.25 + 0.5 * np.arange(200)
    # 28 beats where the music has 24: close enough to the tempo to pull the mean gap
    # out of reach of the search.
    beats = _rushed(lattice, 80, 24, 0.42)
    edges = _stems.bar_edges(beats, lattice[::4], _fit(ok=False), 100.5)
    assert edges == pytest.approx(0.25 + 2.0 * np.arange(51), abs=0.02)


def test_stray_beats_do_not_double_the_tempo_for_the_rest_of_the_track():
    lattice = 0.25 + 0.5 * np.arange(400)
    # Stray beats fast enough to make the next real gap look like two beats.
    beats = _rushed(lattice, 80, 14, 0.31)
    even = _stems._even_beats(beats)
    assert even[even > 52.0] == pytest.approx(lattice[lattice > 52.0])
    edges = _stems.bar_edges(beats, lattice[::4], _fit(ok=False), 200.5)
    assert edges == pytest.approx(0.25 + 2.0 * np.arange(101), abs=0.02)


def test_bar_edges_of_a_steady_track_find_the_tempo_despite_20_ms_steps_and_a_slow_stretch():
    lattice = 0.3 + 60.0 / 140.0 * np.arange(400)
    # The median gap of beats reported in 20 ms steps is 0.42 s, almost 143 BPM,
    # and 24 beats where the music has 40 pull the mean gap the other way.
    beats = np.round(_rushed(lattice, 200, 40, 0.7) / 0.02) * 0.02
    edges = _stems.bar_edges(beats, lattice[::4], _fit(ok=False), 172.0)
    assert edges == pytest.approx(0.3 + 4 * 60.0 / 140.0 * np.arange(101), abs=0.02)


def test_bar_edges_of_a_steady_track_find_the_tempo_when_no_gap_is_the_usual_one():
    # Every fifth beat comes early: the usual gap is 0.52 s, the tempo is 120 BPM.
    index = np.arange(200)
    beats = 0.5 * index + 0.02 * (index % 5)
    edges = _stems.bar_edges(beats, beats[::4], _fit(ok=False), 100.0)
    assert edges == pytest.approx(0.04 + 2.0 * np.arange(50), abs=0.03)


def test_bar_edges_of_a_steady_track_with_two_kinds_of_gaps():
    # Short and long gaps in turn, none within 10% of the median one.
    gaps = np.tile([0.42, 0.58], 200)
    gaps[[79, 80]] = gaps[[80, 79]]
    beats = 0.2 + np.concatenate([[0.0], np.cumsum(gaps)])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        edges = _stems.bar_edges(beats, beats[::4], _fit(ok=False), 202.2)
    assert edges == pytest.approx(0.16 + 2.0 * np.arange(102), abs=0.03)


def test_a_missing_beat_is_filled_in_after_the_tempo_went_up():
    full = np.concatenate([0.5 * np.arange(60), 30.0 + 0.42 * np.arange(40)])
    assert _stems._even_beats(np.delete(full, 80)) == pytest.approx(full)


def test_slower_beats_after_stray_ones_are_not_taken_for_missing_beats():
    steady = 0.5 * np.arange(40)
    stray = steady[-1] + 0.31 * np.arange(1, 9)
    slower = stray[-1] + 0.65 * np.arange(1, 21)
    beats = np.concatenate([steady, stray, slower])
    assert _stems._even_beats(beats) == pytest.approx(beats)


def test_a_wrong_count_of_missing_beats_does_not_repeat_itself():
    # A stray beat at 0.32 s, then a stretch tracked at half tempo: the gap after the stray
    # beat is filled with one beat too many, and the next gaps must not follow.
    beats = np.concatenate([[0.0, 0.32], np.arange(2.0, 31.0), np.arange(30.5, 100.0, 0.5)])
    even = _stems._even_beats(beats)
    assert even[even >= 2.0] == pytest.approx(np.arange(2.0, 100.0, 0.5))
    edges = _stems.bar_edges(beats, np.arange(0.0, 100.0, 2.0), _fit(ok=False), 100.0)
    assert edges == pytest.approx(2.0 * np.arange(51), abs=0.02)


def test_missing_beats_are_filled_in_after_a_long_gap_at_the_start():
    full = np.concatenate([0.5 * np.arange(120), 60.0 + 0.6 * np.arange(80)])
    beats = np.concatenate([full[[0, 20]], full[22:]])
    assert _stems._even_beats(beats) == pytest.approx(full)


def test_missing_beats_are_filled_in_all_along_a_stretch_tracked_at_half_tempo():
    full = np.concatenate([0.5 * np.arange(80), 40.0 + 0.6 * np.arange(80)])
    beats = np.delete(full, np.arange(113, 156, 2))
    assert _stems._even_beats(beats) == pytest.approx(full)


def test_bar_edges_of_a_steady_track_are_even_though_beats_come_in_20_ms_steps():
    beats = np.round((0.3 + 60.0 / 128.0 * np.arange(256)) / 0.02) * 0.02
    edges = _stems.bar_edges(beats, beats[::4], _fit(ok=False), 121.0)
    assert len(edges) == 65
    assert np.diff(edges) == pytest.approx(4 * 60.0 / 128.0, abs=1e-3)


def test_bar_edges_follow_the_tracked_beats_when_the_tempo_is_not_steady():
    beats = _drifting_beats()
    edges = _stems.bar_edges(beats, beats[1::4], _fit(ok=False), 60.0)
    assert edges[:-1] == pytest.approx(beats[1::4])


def test_bar_edges_follow_the_tracked_beats_after_a_splice():
    """A fraction of a beat was added in the middle: one lattice holds either half, not both."""
    beats = 0.5 * np.arange(120.0)
    beats[60:] += 0.2
    edges = _stems.bar_edges(beats, beats[::4], _fit(ok=False), 61.0)
    assert edges[:-1] == pytest.approx(beats[::4])


def test_bar_edges_skip_a_doubled_beat():
    beats = _drifting_beats()
    doubled = np.insert(beats, 41, (beats[40] + beats[41]) / 2)
    edges = _stems.bar_edges(doubled, beats[1::4], _fit(ok=False), 60.0)
    assert edges[:-1] == pytest.approx(beats[1::4])


def test_bar_edges_fill_in_missing_beats():
    beats = _drifting_beats()
    edges = _stems.bar_edges(np.delete(beats, [31, 32, 33]), beats[1::4], _fit(ok=False), 60.0)
    assert edges[:-1] == pytest.approx(beats[1::4], abs=0.01)


def test_missing_beats_are_counted_at_the_tempo_around_them():
    """120 BPM, then 100: three beats lost in the slow part leave room for four gaps, not five."""
    beats = np.concatenate([0.5 * np.arange(80), 40.0 + 0.6 * np.arange(80)])
    edges = _stems.bar_edges(np.delete(beats, [140, 141, 142]), beats[::4], None, 88.0)
    assert edges[:-1] == pytest.approx(beats[::4])


def test_missing_beats_are_filled_in_evenly():
    beats = 0.5 * np.arange(40.0)
    assert _stems._even_beats(np.delete(beats, [10, 11, 12, 13, 30])) == pytest.approx(beats)


def test_bar_edges_without_beats():
    assert len(_stems.bar_edges(np.zeros(3), np.zeros(1), None, 30.0)) == 0


def test_bar_levels_are_rms_per_bar():
    sr = _stems.FRAME * 10
    y = np.concatenate([np.full(sr, 0.5), np.zeros(sr), np.full(sr, -0.25)])
    levels = _stems.bar_levels(_stems.frame_power(y), np.array([0.0, 1.0, 2.0, 3.0]), sr)
    assert levels == pytest.approx([0.5, 0.0, 0.25])


def test_bar_timbre_cuts_each_bar_in_sixteen_steps():
    mel = np.tile(np.arange(40, dtype=np.float32), (3, 1))
    timbre = _stems.bar_timbre(mel, np.array([0.0, 16.0, 32.0]))
    assert timbre.shape == (2, 16 * 3)
    assert timbre[0].reshape(16, 3)[:, 0] == pytest.approx(np.arange(16))
    assert timbre[1].reshape(16, 3)[:, 2] == pytest.approx(16 + np.arange(16))


def _fake_models(monkeypatch, stems, segments, samplerate=SR):
    calls = []
    separator = SimpleNamespace(samplerate=samplerate)
    model = SimpleNamespace(cfg="cfg")
    monkeypatch.setattr(_raveform, "resolve_raveform_weights", lambda: "weights")
    monkeypatch.setattr(_allin1, "_prepare", lambda weights, name: (separator, model))
    monkeypatch.setattr(_allin1, "_separate", lambda path, sep: stems)
    monkeypatch.setattr(_allin1, "_spectrogram", lambda s, sep: "spec")
    monkeypatch.setattr(_allin1, "_probabilities", lambda spec, m: (None, "sections", "functions"))
    monkeypatch.setattr(_allin1, "_clear_mlx_cache", lambda: calls.append("cleared"))
    monkeypatch.setattr(
        _raveform, "raveform_segments",
        lambda sections, functions, cfg, duration: segments,
    )
    return calls


def _stereo(level: np.ndarray, bar_samples: int) -> np.ndarray:
    mono = np.repeat(level, bar_samples).astype(np.float32)
    return np.stack([mono, mono])


def test_analyze_stems_measures_each_stem_per_bar(monkeypatch):
    bar = 2 * SR  # 120 BPM
    drums = np.array([1.0, 1.0, 0.0, 0.5])
    vocals = np.array([0.0, 0.2, 0.2, 0.0])
    stems = {
        "drums": _stereo(drums, bar), "bass": _stereo(drums * 0.5, bar),
        "other": _stereo(np.zeros(4), bar), "vocals": _stereo(vocals, bar),
    }
    segments = [Segment(0.0, 4.0, "intro"), Segment(4.0, 8.0, "drop")]
    calls = _fake_models(monkeypatch, stems, segments)
    y = np.repeat(drums, bar).astype(np.float32)
    beat = BeatAnalysis(bpm=120.0, beats=np.zeros(0), downbeats=np.zeros(0), duration_s=8.0,
                        source="test")

    feats, structure = _stems.analyze_stems("/x.m4a", y, SR, beat, _fit(anchor=0.0))

    # One frame straddles each bar line, so a silent bar picks up a little of its neighbor.
    close = dict(abs=0.06)
    assert feats.starts == pytest.approx([0.0, 2.0, 4.0, 6.0])
    assert feats.drums == pytest.approx(drums, **close)
    assert feats.bass == pytest.approx(drums * 0.5, **close)
    assert feats.other == pytest.approx(np.zeros(4))
    assert feats.vocals == pytest.approx(vocals, **close)
    assert feats.mix == pytest.approx(drums, **close)
    assert feats.timbre.shape == (4, _stems.BAR_SLOTS * _stems.MEL_BANDS)
    assert structure.segments == segments
    assert calls == ["cleared"]


def test_analyze_stems_refuses_a_sample_rate_mismatch(monkeypatch):
    _fake_models(monkeypatch, {}, [], samplerate=48000)
    beat = BeatAnalysis(bpm=120.0, beats=np.zeros(0), downbeats=np.zeros(0), duration_s=8.0,
                        source="test")
    with pytest.raises(RuntimeError, match="48000"):
        _stems.analyze_stems("/x.m4a", np.zeros(SR, dtype=np.float32), SR, beat, _fit())


def test_engine_is_the_default_and_runs_alone():
    assert analysis.DEFAULT_ENGINE == "ml-stems"
    assert "ml-stems" in cli._ENGINE_CHOICES
    assert analysis.effective_parallel_jobs("ml-stems", 8) == 1
    assert not analysis.is_cpu_dsp_engine("ml-stems")
    parser = argparse.ArgumentParser()
    cli._add_engine_arg(parser)
    assert parser.parse_args([]).engine == "ml-stems"


def test_cues_from_beats_keeps_the_layout_of_the_policy(monkeypatch):
    """The pre-drop rule of the other engines must not move pad C here."""
    level = np.concatenate([np.full(16, 0.1), np.ones(32), np.full(16, 0.1)])
    feats = BarFeatures(
        starts=np.arange(64) * 2.0, drums=level, bass=level, other=level * 0.3,
        vocals=np.zeros(64), mix=level, timbre=np.ones((64, 8), dtype=np.float32),
    )
    structure = StructureAnalysis(
        [Segment(0.0, 32.0, "intro"), Segment(32.0, 96.0, "drop"), Segment(96.0, 128.0, "outro")],
        source="raveform",
    )
    monkeypatch.setattr(_stems, "analyze_stems", lambda *args: (feats, structure))
    beat = BeatAnalysis(bpm=100.0, beats=np.arange(0, 128, 0.5), downbeats=np.arange(0, 128, 2.0),
                        duration_s=128.0, source="test")

    track, prop = analysis.cues_from_beats(
        np.zeros(SR, dtype=np.float32), "/x.m4a", beat, known_bpm=120.0, engine="ml-stems",
        pre_drop_bars=16,
    )

    assert track.engine == "ml-stems"
    assert track.bars is feats
    assert track.segments == structure.segments
    assert prop.positions == {"A": 16.0, "B": 32.0, "C": 80.0, "D": 96.0}
    assert [prop.label(pad) for pad in "ABCD"] == [
        "8 Bars to Drop", "Drop", "8 Bars to Break", "Breakdown",
    ]
    assert "djay says 120.0, tracked 100.0" in prop.notes


def test_other_engines_keep_the_djcues_names():
    prop = analysis.CueProposal(positions={"A": 0.0, "D": 30.0})
    assert prop.label("A") == "First Beat"
    assert prop.label("D") == "Drop"
