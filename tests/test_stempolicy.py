"""ml-stems cue placement on made-up tracks."""
from __future__ import annotations

import numpy as np

from autohotcue.backends import Segment
from autohotcue.stempolicy import (
    BarFeatures,
    _loop_moments,
    _Track,
    propose_cues_stems,
    section_runs,
)

BAR_S = 2.0


def S(label, bars, drums=1.0, bass=1.0, other=0.3, vocals=0.0, pattern=None):
    return dict(label=label, bars=bars, drums=drums, bass=bass, other=other, vocals=vocals,
                pattern=pattern)


def _track(*sections) -> tuple[BarFeatures, list[Segment]]:
    """Each section holds its levels for its whole length and repeats an 8-bar phrase of its own."""
    rng = np.random.default_rng(0)
    phrases: dict[object, np.ndarray] = {}
    levels = {name: [] for name in ("drums", "bass", "other", "vocals")}
    timbre, segments, bar = [], [], 0
    for i, sec in enumerate(sections):
        key = i if sec["pattern"] is None else sec["pattern"]
        phrase = phrases.setdefault(key, rng.normal(size=(8, 64)))
        for k in range(sec["bars"]):
            for name in levels:
                levels[name].append(sec[name])
            timbre.append(phrase[k % 8] + 0.05 * rng.normal(size=64))
        if sec["label"]:
            segments.append(Segment(bar * BAR_S, (bar + sec["bars"]) * BAR_S, sec["label"]))
        elif segments:
            last = segments[-1]
            segments[-1] = Segment(last.start, (bar + sec["bars"]) * BAR_S, last.label)
        bar += sec["bars"]
    arrays = {name: np.array(values) for name, values in levels.items()}
    feats = BarFeatures(
        starts=np.arange(bar) * BAR_S,
        mix=np.sqrt(sum(a * a for a in arrays.values())),
        timbre=np.array(timbre, dtype=np.float32),
        **arrays,
    )
    return feats, segments


def _bars(prop) -> dict[str, int]:
    return {pad: round(t / BAR_S) for pad, t in prop.positions.items()}


def _pads(prop) -> dict[str, tuple[str, int]]:
    bars = _bars(prop)
    return {pad: (prop.labels[pad], bars[pad]) for pad in bars}


CLASSIC = (
    S("intro", 16, bass=0.0, other=0.0),
    S("drop", 32),
    S("breakdown", 16, drums=0.1, bass=0.0),
    S("drop", 32),
    S("outro", 16, bass=0.0, other=0.0),
)


def test_classic_track():
    prop = propose_cues_stems(*_track(*CLASSIC))
    assert prop.positions["A"] == 8 * BAR_S and prop.positions["B"] == 16 * BAR_S
    pads = _pads(prop)
    assert pads["A"] == ("8 Bars to Drop", 8)
    assert pads["B"] == ("Drop", 16)
    assert pads["C"] == ("8 Bars to Break", 40)
    assert pads["D"] == ("Breakdown", 48)
    assert pads["H"] == ("Drop 2", 64)
    assert pads["E"][0] == "Loop"
    assert prop.notes == []


def test_loop_is_a_repeated_phrase_away_from_the_other_cues():
    prop = propose_cues_stems(*_track(*CLASSIC))
    bars = _bars(prop)
    # A phrase of a drop that the next 8 bars repeat, not on top of B, C, D or the second drop.
    assert bars["E"] in (24, 32, 72, 80)
    assert all(abs(bars["E"] - bars[pad]) >= 4 for pad in "ABCDH")


def test_at_most_two_loops_and_never_the_same_phrase_twice():
    track = (
        S("intro", 16, bass=0.0, other=0.0),
        S("drop", 48, pattern="x"),
        S("breakdown", 16, drums=0.1, bass=0.0),
        S("drop", 48, pattern="y"),
        S("drop", 32, pattern="x"),
        S("outro", 16, bass=0.0, other=0.0),
    )
    prop = propose_cues_stems(*_track(*track))
    loops = sorted(bar for label, bar in _pads(prop).values() if label == "Loop")
    assert len(loops) == 2
    # One from each phrase: the first drop and the end share theirs.
    assert sum(80 <= bar < 128 for bar in loops) == 1


def test_no_loop_when_a_stem_changes_under_the_same_drums():
    """Bars 24 to 32 sound like the next 8, but their bassline is only half as loud."""
    track = (
        S("intro", 16, bass=0.0, other=0.0),
        S("drop", 8, pattern="x"),
        S("", 8, bass=0.5, pattern="x"),
        S("", 16, pattern="x"),
        S("breakdown", 16, drums=0.1, bass=0.0),
        S("outro", 8, drums=0.1, bass=0.0),
    )
    loops = [m.bar for m in _loop_moments(_Track(*_track(*track)))]
    assert 32 in loops
    assert 24 not in loops


def test_drop_moves_back_to_where_the_groove_came_in():
    track = (
        S("intro", 16, bass=0.0, other=0.0),
        S("buildup", 8, other=0.0),
        S("drop", 32),
        S("breakdown", 16, drums=0.1, bass=0.0),
        S("drop", 32),
        S("outro", 16, bass=0.0, other=0.0),
    )
    pads = _pads(propose_cues_stems(*_track(*track)))
    assert pads["A"][1] == 8
    assert pads["B"][1] == 16
    # The labeled drop itself is not reported as a second drop.
    assert pads["H"] == ("Drop 2", 72)


def test_groove_from_bar_one_keeps_the_labeled_drop():
    track = (
        S("buildup", 24),
        S("drop", 40),
        S("breakdown", 16, drums=0.1, bass=0.0),
        S("drop", 32),
        S("outro", 16, bass=0.0, other=0.0),
    )
    prop = propose_cues_stems(*_track(*track))
    assert _bars(prop)["B"] == 24
    assert any("weak drop" in note for note in prop.notes)


def test_track_opening_on_a_drop_uses_the_stems():
    track = (
        S("drop", 8, bass=0.0, other=0.0),
        S("", 56),
        S("breakdown", 16, drums=0.1, bass=0.0),
        S("drop", 32),
        S("outro", 16, bass=0.0, other=0.0),
    )
    pads = _pads(propose_cues_stems(*_track(*track)))
    assert pads["B"][1] == 8
    assert pads["A"][1] == 0
    assert pads["H"] == ("Drop 2", 80)


def test_track_opening_on_silence_finds_where_the_groove_starts():
    track = (S("drop", 8, drums=0.0, bass=0.0, other=0.0), S("", 24), S("outro", 8, bass=0.0))
    pads = _pads(propose_cues_stems(*_track(*track)))
    assert pads["B"][1] == 8
    assert pads["A"][1] == 0


VERSE = dict(other=0.3, vocals=0.0)
CHORUS = dict(other=0.8, vocals=0.5)


def test_track_without_a_drop_gets_its_fullest_passage_on_b():
    # A song that plays its beat from the first bar to the last: nothing ever arrives.
    track = (S("drop", 24, **VERSE), S("", 16, **CHORUS), S("", 24, **VERSE))
    prop = propose_cues_stems(*_track(*track))
    pads = _pads(prop)
    assert pads["B"] == ("Peak", 24)
    assert pads["A"] == ("8 Bars to Peak", 16)
    assert "B: no drop found; using the fullest passage" in prop.notes


def test_track_that_never_changes_still_gets_b():
    prop = propose_cues_stems(*_track(S("drop", 64)))
    assert _pads(prop)["B"] == ("Peak", 0)
    assert "A" not in prop.positions
    assert "A: B is only 0 bars in" in prop.notes


def test_fullest_passage_starts_where_the_level_goes_up():
    # The second half of the chorus is a little fuller than the first.
    track = (
        S("drop", 24, **VERSE),
        S("", 8, **CHORUS),
        S("", 8, other=0.9, vocals=0.5),
        S("", 24, **VERSE),
    )
    assert _pads(propose_cues_stems(*_track(*track)))["B"] == ("Peak", 24)


def test_fullest_passage_lasts_8_bars():
    # One bar where everything is louder is an accent, not a passage.
    track = (
        S("drop", 24, **VERSE),
        S("", 16, **CHORUS),
        S("", 16, **VERSE),
        S("", 1, other=1.2, vocals=0.75),
        S("", 23, **VERSE),
    )
    assert _pads(propose_cues_stems(*_track(*track)))["B"] == ("Peak", 24)


def test_fullest_passage_is_not_one_that_only_comes_close():
    # The first chorus rises more, from a thin verse, but the second one is clearly fuller.
    track = (
        S("drop", 16, other=0.1, vocals=0.0),
        S("", 16, other=0.7, vocals=0.4),
        S("", 16, other=0.8, vocals=0.0),
        S("", 16, other=0.8, vocals=0.5),
        S("", 16, **VERSE),
    )
    assert _pads(propose_cues_stems(*_track(*track)))["B"] == ("Peak", 48)


def test_groove_coming_in_is_the_drop_when_no_section_is_called_one():
    track = (S("intro", 12, drums=0.0, bass=0.0), S("breakdown", 52), S("outro", 16, bass=0.0))
    prop = propose_cues_stems(*_track(*track))
    pads = _pads(prop)
    assert pads["B"] == ("Drop", 12)
    assert pads["A"] == ("8 Bars to Drop", 4)
    assert not any("no drop found" in note for note in prop.notes)


def test_drop_too_early_for_pad_a():
    track = (S("intro", 4, bass=0.0), S("drop", 60), S("outro", 16, bass=0.0, other=0.0))
    prop = propose_cues_stems(*_track(*track))
    assert _bars(prop)["B"] == 4
    assert "A" not in prop.positions
    assert "A: B is only 4 bars in" in prop.notes


def test_breakdown_with_vocals_is_not_the_way_out():
    track = (
        S("intro", 16, bass=0.0, other=0.0),
        S("drop", 32),
        S("breakdown", 16, drums=0.1, bass=0.0, vocals=0.8),
        S("drop", 32),
        S("breakdown", 16, drums=0.1, bass=0.0),
        S("drop", 32),
        S("outro", 16, bass=0.0, other=0.0),
    )
    prop = propose_cues_stems(*_track(*track))
    pads = _pads(prop)
    assert pads["D"][1] == 96
    assert pads["C"][1] == 88
    assert pads["F"] == ("Vocals", 48)
    assert pads["H"] == ("Drop 2", 64)
    assert ("Drop 3", 112) in pads.values()
    assert prop.notes == []


def test_longer_breakdown_wins():
    track = (
        S("intro", 16, bass=0.0, other=0.0),
        S("drop", 32),
        S("breakdown", 8, drums=0.1, bass=0.0),
        S("drop", 32),
        S("breakdown", 16, drums=0.1, bass=0.0),
        S("drop", 32),
        S("outro", 16, bass=0.0, other=0.0),
    )
    assert _bars(propose_cues_stems(*_track(*track)))["D"] == 88


def test_breakdown_found_a_bar_early_moves_to_the_phrase_start():
    """Tracks cut the beat a bar or two before a section ends."""
    track = (
        S("intro", 16, bass=0.0, other=0.0),
        S("drop", 47),
        S("breakdown", 17, drums=0.1, bass=0.0),
        S("drop", 32),
        S("outro", 16, bass=0.0, other=0.0),
    )
    pads = _pads(propose_cues_stems(*_track(*track)))
    assert pads["D"][1] == 64
    assert pads["C"][1] == 56


def test_vocals_cue_starts_on_the_phrase_even_if_the_beat_rings_on():
    track = (
        S("intro", 16, bass=0.0, other=0.0),
        S("drop", 32),
        S("breakdown", 1, bass=0.0, vocals=0.8),
        S("", 15, drums=0.1, bass=0.0, vocals=0.8),
        S("drop", 32),
        S("outro", 16, bass=0.0, other=0.0),
    )
    assert _pads(propose_cues_stems(*_track(*track)))["F"] == ("Vocals", 48)


def test_bridge_at_the_very_end_stays_inside_the_track():
    track = (
        S("intro", 16, bass=0.0, other=0.0),
        S("drop", 38),
        S("", 10, bass=0.0),
    )
    prop = propose_cues_stems(*_track(*track))
    assert all(bar < 64 for bar in _bars(prop).values())


def test_groove_coming_in_eight_bars_before_the_end_of_a_short_track():
    track = (S("drop", 16, bass=0.0, other=0.0), S("", 8))
    assert _bars(propose_cues_stems(*_track(*track)))["B"] == 16


def test_a_three_bar_gap_is_not_a_breakdown():
    track = (
        S("intro", 16, bass=0.0, other=0.0),
        S("drop", 32),
        S("", 3, drums=0.0, bass=0.0),
        S("", 29),
        S("outro", 16, bass=0.0, other=0.0),
    )
    prop = propose_cues_stems(*_track(*track))
    assert _bars(prop)["D"] == 80
    assert "D: no breakdown without vocals; using the outro" in prop.notes


def test_at_most_two_vocal_cues():
    """One drop, no loop and no bridge leave three pads for three vocal passages."""
    vocal_break = S("breakdown", 16, drums=0.1, bass=0.0, vocals=0.8)
    track = (
        S("intro", 16, bass=0.0, other=0.0),
        S("drop", 8), vocal_break,
        S("", 8), vocal_break,
        S("", 8), vocal_break,
        S("", 8),
        S("outro", 16, bass=0.0, other=0.0),
    )
    labels = [label for label, _ in _pads(propose_cues_stems(*_track(*track))).values()]
    assert labels.count("Vocals") == 2


def test_early_break_before_the_vocals_is_not_the_way_out():
    track = (
        S("intro", 16, bass=0.0, other=0.0),
        S("drop", 16),
        S("breakdown", 8, drums=0.1, bass=0.0),
        S("drop", 32, vocals=0.3),
        S("breakdown", 16, drums=0.1, bass=0.0, vocals=0.8),
        S("drop", 32, vocals=0.3),
        S("outro", 16, bass=0.0, other=0.0),
    )
    prop = propose_cues_stems(*_track(*track))
    assert _bars(prop)["D"] == 120
    assert "D: no breakdown without vocals; using the outro" in prop.notes


def test_cue_a_bar_after_the_phrase_start_moves_back_to_it():
    track = (
        S("intro", 16, bass=0.0, other=0.0),
        S("drop", 33),
        S("breakdown", 15, drums=0.1, bass=0.0),
        S("drop", 32),
        S("outro", 16, bass=0.0, other=0.0),
    )
    assert _bars(propose_cues_stems(*_track(*track)))["D"] == 48


def test_vocals_in_the_first_bars_stay_at_the_start_of_the_track():
    """Phrases are counted from the drop at bar 6, which puts bar 0 two bars into one."""
    track = (
        S("intro", 6, drums=0.1, bass=0.0, vocals=0.8),
        S("drop", 42),
        S("breakdown", 16, drums=0.1, bass=0.0),
        S("outro", 8, drums=0.1, bass=0.0),
    )
    assert _pads(propose_cues_stems(*_track(*track)))["F"] == ("Vocals", 0)


def test_fill_before_a_drop_is_not_a_breakdown():
    """A two-bar section labeled breakdown, right before the second drop."""
    track = (
        S("intro", 16, bass=0.0, other=0.0),
        S("drop", 30),
        S("breakdown", 2, drums=0.0, bass=0.0),
        S("drop", 32),
        S("outro", 16, bass=0.0, other=0.0),
    )
    prop = propose_cues_stems(*_track(*track))
    assert _bars(prop)["D"] == 80
    assert _pads(prop)["H"] == ("Drop 2", 48)


INTRO = S("intro", 8, bass=0.0, other=0.0)
OUTRO = S("outro", 8, bass=0.0, other=0.0)


def test_silent_opening_section_does_not_borrow_the_groove_of_the_next_one():
    track = (
        S("drop", 8, drums=0.0, bass=0.0, other=0.0),
        S("breakdown", 2, drums=0.0, bass=0.0, other=0.0),
        S("drop", 32),
        OUTRO,
    )
    pads = _pads(propose_cues_stems(*_track(*track)))
    assert pads["B"][1] == 10
    assert pads["A"][1] == 2


def test_bassline_leaving_a_bar_late_still_makes_a_breakdown():
    track = (INTRO, S("drop", 17), S("", 8, bass=0.0), S("", 8))
    pads = _pads(propose_cues_stems(*_track(*track)))
    assert pads["D"][1] == 24
    assert pads["C"][1] == 16


def test_breakdown_where_only_the_melody_leaves():
    track = (INTRO, S("drop", 16, other=1.0), S("", 8, other=0.0), S("", 8, other=1.0))
    pads = _pads(propose_cues_stems(*_track(*track)))
    assert pads["D"][1] == 24
    assert pads["C"][1] == 16


def test_fill_followed_by_a_full_buildup_is_not_a_breakdown():
    track = (
        INTRO,
        S("drop", 16),
        S("breakdown", 3, drums=0.0, bass=0.0),
        S("buildup", 5),
        S("drop", 16),
        OUTRO,
    )
    prop = propose_cues_stems(*_track(*track))
    assert _bars(prop)["D"] == 48
    assert "D: no breakdown without vocals; using the outro" in prop.notes


def test_vocals_cue_never_moves_past_a_short_passage():
    track = (
        INTRO,
        S("drop", 22),
        S("", 2, drums=0.1, bass=0.0, vocals=0.8),
        S("", 24),
        OUTRO,
    )
    assert _pads(propose_cues_stems(*_track(*track)))["F"] == ("Vocals", 30)


def test_bridge_longer_than_the_rest_of_the_drop():
    track = (
        INTRO,
        S("drop", 16),
        S("", 48, bass=0.0, vocals=0.5),
        S("", 16),
        S("breakdown", 16, drums=0.1, bass=0.0),
        S("drop", 8),
        OUTRO,
    )
    assert _pads(propose_cues_stems(*_track(*track)))["G"] == ("Bridge", 24)


def test_breakdown_two_bars_early_moves_to_the_phrase_start():
    track = (INTRO, S("drop", 30), S("breakdown", 10, drums=0.1, bass=0.0), S("drop", 16), OUTRO)
    pads = _pads(propose_cues_stems(*_track(*track)))
    assert pads["D"][1] == 40
    assert pads["C"][1] == 32


def test_breakdown_two_bars_late_moves_back_to_the_phrase_start():
    track = (INTRO, S("drop", 34), S("breakdown", 14, drums=0.1, bass=0.0), S("drop", 16), OUTRO)
    assert _pads(propose_cues_stems(*_track(*track)))["D"][1] == 40


def test_labeled_breakdown_whose_beat_rings_on_for_a_bar():
    track = (
        INTRO,
        S("drop", 32),
        S("breakdown", 1),
        S("", 15, drums=0.1, bass=0.0),
        S("drop", 16),
        OUTRO,
    )
    prop = propose_cues_stems(*_track(*track))
    assert _bars(prop)["D"] == 40
    assert prop.notes == []


def test_outro_when_every_breakdown_has_vocals():
    track = (
        S("intro", 16, bass=0.0, other=0.0),
        S("drop", 32),
        S("breakdown", 16, drums=0.1, bass=0.0, vocals=0.8),
        S("drop", 32),
        S("outro", 16, bass=0.0, other=0.0),
    )
    prop = propose_cues_stems(*_track(*track))
    assert _bars(prop)["D"] == 96
    assert "D: no breakdown without vocals; using the outro" in prop.notes


def test_vocals_all_the_way_still_gets_a_breakdown():
    track = (
        S("intro", 16, bass=0.0, other=0.0, vocals=0.5),
        S("drop", 32, vocals=0.5),
        S("breakdown", 16, drums=0.1, bass=0.0, vocals=0.5),
        S("drop", 32, vocals=0.5),
        S("outro", 16, bass=0.0, other=0.0, vocals=0.5),
    )
    prop = propose_cues_stems(*_track(*track))
    assert _bars(prop)["D"] == 48
    assert "D: vocals go on through every breakdown" in prop.notes


def test_bassline_dropping_out_counts_as_a_breakdown():
    track = (
        S("intro", 16, bass=0.0, other=0.0),
        S("drop", 48),
        S("", 16, bass=0.0),
        S("", 32),
        S("outro", 4, bass=0.0, other=0.0),
    )
    pads = _pads(propose_cues_stems(*_track(*track)))
    assert pads["D"][1] == 64
    assert pads["C"][1] == 56


def test_bridge_is_the_beat_without_its_bassline():
    track = (
        S("intro", 16, bass=0.0, other=0.0),
        S("drop", 32),
        S("", 16, bass=0.0, vocals=0.5),
        S("", 32),
        S("breakdown", 16, drums=0.1, bass=0.0),
        S("drop", 32),
        S("outro", 16, bass=0.0, other=0.0),
    )
    pads = _pads(propose_cues_stems(*_track(*track)))
    assert pads["G"] == ("Bridge", 48)
    assert pads["D"][1] == 96


def test_loop_pads_end_8_bars_after_they_start():
    prop = propose_cues_stems(*_track(*CLASSIC))
    loops = [pad for pad in prop.positions if prop.labels[pad] == "Loop"]
    assert loops
    assert prop.loop_ends == {pad: prop.positions[pad] + 8 * BAR_S for pad in loops}


def test_short_track_only_gets_a_and_b():
    prop = propose_cues_stems(*_track(S("intro", 8, bass=0.0), S("drop", 12)))
    assert _pads(prop) == {"A": ("8 Bars to Drop", 0), "B": ("Drop", 8)}
    assert prop.notes == ["track too short (20 bars) for the other cues"]


def test_short_loop_gets_b():
    prop = propose_cues_stems(*_track(S("drop", 16)))
    assert _pads(prop) == {"B": ("Peak", 0)}


def test_8_bar_loop_gets_b():
    assert _pads(propose_cues_stems(*_track(S("drop", 8)))) == {"B": ("Peak", 0)}


def test_track_under_8_bars_gets_nothing():
    prop = propose_cues_stems(*_track(S("drop", 7)))
    assert prop.positions == {}
    assert prop.notes == ["track too short (7 bars); no cues"]


def test_section_runs_snap_merge_and_drop_silence():
    starts = np.arange(40) * BAR_S
    segments = [
        Segment(0.0, 0.3, "start"),
        Segment(0.3, 16.2, "intro"),
        Segment(16.2, 31.9, "drop"),
        Segment(31.9, 48.0, "drop"),
        Segment(48.0, 48.4, "breakdown"),
        Segment(48.4, 70.1, "outro"),
        Segment(70.1, 80.0, "end"),
    ]
    assert section_runs(segments, starts) == [("intro", 0, 8), ("drop", 8, 24), ("outro", 24, 35)]


def test_section_ending_on_the_last_bar_line_leaves_that_bar_to_the_next():
    starts = np.arange(24) * BAR_S
    segments = [Segment(0.0, 46.0, "drop"), Segment(46.0, 48.0, "outro")]
    assert section_runs(segments, starts) == [("drop", 0, 23), ("outro", 23, 24)]
