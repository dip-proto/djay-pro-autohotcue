"""Pad C placed N bars before the drop."""
from __future__ import annotations

import concurrent.futures
import inspect

import numpy as np
import pytest

from autohotcue import analysis, cli
from autohotcue.analysis import CueProposal, TrackAnalysis, place_pre_drop_cue
from autohotcue.gridlock import GridFit, snap_cues

BAR = 2.0  # at 120 BPM
TOO_CLOSE = "C: no cue 8 bars before the drop (drop too close to the start)"


def _fit(ok: bool = True, bpm: float = 120.0, render_bpm: float = 120.0) -> GridFit:
    return GridFit(
        bpm=bpm, render_bpm=render_bpm, anchor_s=0.5, beat_fit=0.0, bar_resid_std=0.0,
        splice_jump=0.0, ok=ok, reason="" if ok else "spliced edit",
    )


def _track(fit=None, downbeats=None, engine="ml-bass", bpm=120.0) -> TrackAnalysis:
    return TrackAnalysis(
        bpm=bpm, first_beat_s=0.5, duration_s=300.0, engine=engine,
        downbeats=downbeats, grid_fit=fit,
    )


@pytest.mark.parametrize("cues, bars, notes, want_c, want_notes", [
    pytest.param({"C": 32.5, "D": 64.5}, 8, [], 48.5,
                 ["C: 8 bars before the drop (replaces buildup at 32.50s)"], id="replaces"),
    pytest.param({"D": 64.5}, 8, ["C (Buildup): omitted (nothing between B and D)"], 48.5,
                 ["C: 8 bars before the drop"], id="fills-empty-pad"),
    pytest.param({"C": 48.6, "D": 64.5}, 8, [], 48.5, ["C: 8 bars before the drop"], id="nudges-nearby-c"),
    pytest.param({"C": 48.75, "D": 65.0}, 8, [], 49.0, ["C: 8 bars before the drop"], id="half-beat-off"),
    pytest.param({"D": 64.62}, 8, [], 48.62, ["C: 8 bars before the drop"], id="unsnapped-drop"),
    pytest.param({"A": 48.5, "B": 48.5, "D": 64.5}, 8, [], None, [], id="first-beat-already-there"),
    pytest.param({"D": 64.5}, 16, [], 32.5, ["C: 16 bars before the drop"], id="16-bars"),
    pytest.param({"C": 32.5, "D": 64.5}, 0, [], 32.5, [], id="turned-off"),
    pytest.param({"G": 200.5}, 8, [], None, [], id="no-drop"),
    pytest.param({"C": 4.5, "D": 10.5}, 8, [], 4.5, [TOO_CLOSE], id="drop-too-early"),
    pytest.param({"B": 40.5, "D": 52.5}, 8, [], None, [TOO_CLOSE], id="before-loop-in"),
])
def test_pre_drop_cue(cues, bars, notes, want_c, want_notes):
    p = CueProposal(positions={"A": 0.5, "B": 0.5, **cues}, notes=list(notes))
    place_pre_drop_cue(p, _track(_fit()), bars=bars)
    assert p.positions.get("C") == (None if want_c is None else pytest.approx(want_c))
    assert p.notes == want_notes


DRIFTING = 0.5 + np.cumsum(np.linspace(2.0, 2.05, 40))  # about 120 down to 117 BPM
STEADY = 0.5 + 2.0 * np.arange(40)


@pytest.mark.parametrize("track, drop, want_c", [
    pytest.param(_track(_fit(ok=False), DRIFTING), DRIFTING[20], DRIFTING[12], id="tempo-drift"),
    pytest.param(_track(_fit(ok=False), STEADY), STEADY[20] + 0.5, STEADY[12] + 0.5, id="off-downbeat"),
    pytest.param(_track(_fit(ok=False), STEADY), STEADY[5], None, id="too-few-downbeats"),
    pytest.param(_track(engine="legacy", bpm=125.0), 100.0, 100.0 - 8 * 240.0 / 125.0, id="legacy"),
    pytest.param(_track(_fit(bpm=160.0, render_bpm=80.0)), 48.5, 24.5, id="double-time-fit"),
])
def test_bar_length_follows_the_track(track, drop, want_c):
    p = CueProposal(positions={"A": 0.5, "B": 0.5, "D": float(drop)})
    place_pre_drop_cue(p, track)
    assert p.positions.get("C") == (None if want_c is None else pytest.approx(float(want_c)))


# 64.75 and 6.25 sit exactly between two grid beats.
@pytest.mark.parametrize("drop, bars", [(64.62, 8), (64.75, 8), (64.75, 3), (6.25, 1)])
def test_gap_survives_snapping(drop, bars):
    fit = _fit()
    p = CueProposal(positions={"A": 0.5, "B": 0.5, "D": drop})
    place_pre_drop_cue(p, _track(fit), bars=bars)
    snapped = snap_cues(p.positions, fit)
    assert snapped["D"] - snapped["C"] == pytest.approx(bars * BAR)


@pytest.mark.parametrize("cmd", ["propose", "viz", "apply"])
def test_cli_flag(monkeypatch, cmd):
    seen = []
    monkeypatch.setattr(cli, f"cmd_{cmd}", seen.append)
    extra = ["out.png"] if cmd == "viz" else []
    cli.main([cmd, "track.opus", *extra])
    assert seen[0].pre_drop_bars == 8
    with pytest.raises(SystemExit):
        cli.main([cmd, "track.opus", *extra, "--pre-drop-bars", "-1"])


class _InlinePool:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def submit(self, fn, *args, **kwargs):
        fut = concurrent.futures.Future()
        fut.set_result(fn(*args, **kwargs))
        return fut


@pytest.mark.parametrize("folder, bars", [(False, 0), (True, 16)])
def test_propose_passes_the_bar_count_on(monkeypatch, tmp_path, folder, bars):
    seen = []
    signature = inspect.signature(analysis.analyze)

    def fake_analyze(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        seen.append(bound.arguments["pre_drop_bars"])
        return _track(_fit()), CueProposal(positions={"A": 0.5})

    monkeypatch.setattr(analysis, "analyze", fake_analyze)
    monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", _InlinePool)
    for name in ("a.opus", "b.opus"):
        (tmp_path / name).write_bytes(b"")
    target = tmp_path if folder else tmp_path / "a.opus"
    cli.main(["propose", str(target), "-j", "2", "--pre-drop-bars", str(bars)])
    assert seen == [bars] * (2 if folder else 1)
