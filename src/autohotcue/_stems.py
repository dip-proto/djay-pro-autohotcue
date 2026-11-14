"""Per-bar stem levels and Raveform sections, for the ml-stems engine.

demucs splits the track into drums, bass, other and vocals.
The Raveform model reads the same stems, so both come out of a single separation.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from autohotcue import _allin1, _raveform
from autohotcue.backends import BeatAnalysis, StructureAnalysis
from autohotcue.gridlock import START_SLACK_S, GridFit, _refine_bpm
from autohotcue.stempolicy import BarFeatures

STEMS = ("drums", "bass", "other", "vocals")
FRAME = 1024
MEL_BANDS = 48
BAR_SLOTS = 16
DOWNBEAT_SLACK_S = 0.05
STEADY_TOLERANCE = 0.12
STEADY_MIN_BEATS = 0.85
DOUBLED_BEAT = 0.6
MISSING_BEAT = 1.5
LOCAL_GAPS = 8
USUAL_GAP = 0.1
SAME_TEMPO = 0.05


def _grid_edges(anchor_s: float, bar_s: float, duration_s: float) -> np.ndarray:
    # Many files open right on a downbeat, which the grid then puts a few milliseconds
    # before the audio. That bar counts, from the start of the file.
    first = anchor_s - bar_s if anchor_s > bar_s - START_SLACK_S else anchor_s
    count = int((duration_s - first) / bar_s + 1e-6)
    edges = first + bar_s * np.arange(count + 1)
    edges[0] = max(edges[0], 0.0)
    return edges


def _off_lattice(times: np.ndarray, phase_s: float, period_s: float) -> np.ndarray:
    """Distance of each time to the nearest lattice beat, in beats."""
    position = (np.asarray(times, dtype=np.float64) - phase_s) / period_s
    return np.abs(position - np.round(position))


def _even_beats(beats: np.ndarray) -> np.ndarray:
    """Tracked beats without the doubled ones, and with the missing ones filled in.

    A gap is one beat when it is about as long as the usual one.
    A much shorter one ends on a doubled beat, a longer one has missing beats in it.
    """
    period = float(np.median(np.diff(beats)))
    out = [float(beats[0])]
    one_beat_gaps = [period]
    for beat in beats[1:]:
        last = out[-1]
        gap = float(beat) - last
        if gap < DOUBLED_BEAT * period:
            continue
        if gap < MISSING_BEAT * period:
            one_beat_gaps.append(gap)
            out.append(float(beat))
            continue
        # The tempo may drift, so missing beats are counted at the tempo of the last two bars,
        # as told by the single beats the tracker reported there.
        steps = round(gap / float(np.median(one_beat_gaps[-LOCAL_GAPS:])))
        out.extend(last + gap * k / steps for k in range(1, steps + 1))
    return np.array(out)


def _tempo_guesses(beats: np.ndarray) -> list[float]:
    """Beat periods to search the tempo around: the mean gap, and the mean of the usual gaps.

    A stretch of stray beats throws the first one off.
    The second one needs most gaps to be alike.
    The median gap alone is too coarse, since the tracker reports beats in 20 ms steps.
    """
    gaps = np.diff(beats)
    usual = gaps[np.abs(gaps / np.median(gaps) - 1.0) < USUAL_GAP]
    return [float(some.mean()) for some in (gaps, usual) if len(some)]


def _beats_per_bar(tempo: float, djay_bpm: float | None) -> int:
    """How many tracked beats make a bar of four djay beats.

    djay sometimes counts half as many beats as the tracker, or twice as many.
    Bars follow djay, since its beats are the ones shown on screen.
    """
    if djay_bpm:
        for per_bar in (2, 8):
            if abs(4.0 * tempo / (per_bar * djay_bpm) - 1.0) < SAME_TEMPO:
                return per_bar
    return 4


def _bar_phase(downbeats: np.ndarray, sections: np.ndarray, per_bar: int) -> int:
    """Which beat starts a bar, given the beat numbers of the downbeats and section starts.

    The tracker calls a downbeat every four of its beats.
    In a bar of eight, that leaves two beats to choose from, and the one where
    more sections start is taken: a drop starts on a bar.
    """
    step = min(per_bar, 4)
    first = int(np.bincount(downbeats % step, minlength=step).argmax())
    candidates = range(first, per_bar, step)
    section_votes = np.bincount(sections % per_bar, minlength=per_bar)
    downbeat_votes = np.bincount(downbeats % per_bar, minlength=per_bar)
    return max(candidates, key=lambda phase: (section_votes[phase], downbeat_votes[phase], -phase))


def _steady_edges(
    beats: np.ndarray,
    downbeats: np.ndarray,
    duration_s: float,
    djay_bpm: float | None,
    section_starts: np.ndarray,
) -> np.ndarray | None:
    """Bars on a straight lattice, when the tracked beats keep one tempo all along.

    The grid fit gives up on a track as soon as a few beats are off, because it would
    overwrite djay's grid.
    Counting bars needs less: most beats on one lattice is enough,
    and a lattice has no extra or missing beat to make a bar longer than the next.
    """
    weights = np.ones(len(beats))
    bpm, _ = min(
        (_refine_bpm(60.0 / guess, beats, weights) for guess in _tempo_guesses(beats)),
        key=lambda found: found[1],
    )
    period = 60.0 / bpm
    phase = float(np.angle(np.mean(np.exp(2j * np.pi * beats / period))) / (2 * np.pi) * period)
    if np.mean(_off_lattice(beats, phase, period) < STEADY_TOLERANCE) < STEADY_MIN_BEATS:
        return None
    per_bar = _beats_per_bar(bpm, djay_bpm)
    on_beat = downbeats[_off_lattice(downbeats, phase, period) < STEADY_TOLERANCE]
    index = np.round((on_beat - phase) / period).astype(int)
    sections = np.round((section_starts - phase) / period).astype(int)
    first = _bar_phase(index, sections, per_bar)
    bar_s = per_bar * period
    return _grid_edges((phase + first * period) % bar_s, bar_s, duration_s)


def bar_edges(
    beats: np.ndarray,
    downbeats: np.ndarray,
    fit: GridFit | None,
    duration_s: float,
    djay_bpm: float | None = None,
    section_starts: Sequence[float] = (),
) -> np.ndarray:
    """Start of every bar, plus the end of the last one."""
    if fit is not None and fit.ok:
        # A bar is four of djay's beats, counted on the fitted grid: cues are snapped to
        # that grid when written, and must stay a whole number of beats apart.
        # The fit may have locked onto double or half of djay's tempo.
        bar_s = max(1, round(4.0 * fit.bpm / fit.render_bpm)) * 60.0 / fit.bpm
        return _grid_edges(fit.anchor_s, bar_s, duration_s)
    downbeats = np.asarray(downbeats, dtype=np.float64)
    if len(beats) < 8 or len(downbeats) == 0:
        return np.zeros(0)
    beats = _even_beats(np.asarray(beats, dtype=np.float64))
    section_starts = np.asarray(section_starts, dtype=np.float64)
    edges = _steady_edges(beats, downbeats, duration_s, djay_bpm, section_starts)
    if edges is not None:
        return edges
    # No steady tempo: bars of tracked beats, starting where most downbeats fall.
    # The tracker sometimes calls a downbeat every two beats, so its downbeats alone won't do.
    per_bar = _beats_per_bar(60.0 / float(np.median(np.diff(beats))), djay_bpm)
    index = np.searchsorted(beats, downbeats - DOWNBEAT_SLACK_S)
    sections = np.searchsorted(beats, section_starts - DOWNBEAT_SLACK_S)
    phase = _bar_phase(index[index < len(beats)], sections[sections < len(beats)], per_bar)
    starts = beats[phase::per_bar]
    # The last bar has no beat after it to close it: close it if the audio goes on that long.
    end = starts[-1] + float(np.median(np.diff(starts)))
    return np.append(starts, end) if end <= duration_s + DOWNBEAT_SLACK_S else starts


def frame_power(x: np.ndarray) -> np.ndarray:
    """Mean square of a signal over frames of FRAME samples."""
    count = len(x) // FRAME
    frames = x[: count * FRAME].reshape(count, FRAME).astype(np.float64)
    return (frames * frames).mean(axis=1)


def bar_levels(power: np.ndarray, edges: np.ndarray, sr: int) -> np.ndarray:
    """RMS of each bar, from frame powers."""
    bounds = (edges * sr / FRAME).astype(int)
    out = np.zeros(len(edges) - 1)
    for i, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        part = power[max(a, 0):b]
        out[i] = np.sqrt(part.mean()) if len(part) else 0.0
    return out


def bar_timbre(mel_db: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """One spectral snapshot per bar: its mel spectrum in BAR_SLOTS steps (16th notes).

    ``mel_db`` has one column per FRAME samples at the rate ``edges`` was scaled to.
    """
    bands = mel_db.shape[0]
    out = np.zeros((len(edges) - 1, BAR_SLOTS * bands), dtype=np.float32)
    for i, (start, end) in enumerate(zip(edges[:-1], edges[1:])):
        steps = (start + (end - start) * np.arange(BAR_SLOTS + 1) / BAR_SLOTS).astype(int)
        for k in range(BAR_SLOTS):
            a, b = steps[k], max(steps[k + 1], steps[k] + 1)
            if 0 <= a < mel_db.shape[1]:
                out[i, k * bands:(k + 1) * bands] = mel_db[:, a:b].mean(axis=1)
    return out


def mel_db(y: np.ndarray, sr: int) -> np.ndarray:
    import librosa

    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=2 * FRAME, hop_length=FRAME, n_mels=MEL_BANDS
    )
    return librosa.power_to_db(mel, ref=np.max)


def bar_features(
    powers: dict[str, np.ndarray], mel: np.ndarray, edges: np.ndarray, sr: int
) -> BarFeatures:
    """``powers`` holds the frame powers of the four stems and of the mix."""
    levels = {name: bar_levels(power, edges, sr) for name, power in powers.items()}
    return BarFeatures(
        starts=edges[:-1],
        timbre=bar_timbre(mel, edges * sr / FRAME),
        **levels,
    )


def _mono(stem) -> np.ndarray:
    x = np.array(stem, copy=False)
    return x.mean(axis=int(np.argmin(x.shape))) if x.ndim == 2 else x


def analyze_stems(
    path: str,
    y: np.ndarray,
    sr: int,
    beat: BeatAnalysis,
    fit: GridFit | None,
    djay_bpm: float | None = None,
) -> tuple[BarFeatures, StructureAnalysis]:
    """Separate the track once; return its bars and its Raveform sections."""
    separator, model = _allin1._prepare(_raveform.resolve_raveform_weights(), _raveform.MODEL_NAME)
    if separator.samplerate != sr:
        raise RuntimeError(f"demucs runs at {separator.samplerate} Hz but the track was decoded at {sr}")
    try:
        stems = _allin1._separate(path, separator)
        powers = {name: frame_power(_mono(stems[name])) for name in STEMS}
        _, sections, functions = _allin1._probabilities(_allin1._spectrogram(stems, separator), model)
        del stems
        segments = _raveform.raveform_segments(sections, functions, model.cfg, beat.duration_s)
    finally:
        _allin1._clear_mlx_cache()
    powers["mix"] = frame_power(y)
    edges = bar_edges(
        beat.beats, beat.downbeats, fit, beat.duration_s, djay_bpm, [s.start for s in segments]
    )
    if len(edges) < 2:
        edges = np.array([0.0, beat.duration_s])
    feats = bar_features(powers, mel_db(y, sr), edges, sr)
    return feats, StructureAnalysis(segments=segments, source="raveform")
