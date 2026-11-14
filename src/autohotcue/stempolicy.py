"""Cue placement for the ml-stems engine.

Pads A to D mark the way in and the way out of a track:
A is 8 bars before the drop, B the drop, C 8 bars before the breakdown, D the breakdown.
B is the one every track gets: without a drop, it marks where the fullest passage starts.
Pads E to H mark other moments worth jumping to:
a loop, a vocals-only passage, a bridge and the second drop.

The Raveform labels say where sections start.
The stems say how much energy each section has and whether someone is singing.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import zip_longest

import numpy as np

from autohotcue.analysis import CueProposal
from autohotcue.backends import Segment

LEAD_BARS = 8
LOOP_BARS = 8
MIN_BARS = 24

# Levels are fractions of the loud level (the 90th percentile bar) of a stem or of the mix.
# Energy is the mean of the drums and bass levels, so a bassline coming in counts
# as much as the drums coming in.
WEAK_DROP_GAIN = 0.1
PEAK_KEEP = 0.95
ARRIVED = 0.85
BREAK_MIN_LOSS = 0.15
THIN_MIN_LOSS = 0.25
STEM_DROP_MIN_GAIN = 0.3
QUIET_STEM = 0.1
BREAK_MIN_BARS = 4
BREAK_FULL_BARS = 16
VOCAL_FREE_MAX = 0.08
VOCALS_LEFT_MAX = 0.5
VOCAL_ON = 0.15
VOCALS_ONLY_MAX_BEAT = 0.45
VOCALS_ONLY_MIN_BARS = 2
BRIDGE_MIN_BARS = 8
BRIDGE_DRUMS_ON = 0.6
BRIDGE_BASS_OFF = 0.3
LOOP_MIN_SIMILARITY = 0.7
LOOP_MAX_CHANGE = 0.2
LOOP_DISTINCT_MAX = 0.7
MAX_PER_KIND = 2
LOOP_MIN_ENERGY = 0.7
CUE_GAP_BARS = 4
PHRASE_SNAP_BARS = 2

_DROP = "drop"
_LOW = frozenset({"breakdown", "cooldown", "intro", "outro", "altintro", "altoutro"})
_SILENCE = frozenset({"start", "end"})

LABELS = {"A": "8 Bars to Drop", "B": "Drop", "C": "8 Bars to Break", "D": "Breakdown"}
PEAK_LABELS = {"A": "8 Bars to Peak", "B": "Peak"}


@dataclass(frozen=True)
class BarFeatures:
    """What the policy knows about each bar of a track."""

    starts: np.ndarray  # bar start times, in seconds
    drums: np.ndarray  # RMS of each stem, and of the whole mix, per bar
    bass: np.ndarray
    other: np.ndarray
    vocals: np.ndarray
    mix: np.ndarray
    timbre: np.ndarray  # (bars, k): one spectral snapshot per bar, to compare bars


@dataclass(frozen=True)
class Moment:
    kind: str
    bar: int


def section_runs(segments: list[Segment], starts: np.ndarray) -> list[tuple[str, int, int]]:
    """Sections as (label, first bar, end bar), with silence dropped and repeats merged."""
    bar_s = float(np.median(np.diff(starts))) if len(starts) > 1 else 0.0
    edges = np.append(starts, starts[-1] + bar_s)
    runs: list[tuple[str, int, int]] = []
    for seg in segments:
        if seg.label in _SILENCE:
            continue
        first = int(np.argmin(np.abs(edges - seg.start)))
        end = int(np.argmin(np.abs(edges - seg.end)))
        if end <= first:
            continue
        if runs and runs[-1][0] == seg.label:
            runs[-1] = (seg.label, runs[-1][1], end)
        else:
            runs.append((seg.label, first, end))
    return runs


def _loud(x: np.ndarray) -> float:
    return max(float(np.percentile(x, 90)), 1e-9)


def _mean(x: np.ndarray, a: int, b: int) -> float:
    part = x[max(a, 0):max(b, 0)]
    return float(part.mean()) if len(part) else 0.0


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2])]


def _run_length(run: tuple[int, int]) -> int:
    return run[1] - run[0]


def unit_bars(timbre: np.ndarray) -> np.ndarray:
    """Bars as unit vectors, so that a dot product is their cosine similarity."""
    x = timbre - timbre.mean(axis=0, keepdims=True)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)


class _Track:
    def __init__(self, feats: BarFeatures, segments: list[Segment]):
        self.n = len(feats.starts)
        mix = _loud(feats.mix)
        # A stem that never gets loud is leakage from the others, not an instrument.
        self.drums = feats.drums / _loud(feats.drums) if _loud(feats.drums) >= QUIET_STEM * mix else None
        self.bass = feats.bass / _loud(feats.bass) if _loud(feats.bass) >= QUIET_STEM * mix else None
        stems = [x for x in (self.drums, self.bass) if x is not None]
        self.energy = np.mean(stems, axis=0) if stems else feats.mix / mix
        self.beat = np.max(stems, axis=0) if stems else feats.mix / mix
        self.vocals = feats.vocals / mix
        every = (feats.drums, feats.bass, feats.other, feats.vocals)
        active = [np.minimum(x / _loud(x), 1.5) for x in every if _loud(x) >= QUIET_STEM * mix]
        self.levels = np.array(active) if active else (feats.mix / mix)[None, :]
        self.fullness = self.levels.mean(axis=0)
        self.runs = section_runs(segments, feats.starts)
        self.phrase_starts = sorted(s for label, s, _ in self.runs if label == _DROP)
        self.unit = unit_bars(feats.timbre)
        # How much each bar sounds like the one 8 bars later.
        self.repeat = np.sum(self.unit[:-LOOP_BARS] * self.unit[LOOP_BARS:], axis=1)

    def loss(self, bar: int, bars: int = LEAD_BARS) -> float:
        """How much the track thins out at ``bar``, against the 8 bars before.

        It is the larger change of the two: beat and bassline, or all the stems.
        A negative loss is a gain.
        """
        changes = [
            _mean(x, bar - LEAD_BARS, bar) - _mean(x, bar, bar + bars)
            for x in (self.energy, self.fullness)
        ]
        return max(changes, key=abs)

    def recovery(self, bar: int) -> int | None:
        """The first bar where what the track lost at ``bar`` is halfway back, if it ever is."""
        lost = [
            (_mean(x, bar - LEAD_BARS, bar), _mean(x, bar, bar + LEAD_BARS), x)
            for x in (self.energy, self.fullness)
        ]
        before, after, x = max(lost, key=lambda entry: entry[0] - entry[1])
        # The first bars may not have thinned out yet: cues are found up to two bars early.
        start = bar + PHRASE_SNAP_BARS
        back = np.flatnonzero(x[start:] >= (before + after) / 2)
        return start + int(back[0]) if len(back) else None

    def drops(self) -> list[int]:
        """Bars where a drop section starts after something else."""
        return [s for i, (label, s, _) in enumerate(self.runs) if label == _DROP and i > 0]

    def last_drop_end(self) -> int:
        return max((e for label, _, e in self.runs if label == _DROP), default=0)

    def phrase_bar(self, bar: int) -> int:
        """How many bars into its 8-bar phrase ``bar`` is.

        Phrases are counted from the drop the bar is in or follows, since drops start
        on the right bar far more reliably than anything else.
        """
        before = [d for d in self.phrase_starts if d <= bar]
        origin = before[-1] if before else min(self.phrase_starts, default=0)
        return (bar - origin) % LOOP_BARS

    def phrase_start(self, bar: int, forward: bool = True) -> int:
        """``bar``, or the phrase start next to it when it is within two bars of one.

        Tracks often cut the beat a bar or two before a section ends,
        and let it ring on for a bar after.
        ``forward=False`` only moves back to the start of the phrase the bar is in.
        """
        into = self.phrase_bar(bar)
        if into <= PHRASE_SNAP_BARS:
            return max(bar - into, 0)
        left = LOOP_BARS - into
        return bar + left if forward and left <= PHRASE_SNAP_BARS and bar + left < self.n else bar

    def vocals_left(self, bar: int) -> float:
        """The share of the track's vocals that comes after ``bar``."""
        total = float(self.vocals.sum())
        if total < VOCAL_FREE_MAX * self.n:
            return 0.0
        return float(self.vocals[bar:].sum()) / total


def _arrival(t: _Track, drop: int) -> int:
    """The bar where the beat and bassline of a labeled drop came in.

    The model often calls the first 8 or 16 bars of full groove a buildup,
    and starts the drop where the hook joins.
    """
    level = _mean(t.energy, drop, drop + LEAD_BARS)
    bar = drop
    while bar >= LEAD_BARS and _mean(t.energy, bar - LEAD_BARS, bar) >= ARRIVED * level:
        bar -= LEAD_BARS
    # A groove that is there from the start never arrives.
    return bar if bar >= LEAD_BARS else drop


def _groove_start(t: _Track, end: int) -> int | None:
    """The first bar where the beat and bassline clearly come in, for 8 bars before ``end``."""
    rise = np.array([
        _mean(t.energy, s, s + LEAD_BARS) - _mean(t.energy, s - CUE_GAP_BARS, s)
        for s in range(1, min(end, t.n) - LEAD_BARS + 1)
    ])
    strong = rise >= STEM_DROP_MIN_GAIN
    if not strong.any():
        return None
    # The rise builds up over the bars before the groove comes in, and peaks on it.
    first, last = _true_runs(strong)[0]
    return 1 + first + int(np.argmax(rise[first:last]))


def _pick_drop(t: _Track) -> tuple[int | None, list[int]]:
    """The drop, and the drops after it."""
    drops = t.drops()
    if t.runs and t.runs[0][0] == _DROP:
        # No intro section: the drop is where the groove comes in, if the stems show it.
        opening = _groove_start(t, t.runs[0][2])
        if opening is not None:
            return opening, drops
    if drops:
        return _arrival(t, drops[0]), drops[1:]
    # No drop section to go by: the stems may still show the groove coming in.
    return _groove_start(t, t.n), []


def _peak(t: _Track) -> int:
    """The first bar of the fullest passage of the track."""
    last = t.n - LEAD_BARS
    level = np.array([_mean(t.fullness, s, s + LEAD_BARS) for s in range(last + 1)])
    # The fullest 8 bars are often in the middle of a longer passage.
    # It starts where the level goes up the most from the 8 bars before.
    rise = np.array([level[s] - level[max(s - LEAD_BARS, 0)] for s in range(last + 1)])
    rise[level < PEAK_KEEP * level.max()] = -np.inf
    return int(rise.argmax())


@dataclass(frozen=True)
class _Exit:
    bar: int
    loss: float  # how much the track thins out there
    bars: int  # how long it stays thin
    final: bool  # nothing drops after it
    vocals: float

    @property
    def score(self) -> float:
        return self.loss * min(self.bars, BREAK_FULL_BARS) / BREAK_FULL_BARS


def _thinning_bars(t: _Track, after: int, known: list[int]) -> list[int]:
    """Bars where the track thins out though no section starts there."""
    loss = np.array([t.loss(s) for s in range(t.n)])
    out = []
    for s in range(max(after, LEAD_BARS), t.n - LEAD_BARS + 1):
        near = loss[s - CUE_GAP_BARS:s + CUE_GAP_BARS + 1]
        if loss[s] >= THIN_MIN_LOSS and loss[s] >= near.max() and _far_from(s, known + out):
            out.append(s)
    return out


def _exits(t: _Track, after: int) -> list[_Exit]:
    """Places from bar ``after`` on where a section ends or the track thins out."""
    last_drop_end = t.last_drop_end()
    drops = [s for label, s, _ in t.runs if label == _DROP]
    sections = [s for label, s, _ in t.runs if label in _LOW and s >= after]
    out = []
    for raw in sections + _thinning_bars(t, after, sections):
        s = t.phrase_start(raw)
        # A gap that ends on a drop is the fill before it.
        if s + LEAD_BARS > t.n or s in drops:
            continue
        labeled = raw in sections
        limit = next((d for d in drops if d > s), t.n) if labeled else t.n
        loss = t.loss(raw, min(limit - raw, LEAD_BARS))
        # A buildup belongs to the breakdown it follows, until the groove is back.
        back = t.recovery(raw) if loss > 0 else None
        end = limit if back is None else min(limit, back)
        if end <= s:
            continue
        final = raw >= last_drop_end if labeled else back is None
        out.append(_Exit(s, loss, end - s, final, _mean(t.vocals, s, s + LEAD_BARS)))
    return sorted(out, key=lambda x: x.bar)


def _pick_breakdown(t: _Track, drop: int | None) -> tuple[int | None, str | None]:
    """The breakdown bar, with a note when it isn't quite one."""
    exits = _exits(t, (drop or 0) + 2 * LEAD_BARS)
    for quiet, note in ((True, None), (False, "D: vocals go on through every breakdown")):
        found = [x for x in exits if x.vocals < VOCAL_FREE_MAX or not quiet]
        inner = [
            x for x in found
            if not x.final and x.loss >= BREAK_MIN_LOSS and x.bars >= BREAK_MIN_BARS
            # A break with most of the singing still to come is not the way out.
            and (t.vocals_left(x.bar) <= VOCALS_LEFT_MAX or not quiet)
        ]
        if inner:
            return max(inner, key=lambda x: (x.score, -x.bar)).bar, note
        ending = [x for x in found if x.final]
        if ending:
            thinning = [x for x in ending if x.loss >= BREAK_MIN_LOSS]
            return (thinning or ending)[0].bar, note or "D: no breakdown without vocals; using the outro"
    return None, None


def _vocal_moments(t: _Track) -> list[Moment]:
    """Passages where someone sings over little or no beat, the most vocal first."""
    solo = (t.vocals >= VOCAL_ON) & (t.beat <= VOCALS_ONLY_MAX_BEAT)
    runs = [(a, b) for a, b in _true_runs(solo) if b - a >= VOCALS_ONLY_MIN_BARS]
    runs.sort(key=lambda r: -float(t.vocals[r[0]:r[1]].sum()))
    # The beat often rings on for a bar or two after the section has started.
    # The cue never moves forward: the passage may be over by the next phrase.
    return [Moment("Vocals", t.phrase_start(a, forward=False)) for a, _ in runs]


def _bridge_moments(t: _Track, drop: int | None) -> list[Moment]:
    """Labeled bridges, then passages where the beat goes on without its bassline."""
    labeled = [(s, e) for label, s, e in t.runs if label == "bridge"]
    stripped = []
    if drop is not None and t.drums is not None and t.bass is not None:
        mask = (t.drums >= BRIDGE_DRUMS_ON) & (t.bass <= BRIDGE_BASS_OFF)
        mask[:drop] = False
        mask[t.last_drop_end():] = False
        stripped = [(a, b) for a, b in _true_runs(mask) if b - a >= BRIDGE_MIN_BARS]
    longest_first = sorted(labeled, key=_run_length, reverse=True)
    longest_first += sorted(stripped, key=_run_length, reverse=True)
    return [Moment("Bridge", t.phrase_start(a)) for a, _ in longest_first]


def _loop_moments(t: _Track) -> list[Moment]:
    """8-bar phrases with the full groove that the track repeats right after, best first.

    Looping such a phrase sounds like the track going on.
    """
    scored = []
    for bar in range(t.n - 2 * LOOP_BARS + 1):
        if t.phrase_bar(bar):
            continue
        nxt = bar + LOOP_BARS
        # The first bar matters most: it is where the loop lands when it wraps around.
        again = min(_mean(t.repeat, bar, nxt), float(t.repeat[bar]))
        # Bars can sound alike while a stem comes or goes: the drums drown it out.
        change = np.abs(t.levels[:, bar:nxt] - t.levels[:, nxt:nxt + LOOP_BARS]).mean(axis=1).max()
        if (
            again >= LOOP_MIN_SIMILARITY
            and change <= LOOP_MAX_CHANGE
            and _mean(t.energy, bar, nxt) >= LOOP_MIN_ENERGY
        ):
            scored.append((again * _mean(t.fullness, bar, nxt), bar))
    return [Moment("Loop", bar) for _, bar in sorted(scored, reverse=True)]


def _same_loop(t: _Track, a: int, b: int) -> bool:
    x, y = t.unit[a:a + LOOP_BARS], t.unit[b:b + LOOP_BARS]
    return float(np.sum(x * y)) / LOOP_BARS >= LOOP_DISTINCT_MAX


def _far_from(bar: int, taken: list[int]) -> bool:
    return all(abs(bar - other) >= CUE_GAP_BARS for other in taken)


def propose_cues_stems(feats: BarFeatures, segments: list[Segment]) -> CueProposal:
    p = CueProposal(labels=dict(LABELS))
    n = len(feats.starts)
    if n < LEAD_BARS:
        p.notes.append(f"track too short ({n} bars); no cues")
        return p
    t = _Track(feats, segments)
    taken: list[int] = []
    placed: dict[str, list[int]] = {}

    def place(pad: str, bar: int, label: str | None = None) -> None:
        p.positions[pad] = float(feats.starts[bar])
        if label is not None:
            p.labels[pad] = label
        taken.append(bar)

    def usable(m: Moment) -> bool:
        if not _far_from(m.bar, taken):
            return False
        if m.kind.startswith("Drop"):
            return True
        same = placed.get(m.kind, [])
        if len(same) >= MAX_PER_KIND:
            return False
        return m.kind != "Loop" or not any(_same_loop(t, m.bar, bar) for bar in same)

    def place_best(pad: str, moments: list[Moment]) -> bool:
        best = next((m for m in moments if usable(m)), None)
        if best is None:
            return False
        moments.remove(best)
        place(pad, best.bar, best.kind)
        if best.kind == "Loop":
            p.loop_ends[pad] = float(feats.starts[best.bar + LOOP_BARS])
        placed.setdefault(best.kind, []).append(best.bar)
        return True

    drop, later = _pick_drop(t)
    if drop is None:
        # B is the cue that matters most, so a track with no drop still gets one.
        drop = _peak(t)
        p.labels.update(PEAK_LABELS)
        p.notes.append("B: no drop found; using the fullest passage")
    elif -t.loss(drop) < WEAK_DROP_GAIN:
        p.notes.append("B: weak drop (the groove is already there before it)")
    place("B", drop)
    if drop >= LEAD_BARS:
        place("A", drop - LEAD_BARS)
    else:
        p.notes.append(f"A: B is only {drop} bars in")
    if n < MIN_BARS:
        p.notes.append(f"track too short ({n} bars) for the other cues")
        return p

    breakdown, note = _pick_breakdown(t, drop)
    if breakdown is None:
        p.notes.append("C/D: no breakdown or outro after B")
    else:
        place("D", breakdown)
        place("C", breakdown - LEAD_BARS)
        if note:
            p.notes.append(note)

    drops = [Moment(f"Drop {i}", bar) for i, bar in enumerate(later, 2)]
    loops, vocals, bridges = _loop_moments(t), _vocal_moments(t), _bridge_moments(t, drop)
    # Loops are the easiest to find, so they pick their bars last.
    wanted = (("H", drops), ("F", vocals), ("G", bridges), ("E", loops))
    free = sorted(pad for pad, moments in wanted if not place_best(pad, moments))
    # Pads left over go to whatever else was found, one kind at a time.
    rest = [m for group in zip_longest(drops, loops, vocals, bridges) for m in group if m]
    for pad in free:
        if not place_best(pad, rest):
            break
    return p
