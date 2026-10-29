"""Cue placement for the ml-raveform engine.

The Raveform labels already say what each section is, so the cues simply follow
them.
"""
from __future__ import annotations

from autohotcue.analysis import CueProposal
from autohotcue.backends import BeatAnalysis, Segment, StructureAnalysis, bpm_octave_ratio
from autohotcue.cuepolicy import OUTRO_BEATS_MIN, _beats_after, _check_monotonicity, _first_beat

# Ambient intros and outros count as normal ones; start and end are silence.
_ALIASES = {"altintro": "intro", "altoutro": "outro"}
_SILENCE = frozenset({"start", "end"})
_MIX_OUT = frozenset({"cooldown", "outro"})


def section_runs(segments: list[Segment]) -> list[tuple[str, float, float]]:
    """Sections as (label, start, end), with silence dropped and repeats merged."""
    runs: list[tuple[str, float, float]] = []
    for seg in segments:
        role = _ALIASES.get(seg.label, seg.label)
        if role in _SILENCE:
            continue
        if runs and runs[-1][0] == role:
            runs[-1] = (role, runs[-1][1], seg.end)
        else:
            runs.append((role, seg.start, seg.end))
    return runs


def _next(starts: list[tuple[str, float]], role: str, after: float) -> int | None:
    return next((i for i, (r, t) in enumerate(starts) if r == role and t > after), None)


def propose_cues_raveform(
    beat: BeatAnalysis,
    structure: StructureAnalysis,
    djay_bpm: float | None = None,
) -> CueProposal:
    p = CueProposal()
    if djay_bpm is not None and bpm_octave_ratio(beat.bpm, djay_bpm) > 0.02:
        p.notes.append(f"djay says {djay_bpm:.1f}, tracked {beat.bpm:.1f}")

    a = _first_beat(beat)
    p.positions["A"] = a
    p.positions["B"] = a
    runs = section_runs(structure.segments)
    if len(beat.downbeats) < 8 or len(runs) < 2:
        p.notes.append("track too short for structure analysis; first beat only")
        return p
    # Snapped once, so two cues can never end up on the same downbeat.
    starts = [(role, beat.nearest_downbeat(start)) for role, start, _ in runs]

    # A track that grooves from bar one opens on a "drop" section; that isn't the drop.
    d = _next(starts, "drop", a)
    if d is None:
        p.notes.append("D (Drop): omitted (no drop after the first beat)")
        return p
    d_t = p.positions["D"] = starts[d][1]
    if d > 0 and starts[d - 1][0] == "buildup" and a < starts[d - 1][1] < d_t:
        p.positions["C"] = starts[d - 1][1]
    else:
        p.notes.append("C (Buildup): omitted (no buildup right before the drop)")

    last_t = d_t
    e = _next(starts, "breakdown", d_t)
    if e is not None:
        last_t = p.positions["E"] = starts[e][1]
        f = _next(starts, "drop", last_t)
        if f is not None:
            last_t = p.positions["F"] = starts[f][1]
        else:
            p.notes.append("F (2nd Drop): omitted (no drop after the breakdown)")
    else:
        p.notes.append("E (Breakdown): omitted (no breakdown after the drop)")
        p.notes.append("F (2nd Drop): omitted (no breakdown)")

    last_drop = max(i for i, (r, _) in enumerate(starts) if r == "drop")
    mix_out = [(r, t) for r, t in starts[last_drop + 1:] if r in _MIX_OUT and t > last_t]
    if not mix_out:
        p.notes.append("G/H (Outro): omitted (no cooldown or outro after the last drop)")
    elif _beats_after(mix_out[0][1], beat.beats) < OUTRO_BEATS_MIN:
        p.notes.append("G/H (Outro): omitted (outro guard: < 8 beats remaining)")
    else:
        p.positions["G"] = mix_out[0][1]
        p.positions["H"] = next((t for r, t in mix_out if r == "outro"), mix_out[0][1])

    _check_monotonicity(p)
    return p
