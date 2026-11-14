"""Run the ml-stems cue rules on tracks saved by `scripts/stems_dump.py`.

    uv run python scripts/stems_review.py tmp/features                       # counts over all tracks
    uv run python scripts/stems_review.py tmp/features out.png "Fico|Babert" # plot matching tracks
    uv run python scripts/stems_review.py tmp/features out.png 40 8          # plot 8 tracks from the 40th

The plot shows, per bar, the level of each stem, the Raveform sections and the cues.
It is how the thresholds in `stempolicy.py` were tuned: change one, rerun, look.
"""
from __future__ import annotations

import collections
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from autohotcue import _stems, analysis
from autohotcue.backends import Segment
from autohotcue.stempolicy import propose_cues_stems

STEM_COLORS = {"drums": "tab:red", "bass": "tab:purple", "other": "tab:green", "vocals": "tab:blue"}
SECTION_COLORS = {
    "drop": "#ffb3b3", "breakdown": "#b3d9ff", "buildup": "#ffe0a3", "cooldown": "#d5c6f0",
    "bridge": "#b8e6b8", "intro": "#dddddd", "outro": "#dddddd",
}
MAX_PLOTS = 8


def load(npz: Path):
    """The name, bar features, sections and bar edges of a saved track."""
    z = np.load(npz, allow_pickle=False)
    bpm, render, anchor, ok = z["fit"]
    fit = SimpleNamespace(ok=bool(ok), bpm=float(bpm), render_bpm=float(render), anchor_s=float(anchor))
    duration = float(z["duration"])
    edges = _stems.bar_edges(z["beats"], z["downbeats"], fit, duration)
    if len(edges) < 2:
        edges = np.array([0.0, duration])
    powers = {
        name: z[f"env_{name}"].astype(np.float64) ** 2 for name in _stems.STEMS + ("mix",)
    }
    feats = _stems.bar_features(powers, z["mel"].astype(np.float32), edges, analysis.SR)
    segments = [
        Segment(float(s), float(e), str(label))
        for s, e, label in zip(z["seg_start"], z["seg_end"], z["seg_label"])
    ]
    return Path(str(z["path"])).stem, feats, segments, edges


def plot(tracks: list, out: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(tracks), 1, figsize=(18, 3.2 * len(tracks)), squeeze=False)
    for ax, (name, feats, segments, edges, prop) in zip(axes[:, 0], tracks):
        bars = len(feats.starts)
        bar_of = lambda t: float(np.interp(t, edges, np.arange(len(edges))))
        for seg in segments:
            a, b = bar_of(seg.start), bar_of(seg.end)
            ax.axvspan(a, b, color=SECTION_COLORS.get(seg.label, "#ffffff"), alpha=0.8, lw=0)
            ax.text(a + 0.3, 1.02, seg.label, fontsize=7, va="bottom")
        loud = max(float(np.percentile(feats.mix, 95)), 1e-9)
        for stem, color in STEM_COLORS.items():
            ax.plot(getattr(feats, stem) / loud, color=color, lw=1.2, label=stem)
        ax.plot(feats.mix / loud, color="black", lw=0.8, alpha=0.5, label="mix")
        for pad, t in prop.positions.items():
            bar = bar_of(t)
            label = pad if pad <= "D" else f"{pad}\n{prop.label(pad)}"
            ax.axvline(bar, color="black", lw=1.5)
            ax.text(bar, 0.9, label, fontsize=11, weight="bold", ha="center",
                    bbox=dict(fc="yellow", ec="black", pad=1))
        ax.set_xlim(0, bars)
        ax.set_ylim(0, 1.15)
        ax.set_xticks(np.arange(0, bars, 8))
        ax.tick_params(labelsize=6)
        ax.grid(axis="x", alpha=0.3)
        ax.set_title(f"{name}  ({bars} bars)", fontsize=9, loc="left")
    axes[0, 0].legend(fontsize=7, ncol=5, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=70)
    plt.close(fig)


def main() -> None:
    files = sorted(Path(sys.argv[1]).glob("*.npz"))
    pads, kinds, notes = collections.Counter(), collections.Counter(), collections.Counter()
    tracks = []
    for npz in files:
        name, feats, segments, edges = load(npz)
        prop = propose_cues_stems(feats, segments)
        pads.update(prop.positions.keys())
        kinds.update(prop.label(pad) for pad in prop.positions)
        notes.update(note.split(" (")[0] for note in prop.notes)
        tracks.append((name, feats, segments, edges, prop))
    print(len(files), "tracks")
    print("pads:", dict(sorted(pads.items())))
    print("cues:", dict(kinds.most_common()))
    print("notes:")
    for note, count in notes.most_common():
        print(f"  {count:4d}  {note}")
    if len(sys.argv) == 4:
        wanted = sys.argv[3].split("|")
        chosen = [t for t in tracks if any(w in t[0] for w in wanted)]
    elif len(sys.argv) == 5:
        chosen = tracks[int(sys.argv[3]):int(sys.argv[3]) + int(sys.argv[4])]
    else:
        return
    plot(chosen[:MAX_PLOTS], sys.argv[2])
    print("wrote", sys.argv[2])


if __name__ == "__main__":
    main()
