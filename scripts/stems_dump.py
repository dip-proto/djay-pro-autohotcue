"""Save what ml-stems measures on each track, so cue rules can be tried without the models.

    uv run python scripts/stems_dump.py ~/Music/djay/tracks tmp/features [COUNT]

Tracks are taken in a fixed random order, and tracks already saved are skipped,
so a run can be stopped and resumed, and a partial run is a fair sample.
Each track takes about 5 seconds and 1.5 MB.
`scripts/stems_review.py` reads the result.
"""
from __future__ import annotations

import hashlib
import random
import sys
import time
from pathlib import Path

import numpy as np

from autohotcue import _allin1, _raveform, _stems, analysis, backends, cli, djaydb
from autohotcue.gridlock import fit_grid


def dump(path: str, dest: Path, db: djaydb.DjayDB, separator, model) -> None:
    bpm = cli._djay_bpm_for_path(db, path, None)
    y = analysis.decode(path)
    beat = backends.track_beats(y, device=analysis._resolve_device(None))
    fit = fit_grid(y, analysis.SR, beat.beats, beat.downbeats, bpm)
    try:
        stems = _allin1._separate(path, separator)
        powers = {name: _stems.frame_power(_stems._mono(stems[name])) for name in _stems.STEMS}
        _, sections, functions = _allin1._probabilities(_allin1._spectrogram(stems, separator), model)
        del stems
        segments = _raveform.raveform_segments(sections, functions, model.cfg, beat.duration_s)
    finally:
        _allin1._clear_mlx_cache()
    powers["mix"] = _stems.frame_power(y)
    np.savez_compressed(
        dest,
        path=path,
        duration=beat.duration_s,
        beats=beat.beats,
        downbeats=beat.downbeats,
        fit=np.array([fit.bpm, fit.render_bpm, fit.anchor_s, float(fit.ok)]),
        djay_bpm=bpm or 0.0,
        mel=_stems.mel_db(y, analysis.SR).astype(np.float16),
        seg_start=np.array([s.start for s in segments]),
        seg_end=np.array([s.end for s in segments]),
        seg_label=np.array([s.label for s in segments]),
        **{f"env_{name}": np.sqrt(power).astype(np.float32) for name, power in powers.items()},
    )


def main() -> None:
    folder, out = sys.argv[1], Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    paths = cli._expand_paths(folder)
    random.Random(1).shuffle(paths)
    if len(sys.argv) > 3:
        paths = paths[: int(sys.argv[3])]
    db = djaydb.DjayDB()
    backends.init_worker(1)
    separator, model = _allin1._prepare(_raveform.resolve_raveform_weights(), _raveform.MODEL_NAME)
    for i, path in enumerate(paths, 1):
        dest = out / (hashlib.sha1(path.encode()).hexdigest()[:16] + ".npz")
        if dest.exists():
            continue
        started = time.time()
        try:
            dump(path, dest, db, separator, model)
            print(f"[{i}/{len(paths)}] {time.time() - started:.1f}s {Path(path).name}", flush=True)
        except Exception as exc:
            print(f"[{i}/{len(paths)}] FAILED {Path(path).name}: {exc!r}", flush=True)


if __name__ == "__main__":
    main()
