"""Section labels from the Raveform model, for the ml-raveform engine.

This is the all-in-one model trained on dance music, with labels like intro,
buildup, drop and breakdown.
It runs through all-in-one-mlx, so it needs `uv sync --extra allin1`.

The weights are downloaded and converted on first use.
"""
from __future__ import annotations

import fcntl
import hashlib
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

from autohotcue import _allin1
from autohotcue.backends import BeatAnalysis, Segment, StructureAnalysis

# The authors haven't published this order; it was worked out by hand.
# Read docs/raveform-backend.md before changing it.
RAVEFORM_LABELS = (
    "start", "end", "altintro", "altoutro", "intro", "outro",
    "breakdown", "buildup", "cooldown", "bridge", "drop",
)

_REPO = "taejunkim/allinone"
_REVISION = "379e5fd010b3fdd0ee8381ff8cbcfa51d70b5c19"
_CHECKPOINT = "all-fold0-40pa2vpn.pth"
_SHA256 = "c5efdcf2cf48dbfd68611b840b5f644fa67d819e24bc7d459113a89ebd3e5587"
MODEL_NAME = "raveform-all-fold0"


def convert_checkpoint(state: dict, config: dict) -> tuple[dict[str, np.ndarray], dict]:
    """Convert a PyTorch checkpoint to what all-in-one-mlx loads.

    Combined checkpoints keep only their Raveform head.
    """
    if "function_classifier.raveform.classifier.weight" in state:
        state = {
            k.replace("function_classifier.raveform.", "function_classifier."): v
            for k, v in state.items()
            if not k.startswith(("function_classifier.harmonix.", "dataset_classifier."))
        }
        config = dict(config, data=config["data"]["raveform"], model="allin1")
    num_labels = state["function_classifier.classifier.weight"].shape[0]
    if num_labels != len(RAVEFORM_LABELS) or config["data"]["num_labels"] != num_labels:
        raise ValueError(f"expected a {len(RAVEFORM_LABELS)}-label Raveform head, got {num_labels}")
    arrays = {}
    for k, v in state.items():
        a = v.numpy()
        arrays[k] = np.transpose(a, (0, 2, 3, 1)) if a.ndim == 4 else a
    return arrays, config


def _download_checkpoint(dest: Path) -> None:
    url = f"https://huggingface.co/{_REPO}/resolve/{_REVISION}/{_CHECKPOINT}"
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            blob = resp.read()
    except OSError as exc:
        raise RuntimeError(f"could not download the Raveform checkpoint from {url}: {exc}") from exc
    digest = hashlib.sha256(blob).hexdigest()
    if digest != _SHA256:
        raise RuntimeError(f"Raveform checkpoint SHA-256 mismatch: got {digest}, expected {_SHA256}")
    dest.write_bytes(blob)


def resolve_raveform_weights(cache_dir: Path | None = None) -> Path:
    """Return the folder with the converted weights, building it on first use.

    The files are built in a scratch folder and moved into place in one step, so a
    failed build leaves nothing behind.
    """
    root = Path(cache_dir) if cache_dir is not None else _allin1.weights_cache()
    final = root / MODEL_NAME
    if weights_ready(root):
        return final
    try:
        import torch
        import yaml
    except ImportError as exc:
        raise _allin1._import_error(exc) from exc
    root.mkdir(parents=True, exist_ok=True)
    with open(root / f".{MODEL_NAME}.lock", "w") as lock:
        # Another run may be building it right now; wait, then check again.
        fcntl.flock(lock, fcntl.LOCK_EX)
        if weights_ready(root):
            return final
        tmp = Path(tempfile.mkdtemp(prefix=f".{MODEL_NAME}-", dir=root))
        try:
            ckpt = tmp / _CHECKPOINT
            _download_checkpoint(ckpt)
            loaded = torch.load(ckpt, map_location="cpu", weights_only=True)
            arrays, config = convert_checkpoint(loaded["state_dict"], loaded["config"])
            ckpt.unlink()
            np.savez(tmp / f"{MODEL_NAME}_mlx.npz", **arrays)
            (tmp / f"{MODEL_NAME}_mlx.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
            if final.exists():
                shutil.rmtree(final)  # incomplete, rebuild it
            os.rename(tmp, final)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
    return final


def weights_ready(cache_dir: Path | None = None) -> bool:
    folder = (Path(cache_dir) if cache_dir is not None else _allin1.weights_cache()) / MODEL_NAME
    return all((folder / f"{MODEL_NAME}_mlx.{ext}").is_file() for ext in ("npz", "yaml"))


def raveform_segments(
    prob_sections: np.ndarray, prob_functions: np.ndarray, cfg, duration_s: float
) -> list[Segment]:
    """Sections from the model output.

    Same steps as all-in-one-mlx, which only knows the pop label names.
    """
    from allin1_mlx.postprocessing.helpers import event_frames_to_time, peak_picking

    boundary = peak_picking(
        boundary_activation=prob_sections, window_past=12 * cfg.fps, window_future=12 * cfg.fps
    ) > 0.0
    model_duration = len(prob_sections) * cfg.hop_size / cfg.sample_rate
    times = event_frames_to_time(boundary, cfg)
    if len(times) == 0 or times[0] != 0:
        times = np.insert(times, 0, 0.0)
    if times[-1] != model_duration:
        times = np.append(times, model_duration)
    cuts = np.flatnonzero(boundary)
    cuts = cuts[cuts > 0]
    labels = [int(p.mean(axis=1).argmax()) for p in np.split(prob_functions, cuts, axis=1)]
    segments = []
    for start, end, label in zip(times[:-1], times[1:], labels):
        start, end = max(0.0, float(start)), min(duration_s, float(end))
        if end > start:
            segments.append(Segment(start=start, end=end, label=RAVEFORM_LABELS[label]))
    return segments


def segment_structure_raveform(
    path: str,
    y: np.ndarray,
    sr: int,
    beat: BeatAnalysis,
) -> StructureAnalysis:
    """Labeled sections from the Raveform model."""
    separator, model = _allin1._prepare(resolve_raveform_weights(), MODEL_NAME)
    try:
        spec = _allin1._track_spectrogram(path, separator)
        _, sections, functions = _allin1._probabilities(spec, model)
        segments = raveform_segments(sections, functions, model.cfg, beat.duration_s)
        return StructureAnalysis(segments=segments, source="raveform")
    finally:
        _allin1._clear_mlx_cache()
