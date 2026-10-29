"""Optional SongFormer structure backend (MPS when available; HF remote code pinned by revision).

We run our own copy of the model's forward pass, `_forward`.
It gives the same result, much faster and with far less memory.
"""
from __future__ import annotations

import contextlib
import importlib.machinery
import itertools
import math
import os
import sys
import types
from pathlib import Path

import numpy as np

from autohotcue import analysis
from autohotcue.backends import BeatAnalysis, Segment, StructureAnalysis, energy_ranks_for_segments

_MODEL_ID = "ASLP-lab/SongFormer"
_HF_REVISION = "a75880ed1b7375ac71860ec6c4fc9c899cf99515"
_MODEL_SR = 24000
SONGFORMER_LABELS = frozenset({"intro", "verse", "chorus", "bridge", "inst", "outro"})
_LABEL_MAP = {"pre-chorus": "verse", "prechorus": "verse"}
_DROP_LABELS = frozenset({"silence"})

# SongFormer only uses the output of the 10th encoder layer.
_EMBED_LAYER = 10
_CHUNK_SAMPLES = 30 * _MODEL_SR
_FRAME_SAMPLES = _MODEL_SR // 25
# Upstream skips pieces of audio this short.
_MIN_SAMPLES = 1024

# Cutting the audio between two frames only changes the 4 frames or so next to the cut.
# That is what lets the conv frontend run in tiles to save memory, and lets the 30 s chunks
# reuse the long window's frontend output.
# The tests fail if the margin drops below 4.
_HALO_FRAMES = 8
_TILE_FRAMES = 1000

_model = None
_model_device: str | None = None


def _import_error(exc: BaseException | None = None) -> RuntimeError:
    msg = (
        "ml-songformer requires optional deps installed:\n"
        "  uv sync --extra songformer\n"
        "First run downloads the SongFormer model (~2.8 GB one-time)."
    )
    if exc is not None:
        return RuntimeError(f"{msg}\n({type(exc).__name__}: {exc})")
    return RuntimeError(msg)


def _fix_ssl_env() -> None:
    """A socket-firewall wrapper overrides SSL_CERT_FILE with a CA that cannot
    validate huggingface.co; re-point trust to certifi from inside the process."""
    import certifi

    cert = os.environ.get("SSL_CERT_FILE", "")
    if "socketFirewallCa" in cert or not Path(cert or "/nonexistent").exists():
        os.environ["SSL_CERT_FILE"] = certifi.where()
        os.environ.pop("SSL_CERT_DIR", None)


def _stub_msaf() -> None:
    """model.py imports msaf only for its eval-time cal_metrics(); stub it so
    inference does not require the (numpy-2-incompatible) msaf package."""
    if "msaf" in sys.modules:
        return
    msaf = types.ModuleType("msaf")
    msaf_eval = types.ModuleType("msaf.eval")

    def _unavailable(**kwargs):
        raise RuntimeError("msaf is stubbed out; cal_metrics() is unavailable")

    msaf_eval.compute_results = _unavailable
    msaf.eval = msaf_eval
    msaf.__spec__ = importlib.machinery.ModuleSpec("msaf", None)
    msaf_eval.__spec__ = importlib.machinery.ModuleSpec("msaf.eval", None)
    sys.modules["msaf"] = msaf
    sys.modules["msaf.eval"] = msaf_eval


def _resolve_device() -> str:
    return analysis._resolve_device(os.environ.get("AUTOHOTCUE_SONGFORMER_DEVICE"))


def _snapshot_download() -> str:
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        return snapshot_download(_MODEL_ID, revision=_HF_REVISION, local_files_only=True)
    except LocalEntryNotFoundError:
        _fix_ssl_env()
        return snapshot_download(_MODEL_ID, revision=_HF_REVISION)


def _get_model(device: str):
    global _model, _model_device
    if _model is not None and _model_device != device:
        # Remote-code modules and SONGFORMER_LOCAL_DIR are process-global; a second
        # load would mix them and double peak memory.
        raise RuntimeError(
            f"ml-songformer model already loaded for device={_model_device!r}; "
            "one device per process, restart to switch"
        )
    if _model is None:
        try:
            import torch
            from transformers import AutoModel
        except ImportError as exc:
            raise _import_error(exc) from exc

        snapshot = _snapshot_download()
        os.environ["SONGFORMER_LOCAL_DIR"] = snapshot
        _stub_msaf()

        try:
            model = AutoModel.from_pretrained(
                _MODEL_ID,
                trust_remote_code=True,
                revision=_HF_REVISION,
            )
        except (ImportError, ModuleNotFoundError):
            sys.path.insert(0, snapshot)
            model = AutoModel.from_pretrained(
                _MODEL_ID,
                trust_remote_code=True,
                revision=_HF_REVISION,
            )

        model.eval()
        model.to(device)
        _model = model
        _model_device = device
    return _model


def _sdpa_attention(self, hidden_states, attention_mask=None,
                    relative_position_embeddings=None, output_attentions=False):
    """Same as transformers' conformer attention with rotary embeddings.

    It lets PyTorch pick a fused kernel instead of building the full attention matrix,
    which takes gigabytes on a 420 s window.
    """
    import torch.nn.functional as F

    if attention_mask is not None:
        raise ValueError("SongFormer attention is unmasked")
    batch, frames, _ = hidden_states.size()
    shape = (batch, frames, self.num_heads, self.head_size)
    query_key = self._apply_rotary_embedding(hidden_states, relative_position_embeddings)
    query = self.linear_q(query_key).view(shape).transpose(1, 2)
    key = self.linear_k(query_key).view(shape).transpose(1, 2)
    value = self.linear_v(hidden_states).view(shape).transpose(1, 2)
    out = F.scaled_dot_product_attention(query, key, value)
    return self.linear_out(out.transpose(1, 2).reshape(batch, frames, -1)), None


@contextlib.contextmanager
def _fused_attention(model):
    """Use _sdpa_attention in the encoder layers while the block runs."""
    attentions = [
        layer.self_attn
        for enc in (model.muq.model, model.musicfm)
        for layer in enc.conformer.layers[:_EMBED_LAYER]
    ]
    for attn in attentions:
        attn.forward = types.MethodType(_sdpa_attention, attn)
    try:
        yield
    finally:
        for attn in attentions:
            del attn.forward


def _frontend_tile(enc, audio):
    """Mel spectrogram and conv layers: (batch, samples) -> (batch, frames, 1024)."""
    import torch

    mel = enc.normalize(enc.preprocessing(audio, features=["melspec_2048"]))["melspec_2048"]
    return enc.conv(mel.unsqueeze(1).contiguous(memory_format=torch.channels_last))


def _frontend(enc, audio):
    """Same output as _frontend_tile, computed in overlapping tiles to use less memory."""
    import torch

    n = audio.shape[-1]
    tiles = []
    first = 0
    while True:
        skip = min(first, _HALO_FRAMES)
        lo = (first - skip) * _FRAME_SAMPLES
        hi = (first + _TILE_FRAMES + _HALO_FRAMES) * _FRAME_SAMPLES
        front = _frontend_tile(enc, audio[..., lo:hi])
        if hi >= n:
            tiles.append(front[:, skip:])
            return torch.cat(tiles, dim=1)
        tiles.append(front[:, skip : skip + _TILE_FRAMES])
        first += _TILE_FRAMES


def _encode(enc, x):
    positions = enc.conformer.embed_positions(x)
    for layer in enc.conformer.layers[:_EMBED_LAYER]:
        x = layer(x, relative_position_embeddings=positions)[0]
    return x


def _chunk_embeddings(enc, window_front, chunks):
    """Embeddings of back-to-back chunks that start where the window starts.

    Only the last chunk can be shorter than 30 s, and it gets its own frontend pass.
    """
    import torch

    edge_len = 2 * _HALO_FRAMES * _FRAME_SAMPLES
    full = [c for c in chunks if c.shape[-1] == _CHUNK_SAMPLES]
    count = len(full)
    parts = []
    if full:
        per_chunk = _CHUNK_SAMPLES // _FRAME_SAMPLES
        fronts = window_front[0, : count * per_chunk].reshape(count, per_chunk, -1).clone()
        slices = [c[:edge_len] for c in full] + [c[-edge_len:] for c in full]
        edges = _frontend_tile(enc, torch.stack(slices))
        fronts[:, :_HALO_FRAMES] = edges[:count, :_HALO_FRAMES]
        fronts[:, -_HALO_FRAMES:] = edges[count:, -_HALO_FRAMES:]
        parts.append(_encode(enc, fronts).reshape(1, -1, fronts.shape[-1]))
    parts += [_encode(enc, _frontend(enc, c.unsqueeze(0))) for c in chunks[count:]]
    return torch.cat(parts, dim=1)


def _window_logits(model, audio, start_s: int, dataset_ids, label_mask):
    import torch

    sr = _MODEL_SR
    window = audio[start_s * sr : (start_s + model.config.win_size) * sr]
    span = audio[start_s * sr : (start_s + model.config.hop_size) * sr]
    chunks = [c for c in span.split(_CHUNK_SAMPLES) if c.shape[-1] > _MIN_SAMPLES]

    short, long = [], []
    for enc in (model.musicfm, model.muq.model):
        front = _frontend(enc, window.unsqueeze(0))
        short.append(_chunk_embeddings(enc, front, chunks))
        long.append(_encode(enc, front))
    embeddings = short + long
    lengths = [e.shape[1] for e in embeddings]
    if max(lengths) - min(lengths) > 4:
        raise ValueError(f"SongFormer embedding lengths differ too much: {lengths}")
    joined = torch.cat([e[:, : min(lengths)] for e in embeddings], dim=-1)

    _, logits = model.songformer.infer(
        input_embeddings=joined,
        dataset_ids=dataset_ids,
        label_id_masks=label_mask,
        with_logits=True,
    )
    return logits["function_logits"][0].cpu().numpy(), logits["boundary_logits"][0].cpu().numpy()


def _forward(model, waveform: np.ndarray) -> list[dict]:
    """Same result as the model's own forward pass at the pinned revision.

    The windows, the averaging and the post-processing all come from the model's code.
    """
    import torch

    remote = sys.modules[type(model).__module__]
    config = model.config
    device = next(model.parameters()).device
    audio = torch.from_numpy(waveform).to(device)
    n = audio.shape[-1]
    sr = _MODEL_SR
    rate = remote.AFTER_DOWNSAMPLING_FRAME_RATES

    total_frames = math.ceil((n // sr // remote.TIME_DUR + 1) * remote.TIME_DUR * rate)
    function = np.zeros([total_frames, config.num_classes])
    boundary = np.zeros([total_frames])
    count = np.zeros([total_frames])

    dataset_ids = torch.tensor(remote.DATASET_IDS, dtype=torch.long, device=device)
    mask = model.dataset_id2label_mask[remote.DATASET_LABEL_TO_DATASET_ID[remote.DATASET_LABEL]]
    label_mask = torch.from_numpy(mask).to(device).view(1, 1, -1)

    lens = 0
    with _fused_attention(model):
        for start_s in itertools.count(0, config.hop_size):
            # Upstream gets stuck on a very short tail, so we drop it.
            if n - start_s * sr <= _MIN_SAMPLES:
                break
            function_logits, boundary_logits = _window_logits(
                model, audio, start_s, dataset_ids, label_mask
            )
            first = int(start_s * rate)
            last = first + min(math.ceil(config.hop_size * rate), boundary_logits.shape[0])
            function[first:last] += function_logits
            boundary[first:last] = boundary_logits
            count[first:last] += 1
            lens += last - first

    # As upstream, the last frame or two of a long track are never written and end up NaN.
    with np.errstate(invalid="ignore"):
        function = function[:lens] / count[:lens, None]
        boundary = boundary[:lens] / count[:lens]
    logits = {
        "function_logits": torch.from_numpy(function).unsqueeze(0),
        "boundary_logits": torch.from_numpy(boundary).unsqueeze(0),
    }
    sections = remote.postprocess_functional_structure(logits, config)
    if sections[-1][-1] != "end":
        raise RuntimeError(f"SongFormer post-processing did not end with 'end': {sections[-1]!r}")
    if not config.no_rule_post_processing:
        sections = remote.rule_post_processing(sections)
    return [
        {"label": label, "start": start, "end": sections[k + 1][0]}
        for k, (start, label) in enumerate(sections[:-1])
    ]


def _map_songformer_segments(raw_segments: list, duration_s: float) -> list[Segment]:
    _known_labels = SONGFORMER_LABELS | _DROP_LABELS
    segments: list[Segment] = []
    for seg in raw_segments:
        label = str(seg["label"]).lower()
        label = _LABEL_MAP.get(label, label)
        if label not in _known_labels:
            raise ValueError(
                f"unknown songformer segment label {seg['label']!r} "
                f"(expected one of {sorted(SONGFORMER_LABELS)})"
            )
        start = float(seg["start"])
        end = float(seg["end"])
        if not (np.isfinite(start) and np.isfinite(end)):
            raise ValueError(
                f"non-finite segment boundary for {seg['label']!r}: "
                f"start={seg['start']!r}, end={seg['end']!r}"
            )
        if label in _DROP_LABELS:
            continue
        start = max(0.0, start)
        end = min(duration_s, end)
        if end > start:
            segments.append(Segment(start=start, end=end, label=label))
    return segments


def segment_structure_songformer(
    path: str,
    y: np.ndarray,
    sr: int,
    beat: BeatAnalysis,
) -> StructureAnalysis:
    """Structure segments from HuggingFace ASLP-lab/SongFormer."""
    del path  # ffmpeg-decoded array is the input; path kept for API stability
    import librosa

    try:
        import torch
    except ImportError as exc:
        raise _import_error(exc) from exc

    device = _resolve_device()
    model = _get_model(device)
    y_model = librosa.resample(y, orig_sr=sr, target_sr=_MODEL_SR)
    waveform = np.asarray(y_model, dtype=np.float32)
    try:
        try:
            with torch.inference_mode():
                raw_segments = _forward(model, waveform)
        except (MemoryError, RuntimeError) as exc:
            if isinstance(exc, MemoryError) or any(
                token in str(exc).lower()
                for token in ("out of memory", "can't allocate", "failed to allocate",
                              "not enough memory")
            ):
                raise RuntimeError(
                    "SongFormer ran out of memory; close memory-heavy apps or use "
                    "--engine ml-allin1 instead."
                ) from exc
            raise
        segments = _map_songformer_segments(raw_segments, beat.duration_s)
        segments = energy_ranks_for_segments(y, sr, segments)
        return StructureAnalysis(segments=segments, source="songformer")
    finally:
        try:
            torch.mps.empty_cache()
        except (AttributeError, RuntimeError):
            pass
