"""Unit tests for ml-songformer label mapping and engine validation (no model)."""
from __future__ import annotations

import sys

import numpy as np
import pytest

import autohotcue._songformer as mod
from autohotcue import analysis
from autohotcue._songformer import (
    SONGFORMER_LABELS,
    _MODEL_SR,
    _get_model,
    _map_songformer_segments,
    _resolve_device,
    _stub_msaf,
    segment_structure_songformer,
)
from autohotcue.backends import BeatAnalysis


def test_normalize_engine_ml_songformer():
    assert analysis.normalize_engine("ml-songformer") == ("ml-songformer", "songformer")


def test_ml_songformer_in_valid_engines():
    assert "ml-songformer" in analysis.VALID_ENGINES
    from autohotcue.bench import VALID_ENGINES

    assert "ml-songformer" in VALID_ENGINES


def test_effective_parallel_jobs_forces_single_worker_for_songformer():
    assert analysis.effective_parallel_jobs("ml-songformer", 8) == 1
    assert analysis.effective_parallel_jobs("ml", 8) == 8


def test_bench_parse_engines_includes_songformer():
    from autohotcue.bench import parse_engines

    assert parse_engines("ml,ml-songformer") == ["ml", "ml-songformer"]


def test_cli_engine_choices_includes_songformer():
    from autohotcue.cli import _ENGINE_CHOICES

    assert "ml-songformer" in _ENGINE_CHOICES


def test_songformer_label_mapping_happy_path():
    raw = [
        {"label": "intro", "start": 0.0, "end": 16.0},
        {"label": "verse", "start": 16.0, "end": 48.0},
        {"label": "chorus", "start": 48.0, "end": 80.0},
        {"label": "bridge", "start": 80.0, "end": 96.0},
        {"label": "inst", "start": 96.0, "end": 112.0},
        {"label": "outro", "start": 112.0, "end": 128.0},
    ]
    segs = _map_songformer_segments(raw, duration_s=160.0)
    assert [(s.label, s.start, s.end) for s in segs] == [
        ("intro", 0.0, 16.0),
        ("verse", 16.0, 48.0),
        ("chorus", 48.0, 80.0),
        ("bridge", 80.0, 96.0),
        ("inst", 96.0, 112.0),
        ("outro", 112.0, 128.0),
    ]
    assert set(s.label for s in segs).issubset(SONGFORMER_LABELS)


def test_songformer_label_case_and_prechorus_mapping():
    raw = [
        {"label": "INTRO", "start": 0.0, "end": 8.0},
        {"label": "pre-chorus", "start": 8.0, "end": 16.0},
        {"label": "prechorus", "start": 16.0, "end": 24.0},
        {"label": "CHORUS", "start": 24.0, "end": 32.0},
    ]
    segs = _map_songformer_segments(raw, duration_s=64.0)
    assert [s.label for s in segs] == ["intro", "verse", "verse", "chorus"]


def test_songformer_drops_silence_segments():
    raw = [
        {"label": "intro", "start": 0.0, "end": 1.0},
        {"label": "silence", "start": 1.0, "end": 1.2},
        {"label": "chorus", "start": 1.2, "end": 5.0},
    ]
    segs = _map_songformer_segments(raw, duration_s=10.0)
    assert [s.label for s in segs] == ["intro", "chorus"]


def test_songformer_silence_with_non_finite_boundary_raises():
    raw = [{"label": "silence", "start": float("nan"), "end": 1.2}]
    with pytest.raises(ValueError, match="non-finite segment boundary"):
        _map_songformer_segments(raw, duration_s=10.0)


def test_songformer_unknown_label_raises():
    raw = [{"label": "unknown_tag", "start": 0.0, "end": 8.0}]
    with pytest.raises(ValueError, match="unknown songformer segment label"):
        _map_songformer_segments(raw, duration_s=160.0)


def test_songformer_non_finite_boundary_raises():
    raw = [{"label": "verse", "start": float("nan"), "end": 8.0}]
    with pytest.raises(ValueError, match="non-finite segment boundary"):
        _map_songformer_segments(raw, duration_s=160.0)

    raw_inf = [{"label": "verse", "start": 0.0, "end": float("inf")}]
    with pytest.raises(ValueError, match="non-finite segment boundary"):
        _map_songformer_segments(raw_inf, duration_s=160.0)


def test_songformer_clamps_and_skips_degenerate():
    raw = [
        {"label": "intro", "start": -4.0, "end": 8.0},
        {"label": "verse", "start": 8.0, "end": 4.0},
        {"label": "chorus", "start": 0.0, "end": 0.0},
        {"label": "outro", "start": 150.0, "end": 200.0},
    ]
    segs = _map_songformer_segments(raw, duration_s=160.0)
    assert len(segs) == 2
    assert segs[0].label == "intro"
    assert segs[0].start == 0.0
    assert segs[0].end == 8.0
    assert segs[1].label == "outro"
    assert segs[1].start == 150.0
    assert segs[1].end == 160.0


def test_songformer_empty_input_returns_empty():
    assert _map_songformer_segments([], duration_s=120.0) == []


def test_resolve_device_default_and_override(monkeypatch):
    import torch

    monkeypatch.delenv("AUTOHOTCUE_SONGFORMER_DEVICE", raising=False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert _resolve_device() == "mps"
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert _resolve_device() == "cpu"

    monkeypatch.setenv("AUTOHOTCUE_SONGFORMER_DEVICE", "mps")
    assert _resolve_device() == "mps"


def test_get_model_raises_when_device_changes(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(mod, "_model", sentinel)
    monkeypatch.setattr(mod, "_model_device", "cpu")

    with pytest.raises(RuntimeError, match="ml-songformer model already loaded"):
        _get_model("mps")
    assert _get_model("cpu") is sentinel


@pytest.fixture
def clean_msaf_modules():
    saved = {k: sys.modules[k] for k in ("msaf", "msaf.eval") if k in sys.modules}
    for key in ("msaf.eval", "msaf"):
        sys.modules.pop(key, None)
    yield
    for key in ("msaf.eval", "msaf"):
        sys.modules.pop(key, None)
    sys.modules.update(saved)


def test_stub_msaf_idempotent_and_blocks_compute_results(clean_msaf_modules):
    _stub_msaf()
    _stub_msaf()
    assert "msaf" in sys.modules
    with pytest.raises(RuntimeError, match="msaf is stubbed out"):
        sys.modules["msaf"].eval.compute_results()


_TWO_SECONDS = BeatAnalysis(
    bpm=120.0,
    beats=np.linspace(0.0, 2.0, 5),
    downbeats=np.array([0.0, 1.0, 2.0]),
    duration_s=2.0,
    source="test",
)


_MODEL = object()


def _fake_forward(monkeypatch, forward):
    monkeypatch.setattr(mod, "_get_model", lambda _device: _MODEL)
    monkeypatch.setattr(mod, "_forward", forward)


def test_segment_structure_songformer_plumbing(monkeypatch):
    received: dict[str, object] = {}

    def forward(model, waveform):
        assert model is _MODEL
        received["waveform"] = waveform
        return [
            {"label": "INTRO", "start": 0.0, "end": 1.0},
            {"label": "silence", "start": 1.0, "end": 1.2},
            {"label": "chorus", "start": 1.2, "end": 5.0},
        ]

    _fake_forward(monkeypatch, forward)
    y = np.random.default_rng(0).standard_normal(44100 * 2).astype(np.float32)
    out = segment_structure_songformer("track.wav", y, 44100, _TWO_SECONDS)

    wf = received["waveform"]
    assert isinstance(wf, np.ndarray)
    assert wf.dtype == np.float32
    assert len(wf) == pytest.approx(2 * _MODEL_SR, rel=0.01)
    assert [s.label for s in out.segments] == ["intro", "chorus"]
    assert out.segments[-1].end == 2.0
    assert out.source == "songformer"
    assert all(0.0 <= s.energy_rank <= 1.0 for s in out.segments)


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (MemoryError("out of memory"), "SongFormer ran out of memory"),
        (RuntimeError("invalid memory format strides mismatch"), "strides mismatch"),
    ],
)
def test_segment_structure_songformer_errors(monkeypatch, error, message):
    def forward(model, waveform):
        assert model is _MODEL
        raise error

    _fake_forward(monkeypatch, forward)
    y = np.zeros(44100 * 2, dtype=np.float32)
    with pytest.raises(RuntimeError, match=message):
        segment_structure_songformer("track.wav", y, 44100, _TWO_SECONDS)


def test_sdpa_attention_matches_stock_attention():
    w2v = pytest.importorskip("transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer")
    import torch

    torch.manual_seed(0)
    config = w2v.Wav2Vec2ConformerConfig(
        hidden_size=64, num_attention_heads=4, position_embeddings_type="rotary"
    )
    attn = w2v.Wav2Vec2ConformerSelfAttention(config).eval()
    x = torch.randn(2, 300, 64)
    positions = w2v.Wav2Vec2ConformerRotaryPositionalEmbedding(config)(x)
    with torch.inference_mode():
        expected = attn(x, relative_position_embeddings=positions)[0]
        actual = mod._sdpa_attention(attn, x, relative_position_embeddings=positions)[0]
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def _tiny_frontend():
    """MuQ's frontend with fewer channels, which does not change how far each frame looks."""
    import torch
    from muq.muq.models.muq_model import MuQModel
    from muq.muq.modules.conv import Conv2dSubsampling
    from muq.muq.modules.features import MelSTFT

    class TinyFrontend:
        preprocessing = MuQModel.preprocessing
        normalize = MuQModel.normalize

        def __init__(self):
            self.preprocessor_melspec_2048 = MelSTFT(n_fft=2048, hop_length=240, is_db=True)
            self.stat = {"melspec_2048_mean": 0.0, "melspec_2048_std": 1.0}
            self.conv = Conv2dSubsampling(1, 8, 16, strides=[2, 2], n_bands=128).eval()

    torch.manual_seed(0)
    return TinyFrontend()


def test_tiled_frontend_and_chunk_reuse_match_single_passes(monkeypatch):
    pytest.importorskip("muq")
    import torch

    monkeypatch.setattr(mod, "_TILE_FRAMES", 300)
    monkeypatch.setattr(mod, "_encode", lambda enc, x: x)
    enc = _tiny_frontend()
    frame = mod._FRAME_SAMPLES
    chunk = mod._CHUNK_SAMPLES
    torch.manual_seed(1)
    for length in (300 * frame + 239, 300 * frame + 240, 308 * frame + 1, 2 * chunk,
                   3 * chunk + 5 * _MODEL_SR + 239):
        audio = torch.randn(2, length) * 0.1
        chunks = list(audio[0].split(chunk))
        with torch.inference_mode():
            window = mod._frontend(enc, audio)
            reused = mod._chunk_embeddings(enc, window[:1], chunks)
            single = mod._frontend_tile(enc, audio)
            per_chunk = torch.cat([mod._frontend_tile(enc, c.unsqueeze(0)) for c in chunks], 1)
        torch.testing.assert_close(window, single, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(reused, per_chunk, rtol=1e-5, atol=1e-5)
