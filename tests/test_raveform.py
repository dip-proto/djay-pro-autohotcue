"""ml-raveform: weights conversion and cache, section naming, cue placement."""
from __future__ import annotations

import hashlib
import io
from types import SimpleNamespace

import numpy as np
import pytest

from autohotcue import _raveform
from autohotcue.backends import BeatAnalysis, Segment, StructureAnalysis
from autohotcue.ravepolicy import propose_cues_raveform

torch = pytest.importorskip("torch")

LOCK = f".{_raveform.MODEL_NAME}.lock"


def _checkpoint():
    state = {
        "embeddings.conv0.weight": torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 5),
        "norm.bias": torch.ones(3),
        "function_classifier.harmonix.classifier.weight": torch.zeros(10, 96),
        "function_classifier.harmonix.classifier.bias": torch.zeros(10),
        "function_classifier.raveform.classifier.weight": torch.ones(11, 96),
        "function_classifier.raveform.classifier.bias": torch.ones(11),
        "dataset_classifier.weight": torch.zeros(2, 96),
        "dataset_classifier.bias": torch.zeros(2),
    }
    config = {
        "model": "allinone",
        "data": {
            "name": "all",
            "raveform": {"name": "raveform", "num_labels": 11},
            "harmonix": {"name": "harmonix", "num_labels": 10},
        },
    }
    return state, config


def test_convert_keeps_only_the_raveform_head():
    state, config = _checkpoint()
    arrays, cfg = _raveform.convert_checkpoint(state, config)
    assert sorted(arrays) == [
        "embeddings.conv0.weight", "function_classifier.classifier.bias",
        "function_classifier.classifier.weight", "norm.bias",
    ]
    conv = state["embeddings.conv0.weight"].numpy()
    assert np.array_equal(arrays["embeddings.conv0.weight"], np.transpose(conv, (0, 2, 3, 1)))
    assert cfg["data"] == {"name": "raveform", "num_labels": 11}
    assert cfg["model"] == "allin1"

    with pytest.raises(ValueError, match="11-label"):
        _raveform.convert_checkpoint(
            {"function_classifier.classifier.weight": torch.zeros(10, 96)}, {"data": {"num_labels": 10}}
        )


def test_weights_are_built_once(tmp_path, monkeypatch):
    state, config = _checkpoint()
    calls = []

    def fake_download(dest):
        calls.append(dest)
        torch.save({"state_dict": state, "config": config}, dest)

    monkeypatch.setattr(_raveform, "_download_checkpoint", fake_download)
    folder = tmp_path / _raveform.MODEL_NAME
    assert _raveform.resolve_raveform_weights(tmp_path) == folder
    assert _raveform.resolve_raveform_weights(tmp_path) == folder
    assert len(calls) == 1
    assert sorted(p.name for p in tmp_path.iterdir()) == [LOCK, _raveform.MODEL_NAME]
    assert np.load(folder / f"{_raveform.MODEL_NAME}_mlx.npz")["norm.bias"].shape == (3,)
    assert "num_labels: 11" in (folder / f"{_raveform.MODEL_NAME}_mlx.yaml").read_text()


@pytest.mark.parametrize("fail_at", ["download", "convert"])
def test_a_failed_build_leaves_nothing_behind(tmp_path, monkeypatch, fail_at):
    state, config = _checkpoint()

    def fake_download(dest):
        if fail_at == "download":
            raise RuntimeError("network down")
        torch.save({"state_dict": state, "config": config}, dest)

    def broken_convert(state, config):
        raise ValueError("bad head")

    monkeypatch.setattr(_raveform, "_download_checkpoint", fake_download)
    if fail_at == "convert":
        monkeypatch.setattr(_raveform, "convert_checkpoint", broken_convert)
    with pytest.raises((RuntimeError, ValueError)):
        _raveform.resolve_raveform_weights(tmp_path)
    assert [p.name for p in tmp_path.iterdir()] == [LOCK]


@pytest.mark.parametrize("pinned", [True, False])
def test_download_checks_the_hash(tmp_path, monkeypatch, pinned):
    blob = b"checkpoint bytes"
    if pinned:
        monkeypatch.setattr(_raveform, "_SHA256", hashlib.sha256(blob).hexdigest())
    monkeypatch.setattr(_raveform.urllib.request, "urlopen", lambda url, timeout: io.BytesIO(blob))
    dest = tmp_path / "x.pth"
    if pinned:
        _raveform._download_checkpoint(dest)
        assert dest.read_bytes() == blob
    else:
        with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
            _raveform._download_checkpoint(dest)
        assert not dest.exists()


@pytest.mark.parametrize("seed", range(6))
def test_sections_match_all_in_one_mlx(seed):
    # all-in-one-mlx only names the 10 pop labels, so compare label numbers on a
    # 10-label output. Seed 0 has no boundary at all.
    pytest.importorskip("allin1_mlx")
    from allin1_mlx.config import HARMONIX_LABELS
    from allin1_mlx.postprocessing.functional_mlx import postprocess_functional_structure_mlx

    rng = np.random.default_rng(seed)
    cfg = SimpleNamespace(fps=100, hop_size=441, sample_rate=44100, min_hops_per_beat=24)
    n = 20000
    sections = 1 / (1 + np.exp(-np.convolve(rng.normal(-2, 3, n), np.ones(25) / 25, "same")))
    sections = sections.astype(np.float32) * (seed != 0)
    logits = rng.normal(size=(10, n)).astype(np.float32)
    functions = np.exp(logits) / np.exp(logits).sum(axis=0)

    ours = _raveform.raveform_segments(sections, functions, cfg, n * cfg.hop_size / cfg.sample_rate)
    ref = postprocess_functional_structure_mlx(
        None, cfg, prob_sections=sections, prob_functions=functions
    )
    assert [_raveform.RAVEFORM_LABELS.index(s.label) for s in ours] == [
        HARMONIX_LABELS.index(s.label) for s in ref
    ]
    assert np.allclose([(s.start, s.end) for s in ours], [(s.start, s.end) for s in ref])


def _beat() -> BeatAnalysis:
    return BeatAnalysis(
        bpm=120.0, beats=np.arange(0.0, 320.0, 0.5), downbeats=np.arange(0.0, 320.0, 2.0),
        duration_s=320.0, source="test",
    )


@pytest.mark.parametrize("sections, want", [
    pytest.param(
        [("start", 0, 2), ("altintro", 2, 32), ("buildup", 32, 64), ("drop", 64, 128),
         ("breakdown", 128, 160), ("buildup", 160, 176), ("drop", 176, 240),
         ("cooldown", 240, 272), ("altoutro", 272, 304), ("end", 304, 320)],
        {"A": 0, "B": 0, "C": 32, "D": 64, "E": 128, "F": 176, "G": 240, "H": 272},
        id="full-layout",
    ),
    pytest.param(
        [("drop", 0, 128), ("breakdown", 128, 160), ("drop", 160, 280), ("outro", 280, 320)],
        {"A": 0, "B": 0, "D": 160, "G": 280, "H": 280},
        id="grooves-from-bar-one",
    ),
    pytest.param(
        [("intro", 0, 80), ("breakdown", 80, 320)], {"A": 0, "B": 0}, id="no-drop",
    ),
    pytest.param(
        [("intro", 0, 32), ("drop", 32, 318), ("outro", 318, 320)],
        {"A": 0, "B": 0, "D": 32},
        id="outro-too-short",
    ),
    pytest.param(
        # The breakdown would snap onto the drop's downbeat.
        [("intro", 0, 32), ("drop", 32, 32.5), ("breakdown", 32.5, 64), ("drop", 64, 280),
         ("outro", 280, 320)],
        {"A": 0, "B": 0, "D": 32, "G": 280, "H": 280},
        id="shared-downbeat",
    ),
])
def test_cues(sections, want):
    structure = StructureAnalysis(
        segments=[Segment(start=a, end=b, label=lab) for lab, a, b in sections], source="test"
    )
    assert propose_cues_raveform(_beat(), structure).positions == want
