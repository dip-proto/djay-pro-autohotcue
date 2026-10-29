# ml-songformer structure backend

Optional structure segmentation via [SongFormer](https://github.com/ASLP-lab/SongFormer)
(ASLP-lab), an MSA model that fuses MuQ + MusicFM SSL embeddings at 30 s / 420 s windows.
Beat/downbeat tracking stays **beat_this**; SongFormer supplies labeled segments only
(intro/verse/chorus/bridge/inst/outro). Cue placement still snaps to beat_this downbeats
via `cuepolicy.py`.

## Why SongFormer

SongFormer leads HarmonixSet boundary accuracy among recent MSA models
(HR.5F **0.703** vs All-In-One **0.596**, [arXiv:2510.02797](https://arxiv.org/abs/2510.02797)).
autohotcue keeps beat_this for beats/downbeats and uses SongFormer for structure labels only.

## Model

| Item | Value |
|------|-------|
| HuggingFace ID | `ASLP-lab/SongFormer` |
| Pinned revision | `a75880ed1b7375ac71860ec6c4fc9c899cf99515` (2026-05-14) |
| Loader | `transformers.AutoModel.from_pretrained(..., trust_remote_code=True)` |
| Download | ~2.8 GB one-time into the HF cache |
| License | CC BY 4.0 (code + weights); attribution: SongFormer, ASLP-lab |

The pinned revision protects against upstream changes to the remote code.
`_forward` is written against this exact revision, so moving to a new one means checking
it again with the parity test (see "Accuracy").

## Prerequisites

- `uv sync --extra songformer`
- `transformers>=4.51,<5` (transformers 5.x meta-device init breaks the bundled
  MusicFM torchaudio frontend)
- `ffmpeg` on `PATH`
- First run downloads the SongFormer weights (one-time network)

## Usage

```bash
uv sync --extra songformer
uv run autohotcue propose "/path/to/track.opus" --engine ml-songformer
```

Engines: `ml-bass` (default), `ml` / `ml-librosa`, `ml-allin1`, `ml-raveform`,
`ml-songformer`, `legacy`.

```bash
uv run autohotcue bench truth.json --engines ml,ml-allin1,ml-songformer,legacy
```

## Device

SongFormer runs on the Apple-Silicon GPU (MPS) when PyTorch can see one, and on the CPU
otherwise.

Set `AUTOHOTCUE_SONGFORMER_DEVICE=cpu` or `mps` to pick one yourself.
This also applies to bench and e2e runs, because the `device` flag of `analyze()` only
affects beat tracking, as with ml-allin1.

## Forward pass

We don't call the model's own forward pass.
`_forward` in `_songformer.py` does the same job: same 420 s windows, same 30 s chunks,
same averaging, and it calls the model's own post-processing at the end.

What changes is how the work gets done:

1. Attention goes through PyTorch's fused attention, which doesn't need the full attention
   matrix.
   On a 420 s window that matrix took 7 GB in every layer.

2. SongFormer only uses the output of the 10th layer of each encoder, so we stop there
   instead of running all 12.

3. The conv layers in front of each encoder cost more than the encoder itself.
   Their input now uses a memory layout that makes them 20 to 35% faster on the GPU.

4. Those conv layers only look a few frames around each point, so cutting the audio only
   changes about 4 frames next to the cut.
   That lets them run on 40 s pieces that overlap a little, which keeps memory low.

5. For the same reason, the 30 s chunks reuse the conv output of the whole window.
   Only a few frames at the edges of each chunk get recomputed.
   This saves about a quarter of the time.

6. The full 30 s chunks go through the encoders together, in one batch.

The model's own code also gets stuck in an endless loop when less than 43 ms of audio is
left after a 420 s window.
`_forward` just drops that bit.

## Accuracy

Accuracy matters more than speed here.
Every change above gives the same result, up to tiny rounding differences.

We compared `_forward` with the model's own forward pass on 20 tracks from the library,
from 4 to 56 minutes long, and on cuts of one track at 20, 60.5, 90, 420, 421 and 450 s.
The segments were the same every time.

On the same device, the raw scores differ by at most 7e-6.
For comparison, the model's own code gives scores that differ by up to 4.9e-4 between the
GPU and the CPU.

Half precision (f16) would make it about 25% faster, but it moves the scores by up to 1.2.
That is too much to trust, so we don't use it.

`tests/test_songformer.py` checks the new attention and the conv shortcuts against the
straightforward computation.
The test fails if the overlap gets too small.

`test_songformer_forward_matches_upstream_forward` in `tests/test_e2e.py` runs both
forward passes on a real track and on cuts of it, and expects the same segments.
Turn it on with `AUTOHOTCUE_RUN_SONGFORMER_E2E=1`.

## Runtime (M5 Max 128 GB, Python 3.14, torch 2.14.1, 2026-10-29)

Time for the forward pass only, with the model already loaded (loading takes about 2 s):

| Track | Length | old, CPU | old, MPS | new, MPS | new, CPU |
|-------|--------|----------|----------|----------|----------|
| Eugenio Fico - Funky Riders | 307 s | 83 s | 6.7 s | 2.9 s | 34 s |
| Uncle Dog - Eatwhite | 514 s | 134 s | 10.6 s | 5.0 s | 57 s |
| DJ 3000 - Somewhere In Detroit Mix | 56 min | not run | 94 s | 44 s | not run |

A full `propose` with `--engine ml-songformer` on a 4.5-minute track takes about 7 s,
startup included.
It used to take a minute.

`effective_parallel_jobs` forces `jobs=1` for `ml-songformer`, because there is one model
per process and the analysis stays in the main process.

## Memory

| | old | new |
|-|-----|-----|
| GPU memory (MPS) | 23 to 35 GB | 7.2 GB |
| Peak RAM (CPU) | 23 to 41 GB | 11 to 12.5 GB |

Most of the old CPU peak came from the conv layers.
On a 420 s window they needed 23 GB on their own, and running them in pieces fixed that.

On the GPU, the process itself stays around 1 GB.
The GPU memory is taken from the same RAM, since it is shared on Apple Silicon.

## Implementation notes

1. **Remote code + `SONGFORMER_LOCAL_DIR`**: `modeling_songformer` reads bundled config
   from the HF snapshot; the backend sets `SONGFORMER_LOCAL_DIR` automatically after
   `snapshot_download`.
2. **Multi-file repo import**: a direct `AutoModel` import of the multi-file repo fails;
   the backend retries after inserting the snapshot dir on `sys.path`.
3. **`msaf` stub**: remote `model.py` imports `msaf` only for eval-time `cal_metrics()`;
   the backend stubs `msaf` in `sys.modules` (`msaf` is not installable on numpy 2 /
   Python 3.13).
4. **Trailing NaN logits**: on tracks longer than 420 s, the last frame or two never get
   a score and end up NaN.
   The model's own code does the same.
   The segment mapper rejects any boundary that isn't a real number.
5. **TLS-intercepting firewalls**: machines with a socket-firewall override
   `SSL_CERT_FILE` in child processes; the backend re-points trust to certifi before
   the one-time download.
6. **Label mapping**: output labels `intro` / `verse` / `chorus` / `bridge` / `inst` /
   `outro`; `pre-chorus` becomes `verse`; `silence` segments are dropped.

## Status / open threads

- **Quantitative bench** (`ml` vs `ml-allin1` vs `ml-songformer`): blocked on the
  ground-truth labeling task (`truth.json` does not exist yet); plan and decision rule in
  [docs/roadmap/03-songformer-bench.md](roadmap/03-songformer-bench.md). Until then the
  evidence is qualitative: musically sensible cue layouts on Afro House tracks (Drop on
  the first chorus, Breakdown on the bridge) plus the paper's boundary numbers.
- The memory and speed problem (roadmap item 07) is solved, see "Forward pass" above.
