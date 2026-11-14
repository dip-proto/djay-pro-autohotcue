# AGENTS.md

Guidance for coding agents working on **autohotcue**. Read this before changing
code — this project writes directly into a DJ app's real, irreplaceable music
library, so some rules below are load-bearing, not stylistic.

## Project overview

autohotcue places "hot cue" markers on tracks automatically and writes them
into **djay Pro**'s library database. It analyzes audio structure directly
(any `ffmpeg`-decodable format, including `.opus`/`.ogg`), so it does not depend
on another DJ app's analysis.

Pure-Python, single package, no service. Default pipeline (`ml-stems`): decode
audio, **beat_this** beat/downbeat tracking, `gridlock` straight-grid fit,
**demucs** stems + **Raveform** section labels (`_stems`), drop / breakdown cue
layout (`stempolicy`), then write the cues into djay's `MediaLibrary.db`.
Pads A and B are 8 bars before the drop and the drop, C and D 8 bars before the
breakdown and the breakdown, E to H a loop, a vocals-only passage, a bridge and
the second drop (`docs/stems-engine.md`). It runs on MLX, so Apple Silicon only.
The other engines place the older djcues layout (First Beat ... Loop Out):
`--engine ml-bass` uses kick-band (30 to 150 Hz) bass-in/out/return detection +
phrase snapping (`bassline`) and runs anywhere, `--engine ml`/`ml-librosa` use librosa
Laplacian segmentation + the pure cue policy, `--engine ml-allin1` swaps structure
for **all-in-one-mlx** (Apple Silicon),
`--engine ml-raveform` runs the same model trained on **Raveform** dance tracks,
which labels drops, breakdowns and buildups (weights downloaded on first use),
`--engine ml-songformer` swaps structure for **SongFormer** (MPS when available;
`uv sync --extra songformer`). A legacy band-energy engine remains behind
`--engine legacy`.

```
src/autohotcue/
    tsaf.py      # parser + serializer for djay's undocumented "TSAF" blob format
    djaydb.py    # MediaLibrary.db (SQLite/YapDatabase) reader/writer, backups, cue builder
    backends.py  # beat_this + librosa / allin1 / songformer structure backends
    _stems.py    # per-bar stem levels + Raveform sections (ml-stems engine, the default)
    stempolicy.py # pure cue policy for ml-stems: drop, breakdown, loop, vocals, bridge
    _allin1.py   # all-in-one-mlx structure (ml-allin1 engine) + the shared demucs separator
    _raveform.py # Raveform-trained all-in-one structure (ml-raveform and ml-stems)
    _songformer.py  # optional SongFormer structure (ml-songformer engine)
    gridlock.py  # pure straight-grid fit + cue snapping (beat/downbeat -> lattice)
    bassline.py  # pure kick-band bass events + phrase snapping (ml-bass policy)
    cuepolicy.py # pure structure-based cue-placement policy (no audio, no I/O)
    ravepolicy.py # pure cue policy for Raveform section labels (ml-raveform)
    analysis.py  # decode, analyze() dispatcher, legacy grid path
    bench.py     # ground-truth eval harness (hit-rate, MAE, runtime)
    viz.py       # waveform + segment/bass spans + downbeat ticks + cue-map PNG
    cli.py       # propose / viz / apply / verify / bench
scripts/
    stems_dump.py    # save what ml-stems measures per track (for tuning the rules)
    stems_review.py  # rerun the ml-stems rules on those dumps: counts and plots
tests/
    test_tsaf.py     # byte-exact round-trip of the whole real library + edge cases
    test_gridlock.py # straight-grid fit, snapping, beatGridEdits encoding
    test_stempolicy.py # ml-stems cue rules on made-up tracks
    test_stems.py    # ml-stems bar features and wiring, models faked
    test_bassline.py # ml-bass kick-band events + phrase snapping + slot rules
    test_cuepolicy.py
    test_predrop.py  # pad C placed N bars before the drop
    test_raveform.py # ml-raveform weights, section naming and cues
    test_backends.py
    test_bench.py
    test_e2e.py      # real-track integration (skipped when paths missing)
```

Key tech: Python 3.14 (pinned; `requires-python >=3.11`), [uv](https://docs.astral.sh/uv/)
for env/deps, numpy + librosa + soundfile + matplotlib + **PyTorch** (via
`beat-this`), MLX with demucs-mlx and all-in-one-mlx on Apple Silicon, and the external
`ffmpeg` binary (must be on `PATH`). First run downloads the beat_this, demucs and
Raveform weights (one-time network).

## Setup commands

- Install deps + create venv: `uv sync` (on Apple Silicon this includes MLX, demucs-mlx
  and all-in-one-mlx, which the default engine needs)
- Requires `ffmpeg` on `PATH` (`ffmpeg -version`); install via Homebrew: `brew install ffmpeg`
- Python is pinned in `.python-version` (3.14.10); `uv` provisions it.
  Python 3.15 doesn't work yet: PyTorch has no build for it, and every engine
  except `legacy` needs it.

## Development workflow

Run the CLI through uv (no manual venv activation needed):

```bash
uv run autohotcue propose "/path/to/track.opus"              # ml-stems engine (default)
uv run autohotcue propose "/path/to/track.opus" --engine ml-bass    # kick-band events, djcues layout
uv run autohotcue propose "/path/to/track.opus" --engine ml         # librosa structure
uv run autohotcue propose "/path/to/track.opus" --engine ml-allin1  # all-in-one structure
uv run autohotcue propose "/path/to/track.opus" --engine ml-songformer  # SongFormer structure
uv run autohotcue propose "/path/to/track.opus" --nudge-beats 1     # shift grid anchor +1 beat
uv run autohotcue propose "/path/to/track.opus" --engine ml-bass --pre-drop-bars 16  # pad C 16 bars before the drop (0 = off)
uv run autohotcue propose "/path/to/track.opus" --engine legacy
uv run autohotcue viz "/path/to/track.opus" map.png          # segments + downbeat ticks
uv run autohotcue verify "/path/to/track.opus"               # read back cues djay has stored
uv run autohotcue apply "/path/to/track.opus"                # WRITE cues into djay's DB
uv run autohotcue bench truth.json --engines ml,ml-allin1,legacy -j 4
```

`--engine {ml-stems,ml-bass,ml,ml-librosa,ml-allin1,ml-raveform,ml-songformer,legacy}` (default `ml-stems`).
`ml-stems` (the default) reads Raveform sections and demucs stems and places the drop /
breakdown layout; read `docs/stems-engine.md` before changing `stempolicy.py`, its
thresholds were tuned on the user's library. `--pre-drop-bars` does not apply to it.
`ml-bass` uses kick-band bass events + 16/32-bar phrase snapping on the
grid-locked lattice; `ml` / `ml-librosa` use librosa Laplacian structure; `ml-allin1`
uses all-in-one-mlx (see `docs/allin1-backend.md`); `ml-raveform` uses the
all-in-one model trained on dance music, whose label order was worked out by hand
(read `docs/raveform-backend.md` before touching `RAVEFORM_LABELS`); `ml-songformer` uses
SongFormer (optional extra; see `docs/songformer-backend.md`; MPS when available). djay's BPM
is cross-checked only under ml engines; placement uses beat_this downbeats. `apply` still
writes absolute seconds.

Use `--library <path>` to target a copy of the DB (do this when testing writes).

`propose`, `apply` and `verify` also accept a **folder** (scanned recursively
for audio files). A folder `apply` takes one backup for the whole run and
reports per-track skips/failures plus a summary instead of aborting.
`-j/--jobs N` fans the audio analysis out to N worker processes (0 = one per
core); the database is only ever touched from the main thread — keep it that
way if you change the parallel path.

### Parallelism rules

- `ProcessPoolExecutor` workers use `initializer=backends.init_worker` to cap
  torch threads per worker.
- Workers always run models on **cpu**; **mps** is used only when effective
  jobs == 1 (single-threaded analysis).
- One lazy model singleton per worker process (macOS spawn reloads each worker).
- Database writes stay on the main thread only.

## Testing instructions

- Run everything: `uv run pytest` (config pins `testpaths = ["tests"]`).
- Focus one test: `uv run pytest -q -k roundtrip`.
- The headline test (`test_roundtrip_entire_library`) parses **and re-serializes
  every blob** in the local djay library and asserts byte-for-byte equality. It
  **skips** automatically when no djay library is present (e.g. CI) — the
  synthetic tests still run. If you have djay installed, this test must pass.
- `tests/test_e2e.py` runs against real tracks when present on the machine.
- Add or update tests for any change to `tsaf.py` or `djaydb.py`.

## The non-negotiable safety invariant

`tsaf.serialize(tsaf.parse(blob)) == blob` must hold for **every** blob in a
real djay library. This byte-exact round-trip is the only thing that makes
writing to the live database safe. If you touch `tsaf.py`:

1. Run the full-library round-trip test before and after.
2. Never make the serializer "lossy but close" — a one-byte drift can corrupt
   the user's entire catalog.
3. The write path (`djaydb.put`, `cli.cmd_apply`) already refuses to edit any
   record it cannot reproduce byte-exact (the "faithful-edit guard"). Do not
   weaken or bypass that guard.

Other write-path rules already enforced — keep them:
- **djay must be quit** before `apply` (it holds the DB open); it is re-checked
  immediately before the write.
- Every `apply` run takes a **unique, timestamped backup** of the DB
  (+ `-wal`/`-shm`) before its first write (one backup per folder batch).
- Track lookup matches the **exact** `file://` source URL and refuses ambiguous
  matches (phantom location records without a titleID lose to the real entry) —
  never write cues to the wrong track.

## Code style

- Match the surrounding code: type hints, `from __future__ import annotations`,
  small focused functions, stdlib-first.
- Comments only state constraints the code can't show (e.g. the TSAF tag table,
  alignment rules). Don't narrate what the next line does.
- No formatter/linter is configured; follow existing formatting (4-space indent,
  ~100-col lines, double-quoted strings).
- Keep `tsaf.py` free of project-specific logic — it is a generic codec.

## Build and deployment

There is no build/deploy step — it's a local CLI. `uv build` produces a wheel if
needed. No CI is configured yet; if you add one, wire `uv sync` + `uv run pytest`
(the library round-trip test will skip in CI, which is expected).

## Pull request / commit guidelines

- Branch off and target `main`.
- Commits in this repo use a noreply identity (no personal email in history);
  preserve that convention.
- Before committing: `uv run pytest` must be green.
- Never commit `.venv/`, the user's `MediaLibrary.db`, or any backup of it.

## Gotchas

- `.opus`/`.ogg` are first-class here precisely because rekordbox/Serato reject
  them — don't assume a format is unsupported; if `ffmpeg` decodes it, it works.
- djay Pro is a Catalyst (iOS-on-Mac) app: its library UI rejects synthetic
  accessibility/keystroke input, so end-to-end GUI automation of "load a track
  and screenshot the cues" is not reliable. Verify writes at the DB level
  (`verify` command / the round-trip test) instead.
- djay syncs to iCloud via its own code path; externally-written cues set
  `userChangedCloudKeys` but bypass that machinery, so they may not propagate to
  other devices. Treat single-machine as the supported case.
- The TSAF container-count header counts objects, arrays/sets, NSData and NSDate
  (f64) but **not** URLs/strings/scalars — see the module docstring in `tsaf.py`.
- djay numbers hot cue pads from 1 (A) to 8 (H). Number 0 is the track's start point,
  not a pad, and djay drops a hot cue that carries it when it loads the track.
  Tag `0x2d` is the integer 1 and tag `0x2e` the integer 0, not the other way around.
  The tool had this wrong until 2026-11-14: pad B was written as 0 and never showed up,
  and pads C to H landed one pad too low. `verify` read the numbers back with the same
  mistake, so it showed nothing wrong. After a change to what is written, have someone
  look at one track in djay.
- Tag `0x02` points back to an object written earlier in the same record.
  `tsaf.Ref` handles it, and the writer works out the position again.
  Never keep that position yourself: once an edit adds cues or a grid object
  before the target, it points at the wrong object and djay crashes.
  This is what broke fingerprint records twice
  (`docs/audio-alignment-fingerprint.md`).
