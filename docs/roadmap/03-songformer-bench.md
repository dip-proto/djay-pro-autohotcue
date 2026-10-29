# 03 — SongFormer engine: bench and keep-or-retire decision

**Goal:** judge the `ml-songformer` structure engine with numbers. The engine
is **already merged to `main`** (June 2026) and fully wired — what's missing is
the quantitative comparison against `ml` and `ml-allin1` on real ground truth.

## Current state (post-merge)

- `--engine ml-songformer` works end-to-end on `main`: SongFormer structure
  (HF `ASLP-lab/SongFormer`, pinned revision), beat_this beats, grid-lock
  snapping, cue policy unchanged. See `docs/songformer-backend.md` for the
  full backend story (install, env knobs, memory).
- Wiring that the original draft of this brief asked to verify is all done:
  `bench.VALID_ENGINES` includes it, `analysis.effective_parallel_jobs`
  clamps it to jobs=1 (allin1-style), unit + opt-in e2e tests exist.
- Qualitative-only evidence so far: on Afro House test tracks the cue layouts
  are musically sensible (Drop on first chorus, Breakdown on the bridge) and
  the paper numbers favor it (HarmonixSet HR.5F 0.703 vs All-In-One 0.596,
  arXiv 2510.02797). No bench table exists — that was and remains the blocker.
- Known costs to weigh in the decision: runtime and memory are in
  `docs/songformer-backend.md`; since item 07 (October 2026) they are small.

## Plan

1. Wait for item 01 (truth.json). Do not bench on machine-generated cues —
   that measures agreement, not quality.
2. Run: `uv run autohotcue bench bench-data/truth-zzzzztest.json
   --engines ml,ml-allin1,ml-songformer` (songformer self-clamps to jobs=1;
   expect about 5 s per track on the GPU).
3. Compare per-slot hit-rates, MAE, and runtime/track.
4. **Decision rule:** `ml-songformer` stays the recommended structure engine
   only if it beats the best current engine on D/E slots by a margin that
   survives the small sample. If item 02's bass-detector engine exists by
   then, songformer must beat *that*. If it loses, keep the engine (it is
   merged, optional, and zero-cost when unused) but record the table and
   demote it in README/docs from "better boundaries" to "alternative".

## Acceptance criteria

- A committed bench table covering `ml`, `ml-allin1`, `ml-songformer` on the
  same truth.json, recorded in `docs/songformer-backend.md`.
- An explicit keep-as-recommended / demote decision recorded here with one
  paragraph of rationale, including the runtime/memory cost in the weighing.

## Notes

- Depends on item 01.
- The runtime column matters: the user batch-processes 100-track playlists.
  At about 5 s per track, a full playlist now takes under 10 minutes.
