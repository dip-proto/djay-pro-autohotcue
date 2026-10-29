# 07 - SongFormer memory and speed fix (done)

Done on 2026-10-29, on an M5 Max with 128 GB.
The details and numbers are in [songformer-backend.md](../songformer-backend.md).

The old plan was to switch on the `flash_conformer` copies that ship with MuQ and MusicFM.
That turned out to be unnecessary.
Replacing the attention inside transformers' own conformer was enough, with no import tricks.

Attention wasn't the only problem.
The conv layers in front of each encoder were more than half of the time on the GPU, and
on the CPU they alone needed 23 GB for one 420 s window.
They now run on overlapping pieces, and the 30 s chunks reuse the window's result.

How it did against the original goals:

- Peak RAM under 10 GB on CPU: 11 to 12.5 GB, so a bit above.
  The model weights alone take 2.8 GB.

- Usable on the GPU: yes, with 7.2 GB, and it is now the default.

- Same segments as before: yes, on 20 library tracks and on cuts around the window and
  chunk edges.

- Tests green and docs updated: yes.

- Old code path behind a switch: no.
  Instead, an opt-in e2e test runs the model's own forward pass and expects the same
  segments.
