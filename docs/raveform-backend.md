# ml-raveform

`--engine ml-raveform` finds sections with the all-in-one model trained on
Raveform, a set of 1,423 dance tracks that DJs labeled with intro, buildup, drop,
breakdown, cooldown and outro (Kim, Kim, Kim and Nam, TISMIR, April 2026).

On Raveform's own test set, it finds 84% of section changes, against 64% for the
pop-trained model that `ml-allin1` uses.

Beats still come from beat_this, and cues still snap to its downbeats.

```bash
uv sync --extra allin1
uv run autohotcue propose "/path/to/track.m4a" --engine ml-raveform
```

## Weights

The engine uses `all-fold0-40pa2vpn.pth` from the authors' Hugging Face repo
`taejunkim/allinone`.

It's the model trained on both pop and dance music, which the paper found as good
as the dance-only one on dance tracks.

The download is pinned to one revision and checked against its SHA-256.

The repo doesn't state a license for these files.
Only use them if you have one.

On first use, the file is downloaded and converted into
`~/.cache/autohotcue/mlx-weights/raveform-all-fold0/`.
Only the dance-music labels are kept.

The conversion was checked on the pop model: converting its PyTorch file gives
exactly the same bytes as the official MLX weights.

## Label order

The model has 11 labels, but the authors haven't published their order.
This is the order used here:

```
0 start  1 end  2 altintro  3 altoutro  4 intro  5 outro
6 breakdown  7 buildup  8 cooldown  9 bridge  10 drop
```

It was worked out three ways.

First, the model's bias for each label roughly follows how much of a track that
label covers.
This was checked on the pop model, whose order is known.
On the dance model, labels 6 and 10 have the highest biases (drop and breakdown
cover 40% and 21% of the labeled time), and 0 and 9 the lowest (start and
bridge, about 0.2% each).

Second, each label behaves differently on real tracks.
Label 10 covers most of a track and is loud.
Label 6 sits in the middle and is quiet.
Label 4 usually opens the track, and is followed by 7 and then 10.
Label 8 comes after a drop and before 5.
Label 1 is silence at the end, and 0 a sliver of silence at the start.
Labels 2 and 3 are quiet and come very early or very late.

Third, the order matches the pop model's layout, which also starts with start,
end, intro and outro.

On the more EDM-like tracks of the library, the result reads like a textbook
track, for example:

```
intro, buildup, breakdown, drop, breakdown, drop, breakdown, cooldown, outro, end
```

If the authors publish the order, compare it with `RAVEFORM_LABELS` first.

## Cues

`ravepolicy.py` turns the labels into pads.

- A and B are the first downbeat.
- D is the first drop after the first beat.
  A track that grooves from bar one is labeled drop from the start, so its drop is
  the next one.
- E is the first breakdown after D, and F the first drop after E.
- C is the buildup right before D.
  By default the pre-drop rule then moves it to 8 bars before the drop.
- G is the first cooldown or outro after the last drop, and H the outro.

The ambient intro and outro count as normal ones.

## Speed

About 6 seconds per track on an M5 Max, the same as `ml-allin1`.
Most of it is demucs separating the stems.
