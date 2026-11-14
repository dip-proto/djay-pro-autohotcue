# ml-stems

`ml-stems` is the default engine.
It places cues for mixing in on a drop and out on a breakdown.

```
+-----+-----------------+---------------------------------------------------+
| Pad | Name            | Where                                             |
+-----+-----------------+---------------------------------------------------+
| A   | 8 Bars to Drop  | 8 bars (32 beats) before B                        |
| B   | Drop            | where the beat and the bassline come in           |
| C   | 8 Bars to Break | 8 bars before D                                   |
| D   | Breakdown       | where the track thins out, with no vocals         |
| E   | Loop            | 8 bars that the track repeats right after         |
| F   | Vocals          | vocals over little or no beat                     |
| G   | Bridge          | the beat without its bassline, 8 bars or more     |
| H   | Drop 2          | the second drop                                   |
+-----+-----------------+---------------------------------------------------+
```

A to B and C to D are the same length on purpose.
Start the next track on its pad A when the playing one reaches pad C,
and its drop lands on the breakdown of the track going out.

Each cue is written with its name, so djay shows what a pad holds.

A Loop cue is written with its end, 8 bars after its start, so djay plays it as a
32-beat loop when the pad is pressed.

B is the cue that matters most, so every track gets one.
When a track has no drop, B is where its fullest passage starts, and it is named Peak
(with A named 8 Bars to Peak).

## Where the information comes from

Three things are computed for every track.

beat_this tracks the beats, and `gridlock` fits djay's straight grid to them.
Bars are counted on that grid.

`gridlock` refuses a grid as soon as a few beats are off, because it would overwrite djay's.
That happens on about one track out of four here, and most of them keep a perfectly steady tempo:
the tracker doubled the tempo for a few bars, or lost the beat in a breakdown.
Counting bars on what the tracker reports makes some bars longer than others:
"8 bars before the drop" came out as 15 beats on one track and 62 on another.
So the tracked beats are cleaned first (doubled ones dropped, missing ones filled in),
and when 85% of them sit on one straight lattice, bars are counted on that lattice,
starting on the beat where most tracked downbeats fall.
Only tracks whose tempo really moves (a live drummer, a spliced edit) get bars of four
tracked beats, and everything else works the same.

A bar is four of djay's beats, whatever the tracker counts.
On some tracks djay counts half as many beats as the tracker (80 BPM against 160):
a bar is then eight tracked beats, so that 8 bars are 32 beats on djay's screen.

A file often opens right on a downbeat, which the grid then puts a few milliseconds
before the audio.
That first bar counts, and a cue on it goes to the very start of the file.

demucs splits the track into drums, bass, other and vocals.
`_stems.py` measures how loud each stem is in every bar.
From these come two numbers per bar:
energy, the mean of the drums and bass levels, and
fullness, the mean of all the stems that the track actually uses.
Each stem is measured against its own loud bars, so a bassline coming in moves the
energy as much as the drums coming in, even when the drums are much louder.

The Raveform model labels the sections: intro, buildup, drop, breakdown, cooldown, outro.
It reads the same stems, so the track is only separated once.

The labels say where to look, and the stems say what happens there.
Neither is enough alone.
The stems can't tell a hook coming in from a louder hi-hat, and the model was trained
on big-room dance music, so it mislabels some house tracks in ways the stems make obvious.

## The rules

They are in `stempolicy.py`, which only sees numbers and has no audio or model code.

### B, the drop

It is the first section labeled drop that follows another section.
The model is precise there: when the stems show energy arriving near a labeled drop,
it is on the labeled bar 97 times out of 100.

Two corrections apply.

The model often calls the first 8 or 16 bars of full groove a buildup, and starts the
drop where the hook joins.
Nothing arrives at such a drop: the beat and the bassline have been playing for a while.
When the 8 bars before the labeled drop already have its energy, the drop moves back,
8 bars at a time, to the bar where that energy came in.
It stays where it is when the groove has been there since the first bars,
because a groove that never arrives has no better bar to offer.

A track with no intro is labeled drop from its first bar.
Its drop is then the first bar where the stems show the groove coming in.
When they show nothing, the first labeled drop after a breakdown is used.

When no section gives a drop, the groove coming in is looked for over the whole track.
Some tracks have a clear one that the model calls a breakdown.

What is left has no drop: songs that play their beat from the first bar to the last,
and loops made for DJs.
B then goes to the first bar of the fullest passage, which in a song is often a chorus.
The level of 8 bars is the mean of all the stems, each against its own loud bars.
Among the bars where that level is within 5% of the highest one,
B is the one where it goes up the most from 8 bars before:
the fullest 8 bars are often in the middle of a longer passage, and this finds its start.
The cue is named Peak, and a note says that no drop was found.

A is 8 bars before B.
It is left out when B comes less than 8 bars into the track.

### D, the breakdown

Candidates are the starts of the sections labeled breakdown, cooldown, intro or outro,
at least 16 bars after B.
Bars where the stems thin out with no section starting there are candidates too:
the model keeps calling a drop a drop when only its bassline leaves.

A candidate qualifies when the 8 bars from it have no vocals and less in them than the
8 bars before: less energy, or less fullness.
At least half of the track's vocals must also be behind it.
A short break before the singing has even started is not where you leave a track.
The best one is the one that loses the most, with short ones scaled down
(a 16-bar breakdown counts in full, an 8-bar one for half).
A breakdown lasts until the next drop, or until half of what it lost is back.
Anything under 4 bars is a fill, not a breakdown.

Sections after the last drop only count when nothing else qualifies.
The note "using the outro" says so.
When vocals run through every breakdown and the outro too, the same choice is made
with the vocals allowed, and a note says that as well.

Tracks cut the beat a bar or two before a section ends, as a fill,
and let it ring on for a bar after.
The model puts the breakdown on the phrase start most of the time, the stems put the
loss of energy one or two bars before it about half of the time.
So a candidate found within two bars of the start of an 8-bar phrase moves to that start.
Phrases are counted from the start of the drop the bar is in.
The bridge cue moves the same way.
The vocal cue only moves back to the start of its phrase, since a short passage
may be over by the next one.

C is 8 bars before D.

### E to H

Each of these pads has a kind of its own:

- H is the second drop: the next section labeled drop after B.
- F is the passage with the most vocals among those where the beat is low
  for at least two bars.
- G is a labeled bridge if the model finds one (on house tracks it never does),
  otherwise the longest stretch of at least 8 bars, between the first and the last drop,
  where the drums play and the bass doesn't.
- E is a loop.

A loop is an 8-bar phrase that the track itself repeats in the 8 bars after it.
Looping it then sounds like the track going on: what follows the last bar is what the
track would have played anyway.
Three things are checked.
The bars must sound like the ones 8 bars later (cosine similarity of their mel spectra,
cut in 16th notes), and the first bar most of all, since it is where the loop lands.
Every stem must be as loud in the phrase as in its repeat, because the drums can hide a
bassline coming in from the first test.
The beat and the bassline must both be playing.
The fullest such phrase wins.

A cue is never placed within 4 bars of another.
Pads whose kind was not found take what is left: the third and fourth drops,
a second loop that sounds different from the first, a second vocal passage.
A track gets at most two loops, two vocal cues and two bridges.

## What to expect

On a library of 1187 tracks, mostly funky and disco house with some techno, 80s pop songs
and loops for DJs, B is a drop on 95% of the tracks and a Peak on 5%.
Three files have no B: two where the tracker finds no beat (white noise, a scratch sample)
and one that djay lists twice, which `apply` refuses to touch.

A is there on 96% of the tracks, C and D on 88%.
The tracks without D are mostly songs and loops that never thin out.
On the 451 tracks measured first, D was the outro more than four times out of ten:
half of the breakdowns in this kind of music have someone singing over them.

A second drop is found on three tracks out of four, a loop on almost half of them
(677 loops in all, each 32 of djay's beats long),
a vocals-only passage on four in ten and a bridge on one in twenty.
With the spare pads filled, E holds something on 73% of the tracks, F on 61%, G on 31%
and H on 75%.

## Limits

There is no ground truth for any of this.
The rules were tuned by looking at the stems of a few hundred tracks, not by scoring
cues against hand-placed ones.
`bench` can do that once someone labels a set of tracks.

The vocal stem also catches vocal chops and some lead synths.
A breakdown with a chopped vocal sample in it counts as having vocals.

When the tempo is not steady, bars are counted on tracked beats, and the tracker is not always right.
On 7 tracks out of the 1022 that have all four pads, A to B and C to D are not the same length,
and on 9 tracks an 8-bar span is a beat or more away from 32 of djay's beats.

Tracks under 24 bars only get A and B, and tracks under 8 bars get no cues.
A file where the tracker finds no beat has no bars to count, so it gets none either.

D is looked for after B, also when B is a Peak.
A track whose fullest passage comes late can end up with no C and no D.

`--pre-drop-bars` belongs to the other engines and is ignored here:
A and C are always 8 bars ahead.

## Changing the rules

The models take 5 seconds per track, the rules a few milliseconds.
So the measurements are saved once and the rules are rerun on them:

```bash
uv run python scripts/stems_dump.py ~/Music/djay/tracks tmp/features
uv run python scripts/stems_review.py tmp/features
uv run python scripts/stems_review.py tmp/features out.png "Fico|Babert"
```

The first command saves the stem levels, the mel spectrogram, the beats, the grid and the
sections of every track (1.5 MB each).
The second prints how many tracks get each cue and each note.
The third plots the stems, the sections and the cues of the tracks whose names match.

Every threshold in `stempolicy.py` has a test in `tests/test_stempolicy.py` that fails
when the threshold is taken away.
Keep it that way.

## Speed

About 5 seconds per track on an M5 Max, like `ml-raveform`.
Most of it is demucs.
The engine runs one track at a time whatever `-j` says, since the GPU is the bottleneck.
