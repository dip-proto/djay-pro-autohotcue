# ADCAudioAlignmentFingerprint — what we know, and the pairing problem

Working notes for reverse-engineering djay's audio-alignment fingerprint object and the
TSAF name↔value pairing semantics around it. Written after the June 2026 corruption
incident (below). The pairing problem is now solved; see the next section.

## Resolution (October 2026)

The last `timestampIdentifier` field isn't a number.
Tag `0x02` points back to an object written earlier in the same record, by its
position.

Positions count the same things as the header (objects, arrays, sets, data and
dates), in the order they finish, children before their parent.
In user data, it always points at the fingerprint.
In `mediaItems`, djay also uses it when `modifiedDate` equals `addedDate`.

This holds for every fingerprint record of the library: all 1,159 point exactly at
their fingerprint, at positions from 4 to 13, depending on how many cues and grid
objects come before it.

The old codec read this field as a plain number and wrote it back unchanged.
Adding cues before the fingerprint then left it pointing at something else.
That explains both incidents.

- **June 2026, Move.** Going from 6 to 8 cues moved the fingerprint from 13 to 15,
  and 13 became the grid object.
  djay asked it for its fingerprint data and crashed.
- **October 2026, Take Control and Your Love.** The guard was lifted because the
  records round-tripped.
  The first `apply` added the cues after the fingerprint, so nothing moved.
  djay then saved both records again, with the cues before the fingerprint.
  The second `apply --force` changed the number of cues, and the old position now
  pointed at the fingerprint's own data.
  djay crashed with `-[_NSInlineData ...]: unrecognized selector`.
  Replaying that edit with the old codec gives exactly this.

`tsaf.py` now reads the field as a `tsaf.Ref` to the object itself, and works out
the position again when writing.
Replaying apply's edits on every fingerprint record of the library keeps every
reference on the right fingerprint.
The fingerprint is allowed in `EDIT_SAFE_OBJ_CLASSNAMES` because of that.

Two things are still open: loading an edited fingerprint record in djay itself,
and what the fingerprint data actually contains.
The hypotheses below are kept for history.

## What it is (observed)

- An Algoriddim object (`ADC*` class family) stored inside `mediaItemUserData` —
  the per-track record that holds user-owned, iCloud-synced state (cues, grid edits,
  rating, play count).
- Present on **72 of 3800** user-data records in the reference library (~2%) —
  plausibly the most-played / most-synced tracks.
- djay's accessor for it is `fingerprintFloatsArray` (from the crash selector), i.e.
  the payload decodes to a float array.
- Purpose (inferred from the name and its placement in synced user data): a compact
  acoustic signature used to **align time-based user data between different copies of
  the same song** (streaming match vs local file, different encodes/intros across
  synced devices). Cue/grid timestamps are file-relative; the fingerprint lets djay
  compute a per-version offset.

## Record anatomy (as parsed by tsaf today)

All 4 records inspected (3 random + Move) share the identical shape; field order as
our parser reports it:

```
('uuid')                 -> str
('userChangedCloudKeys') -> Arr        # contains the string "audioAlignmentFingerprint"
('titleIDs')             -> Arr
[('cuePoints')           -> Arr]       # when present
[('beatGridEdits')       -> Obj ADCBeatGridEdits]   # when present
('timestampIdentifier')  -> Obj ADCAudioAlignmentFingerprint
    ('timestampIdentifierData') -> Data, ~4.7 KB
    ('version')                 -> Marker 0x2e (= int 1)
('playCount')            -> Int | Marker
[('manualGain')          -> F32]
('timestampIdentifier')  -> Int        # duplicate root field name!
```

Raw-byte facts (consistent across all inspected records):

- `audioAlignmentFingerprint` appears **exactly once** in the bytes — inside the
  cloudKeys array. There is **no field named `audioAlignmentFingerprint`**.
- `timestampIdentifier` appears twice: once as the (parsed) name of the fingerprint
  Obj, once as the name of a trailing Int. `timestampIdentifierData` once (inside the
  Obj).
- The Data blob is **zlib-compressed** (`78 9c` header), length-prefixed.
  Move's: 4725 B compressed → **5040 B** decompressed = 1260×f32 or 630×f64.
  Plain little-endian f32 gives implausible values (±1e38) → **encoding open**;
  try big-endian f32, little-endian f64, or a (time, value) pair layout.
  djay calls it `fingerprintFloatsArray`, so floats are in there somehow.

## The June 2026 corruption incident

1. Baseline: Move's record (djay-canonical, 5613 B) contained cues + `beatGridEdits` +
   fingerprint and loaded fine in djay.
2. `apply` performed two in-place value replacements via `Obj.set`: new `cuePoints`
   array (6→8 cues) and new `beatGridEdits` value. Byte-exact round-trip held;
   the written record reconstructs to 5565 B with the fingerprint intact.
3. User opened djay, loaded Move → **"Could not load track"**,
   `-[ADCBeatGridEdits fingerprintFloatsArray]: unrecognized selector`.
4. djay re-saved the record as **761 B**: fingerprint object gone, the
   `ADCBeatGridEdits` object now sitting under the field name `timestampIdentifier`,
   `beatGridEdits` name token gone. Every subsequent load crashed (djay reads
   `timestampIdentifier`, expects the fingerprint class, calls
   `fingerprintFloatsArray` on our grid object).
5. Recovery: raw SQL `UPDATE` of the record bytes from the timestamped backup
   (no TSAF parsing involved). Track loads fine again.
6. **Forensics lesson: the 761 B scrambled record was overwritten during recovery
   without preserving a copy. Next time, dump the corrupted bytes to a file first.**
   What we recorded of it: fields `uuid, userChangedCloudKeys, titleIDs, cuePoints,
   timestampIdentifier(=ADCBeatGridEdits), playCount, manualGain,
   timestampIdentifier(Int)` — today's cues and anchor were preserved; only the
   fingerprint and the `beatGridEdits` name were lost.

The other 12 test tracks (no fingerprint) went through the identical write path and
load fine — djay even adopted our anchors and refined `fractionalBeatShift` itself.

## The pairing problem (hypotheses, ranked)

The crash + scramble prove djay binds these fields differently than our parser models,
at least when rewriting. Candidate explanations:

1. **Duplicate-name disambiguation.** djay genuinely keys the fingerprint Obj under
   `timestampIdentifier` (the cloud key `audioAlignmentFingerprint` being a different
   namespace), and disambiguates the two same-named root fields by order or by type.
   Our edits changed *relative positions/sizes* of content around them
   (cue count 6→8), possibly breaking djay's disambiguation on rewrite.
2. **Pairing frame-shift.** TSAF pairs `value` then `name`, names optional. If djay's
   true reading of this region differs from ours by one slot (e.g. the fingerprint Obj
   is anonymous and `timestampIdentifier` names the trailing Int), then any edit made
   through our model — even a pure value swap — can land in a slot djay reads
   differently. Note our model has no field named `audioAlignmentFingerprint` anywhere,
   yet djay's cloudKeys advertise one: *something* about our naming of this region is
   provably off.
3. **djay-side lazy-load bug.** djay may load the fingerprint lazily; our (valid)
   record changed sizes/offsets, the lazy load failed, and djay's writer serialized the
   record without the fingerprint, mis-writing the remaining fields. Under this theory
   our record was readable and the scramble is djay's writer bug — still triggered by
   our edit.

## Reverse-engineering avenues

1. **djay's writer as canonicalization oracle** (most promising, no disassembly).
   On a sacrificial library copy + the real djay app: mutate ONE thing at a time in a
   fingerprint record, open djay, load the track, quit, and diff what djay re-saves.
   Matrix: (a) byte-identical rewrite (control); (b) one cue time value changed;
   (c) cue count changed (the incident's delta); (d) beatGridEdits value changed;
   (e) c+d together (the incident). The first variant that djay scrambles isolates the
   trigger. djay's rewrite of healthy records also reveals its canonical field order
   and which names it re-emits — directly testing hypotheses 1 vs 2.
2. **Class-dump the app.** djay Pro is a Catalyst app under `/Applications`. `strings`
   / `otool -ov` / a class-dump tool on the main binary + frameworks for
   `ADCAudioAlignmentFingerprint`, `ADCMediaItemUserData`, `ADCBeatGridEdits`:
   property lists, ivar names, and the TSAF (de)serializer method names. The property
   names tell us the true field keys (is there a `timestampIdentifier` property on
   the user-data class AND on the fingerprint? what type does each expect?).
3. **Corpus diffing.** All 72 fingerprint records: assert the structural invariant
   holds library-wide (script in this repo's session history; 4/4 so far). Outliers =
   extra evidence about optional fields and ordering.
4. **Decode the blob.** Try BE f32, LE f64, strided layouts on the 5040 B payload;
   plot candidates — a real fingerprint should look like a smooth/structured curve,
   not noise. Knowing the content confirms what alignment djay does (and whether we
   could ever *regenerate* the object ourselves).
5. **tsaf grammar audit.** Targeted differential tests around the suspected zone:
   duplicate field names, Marker-valued fields followed by string tokens, and the
   anonymous-field rule (docstring: "if the token after a value is not a string token,
   the field is anonymous"). Construct minimal synthetic blobs for each ambiguous
   sequence and verify our pairing against djay's rewrite of the same bytes (via
   avenue 1).

## Safety rules in force

- `apply` refuses records whose parsed tree contains any Obj classname outside
  {user-data root, titleIDs, cue points, ADCBeatGridEdits, ADCAudioAlignmentFingerprint}
  — skip, never edit.
- A byte-exact round-trip (`serialize(parse(b)) == b`) is needed but isn't enough.
  It shows we can rewrite the record, not that we understand it: the wrong
  back-reference round-tripped perfectly and still broke records twice.
  Don't widen the guard on round-trip evidence alone.
- `tests/test_tsaf.py::test_backrefs_follow_their_target_through_cue_edits` replays
  apply's edits on every fingerprint record of the local library, and must pass.
- Any experiment that writes near a fingerprint record happens on a **copy** of the
  library, and corrupted artifacts get preserved before restoring.
