# ADCAudioAlignmentFingerprint — what we know, and the pairing problem

Working notes for reverse-engineering djay's audio-alignment fingerprint object and the
TSAF name↔value pairing semantics around it. Written after the June 2026 corruption
incident (below). The pairing problem is now solved; see the next section.

## Resolution (October 2026)

The trailing `timestampIdentifier` field is not an integer.
Tag `0x02` is a **container back-reference**: a u8 index pointing at an object
written earlier in the same blob.
Containers are numbered in completion order (children before their parent), counting
the same things as the header count (objects, arrays, sets, NSData, NSDate).
In user data it always points at the fingerprint object; in `mediaItems` djay also uses
it for a `modifiedDate` equal to `addedDate`.

Evidence, from a library of 1,185 user-data records:

- All 1,159 fingerprint records hold a `0x02` whose index resolves to exactly the
  `ADCAudioAlignmentFingerprint` container, at indexes from 4 to 13 depending on how
  many cues and grid objects come before it.
- The `mediaItems` case only makes sense in completion order: pre-order would point
  `modifiedDate` at the title object, completion order points it at the NSDate.
  No other `mediaItems` record stores two identical inline dates, because djay shares
  the object whenever they are equal.

The old codec parsed the field as a plain `Int` and wrote the stale index back.
That is enough to explain both incidents:

- **June 2026 (Move).** Going from 6 to 8 cues moved the fingerprint from container 13
  to 15, and container 13 became the `ADCBeatGridEdits` object.
  djay called `fingerprintFloatsArray` on it.
- **October 2026 (Take Control, Your Love).** The guard was lifted on round-trip
  evidence alone. The first `apply` appended `cuePoints` after the fingerprint, so the
  index stayed right by luck, and djay re-saved both records in its own field order
  with `cuePoints` ahead of the fingerprint.
  The second `apply --force` replaced 5 cues with a different count, and the stale
  index then landed on the fingerprint's own NSData: djay crashed with
  `-[_NSInlineData ...]: unrecognized selector` in `yapDatabaseModified:`.
  Replaying that edit with the old codec reproduces it (5 to 6 cues lands on Data,
  5 to 7 lands on `ADCBeatGridEdits`).

The fix is in `tsaf.py`: `0x02` parses to a `tsaf.Ref` holding the target node, and the
serializer recomputes the index from where the target actually lands.
A reference to an object missing from the tree, or an index past 255, refuses to
serialize.
Replaying apply's edit (1 to 8 cues plus a grid object) on every fingerprint record of
the library keeps every reference on the same fingerprint, byte-identical.
`ADCAudioAlignmentFingerprint` is in `EDIT_SAFE_OBJ_CLASSNAMES` on that basis.

Still open: a real djay load of an edited fingerprint record (roadmap 04's acceptance
bar), and what the fingerprint blob itself encodes.
The ranked hypotheses below are kept for history; the first two were both partly right.

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
- Byte-exact round-trip (`serialize(parse(b)) == b`) is **necessary but not
  sufficient** for edit safety: it proves reproduction, not that our model of the
  record matches djay's. The stale back-reference round-tripped perfectly while
  corrupting records twice. Do not widen the guard based on round-trip evidence alone.
- `tests/test_tsaf.py::test_backrefs_follow_their_target_through_cue_edits` replays
  apply-style edits on every fingerprint record of the local library. It must pass.
- Any experiment that writes near a fingerprint record happens on a **copy** of the
  library, and corrupted artifacts get preserved before restoring.
