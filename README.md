# leap-asl

Turning a Leap Motion Controller into an ASL sensor, aimed at a two-way
communication device for the Deaf.

The device is not used the way Ultraleap intended. Its tracking daemon is
bypassed entirely; what we want is the raw stereo infrared, which turns out to
see far more than the daemon will report.

## The finding that shapes everything else

Head-to-head on 36 identical IR frames, hands at chest height ~40-50cm, which is
where people actually sign:

| Tracker | Hand found |
|---|---|
| MediaPipe on the raw IR | **30.6%** @0.5 conf, 75% @0.1 |
| `leapd`, the stock daemon | **2.8%** |

They agreed on **zero** frames. Ultraleap tuned `leapd` for hands in a cone
*above* the device — a desk puck for VR, not a camera pointed at a person. The
optics are fine. The daemon's operating envelope is the problem.

`leapd` also emits low-confidence (0.21-0.29) phantom locks, frozen to within
1-2mm across 20 frames. Treat any `leapd` hand under ~0.5 confidence as noise.

**So: don't use `leapd`'s skeleton. Run MediaPipe on the raw stereo IR and
triangulate.**

## Preprocessing is narrow and mandatory

MediaPipe will not see a hand in a raw Leap frame. Three steps, and the third is
the one that unlocks it:

1. Rotate 180° — raw frames are upside down.
2. CLAHE local contrast.
3. **Stretch 640x240 -> 640x480**, undoing the 2:1 vertical squash.

Six other variants failed outright. With the aspect correction, detection scores
land at 0.89-0.999.

## Metric 3D

The Leap ships a 64x64 distortion grid per eye (`LEAP_IMAGE.distortion_matrix`)
mapping ray slope, over [-4,4], to normalized image coordinate. Invert it
numerically, then triangulate:

```
z = baseline / (slope_x_L - slope_x_R)
```

**The baseline is 60mm on this unit, not the 40mm commonly quoted.** Two
independent confirmations: triangulated depth 352mm against `leapd`'s own palm
distance 349mm (0.9%), and a reconstructed knuckle row of 87mm, correct adult
hand breadth — at 40mm it would come out child-sized at 58mm.

### Two gates, both necessary

**Epipolar.** MediaPipe matches independently per eye and will lock onto
different objects. Rectified cameras must agree in y, so reject
`|slope_y_L - slope_y_R| > 0.15`. Without it, hand spans blow out to 420mm.

**Rigid bone.** The epipolar gate is necessary and *not sufficient* — frames pass
it and still reconstruct as an impossible hand. The knuckle row (index MCP to
pinky MCP) is metacarpal bone, so it reads the same whether the hand is a fist or
flat. Reject anything outside 72-105mm.

On a 20-frame set, 8 survive both gates (**40%**), clustering at 87mm ±8%. That
±8% is the triangulation noise, measured rather than assumed.

Note the honest limit: using anatomy to *filter* frames means the anatomy cannot
then be cited as proof the anatomy came out right. What survives is a
*self-consistency* claim, which epipolar gating alone does not deliver.

## Fingerspelling classifier

Trained on [ASLA-Leap](https://github.com/WenjinTao/ASLA-Leap) — 54,000 stereo
32x32 IR hand crops, 5 subjects, **24 classes: the alphabet minus J and Z**,
which require motion and are structurally impossible for a static-frame model.

| Evaluation | Accuracy |
|---|---|
| Random 80/20 split | 99.82% |
| **Leave-one-subject-out** | **81.11%** |

**Quote the LOSO number.** The random split is inflated by near-duplicate
consecutive frames of the same signer appearing on both sides. LOSO is what
happens when a stranger walks up to the device. Per subject: 89.3 / 80.0 / 67.7 /
94.9 / 73.6 — a 27-point swing depending on whose hands they are.

The errors are linguistically real, not noise:

- **T→N, M↔N, S→A/M/N** — the fist family, identical but for where the thumb tucks
- **U→R** — two fingers together vs. crossed
- **K→V** — both extend index and middle; K puts the thumb between

Every one is a thumb-position and occlusion problem, which is exactly what the
stereo depth is positioned to fix and what 32x32 crops cannot carry.

Not yet scored against ground truth on a live signer.

## `.pose` output for the open ASL toolchain

The open pipeline ([spoken-to-signed-translation](https://github.com/sign-language-processing/spoken-to-signed-translation),
[sign.mt](https://github.com/sign/translate)) is lexicon-driven: text → gloss →
pose sequence → avatar. It speaks `.pose` files in the MediaPipe **Holistic**
schema, so `asl/leap_pose.py` emits exactly that, and Leap recordings become
lexicon entries.

**Our `z` is real millimetres. Holistic's `z` is pseudo-depth** — image-scaled,
wrist-relative, no metric meaning, no stable scale between frames. Same
container, better contents.

Two traps found:

- `pose_format.utils.holistic` imports `mp.solutions` and pins
  `mediapipe<0.10.30`. It **cannot be imported** against MediaPipe 1.0. Build the
  header by hand.
- `pose_normalization_info()` anchors on shoulder landmarks 11/12 in
  `POSE_LANDMARKS`. With no body camera those are zeros — and `normalize()` does
  **not raise** on a zero-length anchor, it silently returns finite nonsense. Use
  `normalize_knuckles()`, which anchors on the rigid metacarpal row instead.

## From classifier to reader

A per-frame classifier is not a reader. Three things have to be added, and each
is a different kind of problem:

**Silence.** A softmax always sums to 1, so a bare classifier is confidently
wrong about an empty room. `stream.py` makes "I don't know" expressible with an
entropy ceiling and a top-two margin floor, and clears its buffer whenever the
hand leaves.

**Patience.** At 115fps a held letter spans ~30 frames. Voting across a window
beats trusting any single frame, and it quietly recovers frames the geometry
gates drop, because a neighbour covers the gap.

**Motion.** J and Z are handshapes *plus a path*. The CNN is structurally blind
to them. They are handled by a rule over the landmark trajectory: Z is an index
finger reversing horizontally at least twice; J is the I handshape with the
pinky travelling far and hooking once. Rule-based on purpose — with no training
data for either letter, a legible rule that a signer can help tune beats a model
fit to nothing.

Verified against synthetic trajectories: J fires on a hook and not on a fist, Z
fires on a zigzag, a held hand stays silent. On *real* capture neither ever
fired until the trajectory window was separated from the vote window: the vote
window is deliberately shorter than a letter (0.2s), and a Z is three strokes
over about a second, so the two reversals it looks for could not be inside it.
With a 0.9s trajectory buffer, an 8-second burst of repeated J's now yields J's
and a burst of Z's yields Z's -- mixed with static letters read during the
pauses between repetitions.

## Segmentation: why a letter appears

The reader's hardest job is not naming a handshape, it is deciding *when* a
handshape was a letter somebody meant. Crossing from H to I the hand passes
through shapes that look like other letters, and an early version committed
them -- spelling HILLY produced a Q that was never signed.

What decides now, in order:

1. **Dwell.** The frame-level prediction must stay the same letter for
   `dwell_sec` before anything is committed. A shape on the way somewhere does
   not stand still for a sixth of a second.
2. **A movement lockout.** Sustained coherent movement pushes commitment out by
   `settle_sec`, so the quiet instant at the top of a transition arc is still
   inside the shadow of the movement that made it.
3. **Agreement, entropy, margin** across the vote window, as before.
4. **Retraction.** A letter committed on a near-minimum dwell, immediately
   followed by one held much longer, is taken back out of the word.

Two measurements changed the design. Movement is **speed plus straightness**,
never speed alone: a dead-still D dances at 10.9 spans/sec of centroid travel --
faster than a real J stroke -- but covers 5% of its own path length, where J
covers 71% and Z 97%. And stillness is **not** required for dwell, because some
handshapes are tracked too badly to ever be still: that same D reads 25-70
spans/sec of raw landmark jitter while the classifier calls it D on 400 frames
out of 400. The tracker is noisy; the shape is not.

Every threshold is in seconds and hand-spans per second, never frames or pixels,
so the reader behaves the same at 55fps and 115fps and at any distance from the
sensor.

### Patience presets

| preset | dwell | keeps up with |
|---|---|---|
| fast | 100ms | ~6 letters/sec |
| normal | 160ms | ~5 letters/sec |
| careful | 240ms | ~3.5 letters/sec |
| deliberate | 360ms | slower than 3/sec |

Switchable live from the reader page, or `--preset` at startup. The trade is
one-for-one: patience is bought with speed.

## Layout

```
asl/leap_pose.py     stereo IR -> metric 3D -> .pose        (the bridge)
asl/stream.py        rolling-window reader: dwell, voting, rejection, J/Z motion
asl/words.py         word-level regression: synthetic pace sweep + clip replay
asl/reader.py        live server -- capture, read, serve the interface
asl/reader.html      picture-in-picture UI: self-view skeleton + readout
asl/capture.py       labeled capture, built for one hour with a fluent signer
asl/score.py         per-frame vs. after-voting vs. coverage
asl/finetune.py      adapt the ASLA-Leap model to this rig and these signers
asl/prep_data.py     ASLA-Leap -> cached arrays
asl/train.py         classifier, random + LOSO evaluation
src/Recorder.c       both eyes + native verdict + distortion grids
src/GrabImage.c      stereo PGM with brightness stats
stereo/, h2h/        recordings the results above were measured on
```

### Running a session

```bash
venv/bin/python asl/capture.py posed --subject interp1     # all 26, held
venv/bin/python asl/capture.py speed --subject interp1 --words CISCO MERCY
venv/bin/python asl/score.py --subject interp1             # the actual number
venv/bin/python asl/finetune.py --subjects interp1         # adapt to this rig
venv/bin/python asl/reader.py                              # live, port 8770
venv/bin/python asl/reader.py --preset careful             # hold each letter longer
venv/bin/python asl/words.py --synth --sweep               # pace vs. preset
venv/bin/python asl/words.py --clip asl/data/raw/live/HILLY-1786553040 --sweep
```

The reader page has a **Record** button. Type the word you are about to sign,
press it, spell, press stop -- the raw frames land in `asl/data/raw/live/<WORD>-<ts>/`
with their real timestamps. That clip is the only kind of data that contains
*transitions*, so it is the only thing that can measure letters the reader
invented while the hand was on its way somewhere. Posed captures cannot: there
is nothing between the letters in them.

The default model is now `model_cisco26b.pt` (99.5% per frame on Cisco's posed
set). It used to be `model_loso0.pt`, the ASLA-Leap transfer model, which scores
**37.8%** on the same frames -- so anyone following this README was reading with
the wrong model, which on its own produces letters nobody signed.

Capture is resumable and never asks a question mid-session — Ctrl-C leaves
everything recorded so far on disk.

## Setup

```bash
uv venv --python 3.12 venv
uv pip install --python venv/bin/python mediapipe opencv-python-headless scipy pose-format torch
mkdir -p models && curl -sSL -o models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

venv/bin/python asl/leap_pose.py stereo out.pose
```

`src/` also needs Ultraleap's SDK samples (`ExampleConnection.c` and friends),
which are theirs and not redistributed here. Install the Hyperion or Gemini SDK.

### Ubuntu 26.04 install traps

Ultraleap's apt repo is dead; pull the `.deb` from their S3 directly. Then the
postinst **silently does nothing**: it calls `$(which udevadm)`, `$(which
systemctl)`, `$(which leapctl)`, and 26.04 has no `which`, so it prints "No
udevadm detected" and exits 0. Afterwards, by hand:

```bash
sudo udevadm control --reload
sudo systemctl enable --now ultraleap-hand-tracking-service
sudo leapctl eula -y     # without this every client gets eLeapRS_Timeout
```

Their `99-LMC.rules` also ships `GROUP="plugdev"` on its own line after a blank
line, making it a standalone rule that matches **every device on the system** —
including your NVMe. Override it in `/etc/udev/rules.d/`.

## Status

Working: capture, MediaPipe on IR, metric triangulation, fingerspelling
classifier, `.pose` export.

Open: accuracy on a live signer, false-positive rate on empty frames, temporal
smoothing across the 115fps stream, and non-manual markers — facial grammar —
which live outside this sensor entirely and need a second camera.
