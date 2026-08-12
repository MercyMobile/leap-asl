"""Streaming fingerspelling: frames in, letters out.

Single-frame classification is not a reader. A reader needs four things the
classifier does not have:

  0. Segmentation. Naming a handshape is the easy half; deciding WHICH shapes
     were letters somebody meant is the hard one. Crossing from H to I the hand
     passes through shapes that look like other letters, and committing them put
     a Q in the middle of HILLY that was never signed. So a shape must hold
     still, unchanged, for `dwell_sec` before it counts, and sustained movement
     locks commitment out for `settle_sec` afterwards.


  1. Silence. It must say nothing when nobody is signing. Untested until now --
     a softmax always sums to 1, so a bare classifier is maximally confident
     about an empty room.
  2. Patience. At 115fps a letter spans ~30 frames. Voting across them beats
     trusting any one, and it recovers frames the geometry gates discard, since
     a neighbour covers the gap.
  3. Motion. J and Z are not handshapes, they are handshapes plus a path. The
     CNN is structurally blind to them; they need the landmark trajectory.

So this module wraps the CNN in a state machine over a rolling buffer.

    engine = Engine(model_path)
    ev = engine.push(gray_L, gray_R, t)     # -> Emission or None

Run `stream.py --replay <recdir>` to exercise it on recorded frames.
"""
import os
import sys
from collections import deque

import numpy as np
import cv2
import torch

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import Net, LETTERS
from leap_pose import (MODEL_TASK, IDX_MCP, PINKY_MCP,
                       KNUCKLE_MIN_MM, KNUCKLE_MAX_MM,
                       load_distortion, build_inverse, triangulate)

WRIST, MID_MCP = 0, 9
INDEX_TIP, PINKY_TIP = 8, 20

# ---- tuning
#
# Everything here is in SECONDS and in HAND-SPANS PER SECOND, never in frames or
# pixels. Frames were wrong because the capture rate is not constant -- a 12-frame
# window is 0.22s at 55fps and 0.10s at 115fps, so the reader's whole character
# changed with the load on the box. Pixels were wrong because the same movement
# measures twice as large with the hand held twice as close to the sensor.
#
# The speed numbers below come from measuring the posed captures rather than
# guessing: a held letter sits at 0.9-2.6 spans/sec of palm travel (L 0.93,
# Y 0.91, B 0.92, A 1.36, Q 2.57, C 3.74) and 1.0-2.1 of shape change.

class Tuning:
    """Live-adjustable segmentation parameters. The reader edits these at runtime."""

    def __init__(self, **kw):
        self.window_sec = 0.20     # rolling vote window
        self.min_votes = 5         # usable frames before anything may emit
        self.agree_min = 0.72      # fraction of frames that must INDIVIDUALLY agree
        self.entropy_max = 2.10    # nats. Uniform over 24 letters is ln(24) = 3.18
        self.margin_min = 0.12     # top prob minus runner-up, averaged

        # A letter is not committed the instant it wins a vote. It must persist,
        # unchanged and still, for dwell_sec. This is THE fix for letters appearing
        # that were never signed: crossing from H to I the hand passes through
        # shapes that briefly look like a letter, but it passes through them --
        # a shape on the way somewhere does not stand still for a sixth of a second.
        self.dwell_sec = 0.16
        # ...and after any real movement, nothing may commit until the hand has
        # been quiet this long. A transition always contains fast motion, so this
        # blanks the whole curve rather than trusting the vote to survive it.
        self.settle_sec = 0.09

        # Whether stillness is REQUIRED for the dwell clock to run, or whether a
        # stable prediction is enough on its own. OFF by default, and that is a
        # measured decision rather than a preference: some handshapes are tracked
        # so badly that they are never still by any threshold. A dead-still D
        # measures 8-22 spans/sec of centroid travel and 25-70 of raw landmark
        # jitter, frame after frame -- while the classifier calls it D on 400
        # frames out of 400. Stillness could not pass that letter at any setting,
        # and prediction stability passes it perfectly. The tracker is noisy; the
        # SHAPE is not, and the shape is what a letter is.
        self.still_gate = False
        self.still_travel = 6.0    # spans/sec the hand may drift and still count as held
        self.still_shape = 5.0     # spans/sec of fingertip change that counts as held
        self.still_grace = 3       # consecutive over-threshold frames before it counts
        # Movement is speed AND direction, never speed alone. A held D dances at
        # 10.9 spans/sec of centroid travel -- faster than an actual J stroke --
        # but it dances in place: over a fifth of a second it gets 5% of the way
        # along the path it travelled. A stroke keeps going: J 0.71, Z 0.97. So a
        # straightness ratio separates a moving hand from a badly tracked still
        # one, which a threshold on speed cannot do at any value.
        # (measured over 0.2s windows: J/Z/D/O/L/C, posed captures)
        self.move_min = 6.0        # spans/sec of coherent travel
        self.move_straight = 0.35  # net displacement / path length
        # The movement window TRAILS: a movement stays inside it for its whole
        # length after the hand has already stopped, and every millisecond of that
        # is added to the wait before the next letter can commit. Short enough to
        # let go quickly, long enough to average a couple of frames.
        self.move_win = 0.12
        self.move_frames = 3       # ...and the movement must persist. One frame of
                                   # coherent displacement is a tracker glitch or a
                                   # dropped frame, not a hand going somewhere.
        # J and Z are read from a TRAJECTORY, and it has to be long enough to
        # contain one. The vote window is a fifth of a second by design -- shorter
        # than a letter -- but a Z is three strokes and takes about a second, so
        # asking the vote window for two direction reversals was asking for
        # something that could not be there. It never once fired on real capture.
        self.motion_sec = 0.9
        self.cooldown = 0.45       # before the same letter may repeat

        # A track that has latched onto the wrong hand hypothesis jitters ~5x
        # harder than a real held hand (measured: 15.4 spans/sec against 0.9-3.2).
        # Above this for jitter_sec, the landmarker is rebuilt.
        self.jitter_max = 8.0
        self.jitter_sec = 0.35

        # Retraction. A letter committed on a bare-minimum dwell, immediately
        # followed by one that dwelled longer, was a transition artefact -- the
        # reader takes it back instead of leaving it in the word.
        self.retract = True
        self.retract_sec = 0.60    # how long a weak letter stays retractable
        self.solid_sec = 0.26      # dwell at or above this is never retracted
        self.__dict__.update(kw)

    def as_dict(self):
        return dict(self.__dict__)


PRESETS = {
    # dwell_sec, settle_sec, agree_min -- one knob the signer can actually turn
    "fast":     dict(dwell_sec=0.10, settle_sec=0.05, agree_min=0.66),
    "normal":   dict(dwell_sec=0.16, settle_sec=0.09, agree_min=0.72),
    "careful":  dict(dwell_sec=0.24, settle_sec=0.14, agree_min=0.78),
    "deliberate": dict(dwell_sec=0.36, settle_sec=0.20, agree_min=0.82),
}

MAX_BUF = 240          # hard cap on the rolling buffer, whatever the frame rate

# The stereo gate costs a second MediaPipe pass -- the dominant per-frame cost
# once the interpolator is cached. It exists to catch a reconstruction that is
# not a hand, which is a sustained condition, not a single-frame event. So check
# it periodically and carry the verdict between checks. 1 = every frame.
STEREO_EVERY = 3


class Emission:
    """A letter the reader is willing to commit to.

    `dwell` is how long the hand actually held the shape before it was committed.
    That number is what makes a retraction decidable later: a letter the hand
    barely touched is a candidate for being taken back, one it sat on is not.
    """

    def __init__(self, letter, conf, t, votes, kind="static", dwell=0.0,
                 retract_prev=False):
        self.letter, self.conf, self.t = letter, conf, t
        self.votes, self.kind = votes, kind
        self.dwell, self.retract_prev = dwell, retract_prev

    def __repr__(self):
        return (f"<{'-prev ' if self.retract_prev else ''}{self.letter} "
                f"{self.conf*100:.0f}% {self.kind} n={self.votes} "
                f"dwell={self.dwell*1000:.0f}ms>")


# ------------------------------------------------------------------ J and Z

def _reversals(path, axis, min_travel):
    """Count direction changes along one axis, ignoring jitter."""
    d = np.diff(path[:, axis])
    sig = d[np.abs(d) > min_travel]
    if len(sig) < 2:
        return 0
    return int(np.sum(np.sign(sig[1:]) != np.sign(sig[:-1])))


def classify_motion(paths, shapes):
    """Return 'J', 'Z', or None from a trajectory buffer.

    J is the 'I' handshape -- pinky alone -- swept down and hooked. So: the
    static classifier should have been calling it I throughout, and the pinky
    tip should travel far with a single direction change as the hook turns.

    Z is an index finger drawing the letter in the air: three strokes, so the
    index tip reverses horizontally at least twice.

    Deliberately rule-based rather than a second network. With no training data
    for either letter, a legible rule that can be tuned against a real signer
    beats a model fit to nothing.
    """
    if len(paths) < 10:
        return None
    P = np.array(paths)                       # (T, 21, 2) pixels
    span = np.linalg.norm(P[:, IDX_MCP] - P[:, PINKY_MCP], axis=1).mean()
    if span < 1e-3:
        return None
    jitter = span * 0.06

    # One handshape must dominate the whole stroke. J is the I handshape moved;
    # Z is one pointing handshape moved. Without this, the trajectory window --
    # which has to be ~1s long to contain a Z at all -- spans three or four
    # letters of ordinary fast spelling, finds plenty of fingertip travel and
    # plenty of direction reversals in them, and calls it a Z: GRACE came out
    # GRAZE and CISCO came out CIJ. A word being spelled has no dominant shape.
    if not shapes:
        return None
    common = max(set(shapes), key=shapes.count)
    if shapes.count(common) / len(shapes) < 0.60:
        return None

    pinky = P[:, PINKY_TIP]
    index = P[:, INDEX_TIP]
    pinky_travel = np.linalg.norm(np.diff(pinky, axis=0), axis=1).sum() / span
    index_travel = np.linalg.norm(np.diff(index, axis=0), axis=1).sum() / span

    # Z: index leads, zigzags horizontally, AND the index is actually extended.
    # Without the last condition F fired as Z every time: forming F pinches the
    # index to the thumb, and that fingertip travel looks exactly like a stroke.
    THUMB_TIP = 4
    pinch = np.linalg.norm(P[:, INDEX_TIP] - P[:, THUMB_TIP], axis=1).mean() / span
    if (index_travel > 1.4 and index_travel > pinky_travel and pinch > 0.9
            and _reversals(index, 0, jitter) >= 2):
        return "Z"

    # J: pinky leads, one hook, and the shape was reading as I throughout
    if (common == "I" and pinky_travel > 1.0 and pinky_travel > index_travel
            and _reversals(pinky, 1, jitter) >= 1):
        return "J"
    return None


# ------------------------------------------------------------------ engine

class Engine:
    def __init__(self, model_path, distortion_dir=None, device=None, video=True,
                 tuning=None):
        self.tune = tuning or Tuning()
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = Net().to(self.dev)
        self.net.load_state_dict(torch.load(model_path, map_location=self.dev))
        self.net.eval()

        # VIDEO mode, not IMAGE. IMAGE re-runs full palm detection on every frame;
        # VIDEO tracks the hand between frames and measured 7.68ms against 15.14ms
        # with identical detection (60/60 on the same input). Two landmarkers,
        # because VIDEO carries temporal state -- feeding the left and right eye
        # through one instance would have each eye corrupting the other's track.
        # video=False gives independent-frame IMAGE mode, which is what offline
        # tile building needs -- there, consecutive frames are different letters
        # and tracking state would leak across them.
        # The two eyes are NOT the same kind of stream, so they do not get the
        # same mode. The left eye is continuous -> VIDEO, which tracks between
        # frames and halves the cost. The right eye is sampled every
        # STEREO_EVERY frames -> IMAGE, because VIDEO would try to track across
        # gaps that are not there. Running the right eye in VIDEO raised the
        # geometry-gate rejection rate from 11/20 to 16/20 on the stereo set.
        self.video = video
        self.lm = self._mk(vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE)
        self.lm_r = self._mk(vision.RunningMode.IMAGE)
        self._ts = 0            # monotonic ms, required by detect_for_video

        self.inv = None
        if distortion_dir:
            self.inv = (build_inverse(load_distortion(os.path.join(distortion_dir, "distortion_L.txt"))),
                        build_inverse(load_distortion(os.path.join(distortion_dir, "distortion_R.txt"))))

        self.buf = deque(maxlen=MAX_BUF)      # (t, probs, pts_px, shape_letter)
        self.frame_pred = None                # this frame's answer, or None
        self._changed = 0                     # consecutive frames disagreeing with the buffer
        self.armed = True                     # may the next letter be emitted?
        self.stereo_ok = True                 # carried between periodic checks
        self.last_knuckle = float("nan")
        self.last_letter, self.last_t = None, -1e9
        self.state = "IDLE"
        self.stats = dict(frames=0, no_hand=0, gated=0, voted=0, emitted=0,
                          retracted=0, relocks=0)

        # motion + dwell bookkeeping
        self._prev_pts, self._prev_t, self._prev_frame = None, None, None
        self._palm_h = deque(maxlen=3)        # raw speeds; the metrics are medians
        self._shape_h = deque(maxlen=3)
        self._jit_h = deque(maxlen=3)
        self.travel_speed = 0.0               # spans/sec, the hand moving
        self.shape_speed = 0.0                # spans/sec, the hand changing shape
        self.jitter = 0.0
        self.move_speed, self.straightness = 0.0, 0.0
        self._fast = 0                        # consecutive frames of real movement
        self._unstill = 0                     # consecutive frames judged not-still
        self._traj = deque(maxlen=150)        # (t, pts, pred) for J/Z, ~1.5s
        self._path = deque(maxlen=90)         # (t, centroid, span) -- survives a
                                              # buffer flush, so movement is still
                                              # measurable right after an emission
        self._jitter_since = None
        self._moving_until = -1e9             # nothing commits before this time
        self._cand, self._cand_since = None, None
        self.dwell = 0.0                      # how long the current shape has held
        self._last_em = None                  # (letter, t, dwell) of the last commit

    # -- perception -------------------------------------------------------
    @staticmethod
    def _mk(mode):
        return vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=MODEL_TASK),
                running_mode=mode, num_hands=1,
                min_hand_detection_confidence=0.1,
                min_hand_presence_confidence=0.1))

    def _landmarks(self, gray, right_eye=False):
        c = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
        up = cv2.resize(c, (640, 480), interpolation=cv2.INTER_CUBIC)
        rot = cv2.rotate(up, cv2.ROTATE_180)
        rgb = np.ascontiguousarray(cv2.cvtColor(rot, cv2.COLOR_GRAY2RGB))
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        if right_eye:
            res = self.lm_r.detect(img)          # always IMAGE mode
        elif self.video:
            self._ts += 10                       # must strictly increase
            res = self.lm.detect_for_video(img, self._ts)
        else:
            res = self.lm.detect(img)
        if not res.hand_landmarks:
            return None, None
        pts = np.array([[p.x * 640, p.y * 480] for p in res.hand_landmarks[0]], np.float32)
        return pts, rot

    @staticmethod
    def _silhouette(img, pts):
        """Upright, hull-masked, bright-on-black 32x32 -- matches the training domain."""
        v = pts[MID_MCP] - pts[WRIST]
        ang = np.degrees(np.arctan2(v[0], -v[1]))
        c = (float(pts[WRIST][0]), float(pts[WRIST][1]))
        M = cv2.getRotationMatrix2D(c, -ang, 1.0)
        rimg = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        rp = (M[:, :2] @ pts.T).T + M[:, 2]

        mask = np.zeros(rimg.shape[:2], np.uint8)
        cv2.fillConvexPoly(mask, cv2.convexHull(rp.astype(np.int32)), 255)
        span = np.linalg.norm(rp.max(0) - rp.min(0))
        k = max(3, int(span * 0.16) | 1)
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))

        f = rimg.astype(np.float32)
        inside = f[mask > 0]
        if inside.size < 50:
            return None
        lo, hi = np.percentile(inside, 25), np.percentile(inside, 99)
        f = np.clip((f - lo) / max(hi - lo, 1e-3), 0, 1)
        f[mask == 0] = 0.0

        ys, xs = np.where(mask > 0)
        crop = f[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        h, w = crop.shape
        s = max(h, w)
        sq = np.zeros((s, s), np.float32)
        sq[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = crop
        return cv2.resize(sq, (32, 32), interpolation=cv2.INTER_AREA)

    # -- motion -----------------------------------------------------------
    #
    # Three different speeds, because they answer three different questions and
    # the old single number could not tell them apart:
    #
    #   palm   the rigid part -- wrist and knuckles. This is TRAVEL: the hand
    #          moving from one place to another. Finger noise cannot fake it.
    #   shape  the fingertips expressed in the palm's own frame. This is
    #          MORPHING: one handshape becoming another, which is exactly what a
    #          transition between two letters is, even if the hand never moves.
    #   jitter every landmark, unweighted. Only used to detect a broken track.
    #
    # All normalised by the knuckle span and divided by real elapsed time, so a
    # number means the same thing near the sensor and far from it, at 55fps and
    # at 115fps.
    PALM_PTS = [0, 5, 9, 13, 17]
    TIP_PTS = [4, 8, 12, 16, 20]

    @staticmethod
    def _palm_frame(p):
        o = p[WRIST]
        v = p[MID_MCP] - o
        s = float(np.linalg.norm(v))
        if s < 1e-3:
            return None
        e1 = (v / s).astype(np.float32)
        e2 = np.array([-e1[1], e1[0]], np.float32)
        return o, np.stack([e1, e2]), s

    def _motion(self, pts, t):
        span = float(np.linalg.norm(pts[IDX_MCP] - pts[PINKY_MCP]))
        pf = self._palm_frame(pts)
        prev, pt, pprev_f = self._prev_pts, self._prev_t, self._prev_frame
        self._prev_pts, self._prev_t, self._prev_frame = pts, t, pf
        if prev is None or pt is None or span < 1e-3 or pf is None or pprev_f is None:
            return
        dt = t - pt
        if not (1e-4 < dt < 0.5):        # a gap in the stream is not a movement
            return
        # Centroid, not per-landmark mean. Independent landmark noise cancels in a
        # centroid while real travel does not, and that difference is large: a held
        # D measures 26.3 spans/sec as a per-landmark mean and 10.2 as a centroid,
        # a held C 4.4 against 2.3, a held B 1.1 against 0.5. The old per-landmark
        # number could not tell a dancing landmark from a moving hand.
        self._path.append((t, pts.mean(0), span))
        travel = float(np.linalg.norm(pts.mean(0) - prev.mean(0))) / span / dt
        jit = float(np.linalg.norm(pts - prev, axis=1).mean()) / span / dt
        o, R, s = pf
        po, pR, ps = pprev_f
        now = (R @ (pts[self.TIP_PTS] - o).T).T / s
        was = (pR @ (prev[self.TIP_PTS] - po).T).T / ps
        # Median over the five fingertips: one tip the tracker has lost does not
        # get to declare that the handshape changed.
        shp = float(np.median(np.linalg.norm(now - was, axis=1))) / dt

        # Median of the last three frames, not an exponential average. Both kill
        # the single-frame spike, but an EMA also carries the spike forward: after
        # a real transition its tail kept the reader in SETTLING for another 40-50ms
        # of hold time, which is most of a fast letter. A median forgets the spike
        # completely two frames later.
        for h, v in ((self._palm_h, travel), (self._shape_h, shp), (self._jit_h, jit)):
            h.append(v)
        self.travel_speed = float(np.median(self._palm_h))
        self.shape_speed = float(np.median(self._shape_h))
        self.jitter = float(np.median(self._jit_h))

    def _movement(self, t):
        """(speed, straightness) of the hand's centroid over the recent window."""
        w = [p for p in self._path if t - p[0] <= self.tune.move_win]
        if len(w) < 4:
            return 0.0, 0.0
        C = np.array([p[1] for p in w])
        span = float(np.mean([p[2] for p in w])) or 1.0
        dur = w[-1][0] - w[0][0]
        if dur <= 1e-3:
            return 0.0, 0.0
        path = float(np.linalg.norm(np.diff(C, axis=0), axis=1).sum()) / span
        net = float(np.linalg.norm(C[-1] - C[0])) / span
        return path / dur, net / max(path, 1e-6)

    def _track_health(self, t):
        """A landmarker that has latched onto the wrong hypothesis never recovers.

        Measured, not assumed: cold-starting the VIDEO tracker on a sideways H
        read G on 400 frames out of 400, with landmark jitter of 15.4 spans/sec.
        The same frames, with the tracker warmed on the previous letter, read H
        398/400 at 3.2. So sustained jitter far above what any held hand produces
        is not noise to be voted away -- it is a dead track, and the only fix is
        to throw the tracker away and let it find the hand again.
        """
        T = self.tune
        if self.jitter <= T.jitter_max:
            self._jitter_since = None
            return False
        if self._jitter_since is None:
            self._jitter_since = t
            return False
        if t - self._jitter_since < T.jitter_sec:
            return False
        self._jitter_since = None
        try:
            self.lm.close()
        except Exception:
            pass
        self.lm = self._mk(vision.RunningMode.VIDEO if self.video
                           else vision.RunningMode.IMAGE)
        self.buf.clear()
        self.jitter = 0.0
        self._prev_pts = None
        self._path.clear()
        self._traj.clear()
        self.stats["relocks"] += 1
        self.state = "RELOCK"
        return True

    # -- geometry gate ----------------------------------------------------
    def _metric_ok(self, pL, pR):
        """True if the two eyes reconstruct something the size of a hand.

        Cheap and decisive: a wrong stereo correspondence almost never produces
        a plausible metacarpal row. Skipped when only one eye is available.
        """
        if self.inv is None or pR is None:
            return True, float("nan")
        nL = pL.copy(); nL[:, 0] /= 640.0; nL[:, 1] /= 480.0
        nR = pR.copy(); nR[:, 0] /= 640.0; nR[:, 1] /= 480.0
        nL[:, 0] = 1.0 - nL[:, 0]; nL[:, 1] = 1.0 - nL[:, 1]
        nR[:, 0] = 1.0 - nR[:, 0]; nR[:, 1] = 1.0 - nR[:, 1]
        xyz, ok = triangulate(nL.astype(np.float64), nR.astype(np.float64), *self.inv)
        if not (ok[IDX_MCP] and ok[PINKY_MCP]):
            return False, float("nan")
        kn = float(np.linalg.norm(xyz[IDX_MCP] - xyz[PINKY_MCP]))
        return (KNUCKLE_MIN_MM <= kn <= KNUCKLE_MAX_MM), kn

    # -- main -------------------------------------------------------------
    def push(self, gray_L, gray_R=None, t=None):
        t = float(self.stats["frames"]) / 115.0 if t is None else t
        self.stats["frames"] += 1
        # What THIS frame classified as, or None if it never reached the CNN.
        # Scoring must not fall back to reading the buffer's tail: a gated frame
        # would then be scored using its predecessor's answer and counted twice.
        self.frame_pred = None

        pL, rotL = self._landmarks(gray_L)
        if pL is None:
            self.stats["no_hand"] += 1
            self.buf.clear()                  # silence resets the vote
            self.state = "IDLE"
            self.armed = True
            self._prev_pts = None
            self._cand, self._cand_since = None, None
            return None

        self._motion(pL, t)
        if self._track_health(t):
            return None

        # Re-check immediately after a failure rather than carrying it: a stale
        # bad verdict would gate STEREO_EVERY frames for one bad check. Carrying
        # a *good* verdict is safe, carrying a bad one is not.
        if gray_R is not None and self.inv is not None and \
                (not self.stereo_ok or self.stats["frames"] % STEREO_EVERY == 1):
            pR, _ = self._landmarks(gray_R, right_eye=True)
            self.stereo_ok, self.last_knuckle = self._metric_ok(pL, pR)
        if not self.stereo_ok:
            self.stats["gated"] += 1
            return None                       # drop the frame, keep the window

        tile = self._silhouette(rotL, pL)
        if tile is None:
            return None
        x = torch.tensor(np.stack([tile, tile])[None], dtype=torch.float32).to(self.dev)
        with torch.no_grad():
            probs = torch.softmax(self.net(x), 1)[0].cpu().numpy()

        self.frame_pred = LETTERS[int(probs.argmax())]

        # If the hand has moved to a NEW letter, the frames from the old one are
        # no longer evidence -- they are interference. Without this, a letter that
        # was just committed keeps refilling the buffer while it is still being
        # held, and the next letter never wins agreement against it. That is what
        # dropped the middle letter of a word: C-A-B came out CB because A spent
        # its whole life outvoted by leftover C frames. Two consecutive
        # disagreeing frames, so single-frame noise does not flush the buffer.
        if self.buf:
            majority = max(set(b[3] for b in self.buf),
                           key=[b[3] for b in self.buf].count)
            if self.frame_pred != majority:
                self._changed += 1
                if self._changed >= 2:
                    self.buf.clear()
                    self.armed = True          # a different letter may always emit
                    self._changed = 0
            else:
                self._changed = 0

        self.buf.append((t, probs, pL, self.frame_pred))
        self._traj.append((t, pL, self.frame_pred))

        # ---- dwell. The candidate clock runs only while the hand is BOTH holding
        # this shape and still. Any movement, or any change of shape, restarts it.
        # That is the whole anti-transition mechanism: a shape the hand is passing
        # through never accumulates dwell, because it is moving the entire time it
        # is being passed through.
        T = self.tune
        # One noisy frame is not a movement. Some handshapes jitter hard enough
        # while dead still (H, D, O) that a per-frame stillness test would lock
        # them out of the reader entirely, so it takes still_grace consecutive
        # over-threshold frames to break a hold. A real transition is a hundred
        # milliseconds long and clears that easily.
        over = (self.travel_speed > T.still_travel or self.shape_speed > T.still_shape)
        self._unstill = self._unstill + 1 if over else 0
        self.move_speed, self.straightness = self._movement(t)
        self._fast = (self._fast + 1 if (self.move_speed > T.move_min
                                         and self.straightness > T.move_straight) else 0)
        moving = self._fast >= T.move_frames
        still = ((not T.still_gate) or self._unstill < T.still_grace) and not moving
        # The lockout is driven by SUSTAINED motion only. A single noisy frame
        # cannot silence the reader, and a real hand crossing from one letter to
        # the next cannot avoid tripping it.
        if moving:
            self._moving_until = t + T.settle_sec
        if self.frame_pred != self._cand:
            self._cand, self._cand_since = self.frame_pred, (t if still else None)
        elif not still:
            self._cand_since = None
        elif self._cand_since is None:
            self._cand_since = t
        self.dwell = 0.0 if self._cand_since is None else t - self._cand_since

        return self._vote(t)

    def _vote(self, t):
        T = self.tune
        # The window is a duration, not a frame count. Trim by time so the reader
        # behaves identically whether the box is delivering 55fps or 115.
        while len(self.buf) > 1 and t - self.buf[0][0] > T.window_sec:
            self.buf.popleft()

        if len(self.buf) < T.min_votes:
            self.state = "WATCHING"
            # Deliberately does NOT re-arm. The buffer is emptied after every
            # emission, so WATCHING is the normal state immediately AFTER a letter
            # -- re-arming here let the same held letter fire again as soon as the
            # buffer refilled, turning CAB into CCABB. Re-arming belongs to the
            # states that mean the hand actually left the letter: SETTLING, UNSURE,
            # MOVING, IDLE.
            return None

        pts = np.array([b[2] for b in self.buf])

        # a deliberate stroke means J or Z, not a held handshape
        if self._fast >= T.move_frames:
            self.state = "MOVING"
            tj = [x for x in self._traj if t - x[0] <= T.motion_sec]
            mot = classify_motion([x[1] for x in tj], [x[2] for x in tj])
            self.armed = True                 # a stroke means the hand left the letter
            if mot and (mot != self.last_letter or t - self.last_t > T.cooldown):
                self.buf.clear()
                self.last_letter, self.last_t = mot, t
                self.stats["emitted"] += 1
                self._last_em = (mot, t, T.solid_sec)   # never retract a stroke
                return Emission(mot, 0.60, t, len(pts), kind="motion", dwell=T.solid_sec)
            return None

        if T.still_gate and self._unstill >= T.still_grace:
            self.state = "SETTLING"
            self.armed = True          # moved off the letter -- it may repeat now
            return None

        # Transition lockout. Any movement above the still thresholds pushes this
        # forward, so the quiet moment INSIDE a transition -- the instant the hand
        # slows at the top of its arc, which is what was firing spurious letters --
        # is still inside the shadow of the movement that produced it.
        if t < self._moving_until:
            self.state = "SETTLING"
            return None

        # ...and the shape itself must have stood still long enough to be a letter
        # somebody meant, rather than one the hand drove through.
        # A millisecond of tolerance. The dwell is a difference of two frame
        # timestamps, so it lands on the threshold from below by a hair -- a
        # letter held for exactly the required time measured 0.15999999s and was
        # dropped, which cost CAB its A.
        if self.dwell < T.dwell_sec - 1e-3:
            self.state = "DWELL"
            return None

        P = np.array([b[1] for b in self.buf]).mean(axis=0)
        order = P.argsort()[::-1]
        top, second = order[0], order[1]

        # do the individual frames actually agree, or does only the average look good?
        winner = LETTERS[top]
        agree = sum(1 for b in self.buf if b[3] == winner) / len(self.buf)
        if agree < T.agree_min or winner != self._cand:
            self.state = "SETTLING"
            self.armed = True
            return None
        entropy = float(-(P * np.log(P + 1e-9)).sum())
        margin = float(P[top] - P[second])
        self.stats["voted"] += 1

        # Rejection. A softmax is never uncertain on its own -- it will hand back
        # a confident letter for a coffee mug. Entropy and margin are what make
        # "I don't know" expressible.
        if entropy > T.entropy_max or margin < T.margin_min:
            self.state = "UNSURE"
            self.armed = True
            return None

        letter = LETTERS[top]
        # A held letter must not stutter. Re-emitting the same letter after a
        # cooldown turns a signer who pauses on F into "FF". Genuine doubles in
        # fingerspelling are made with a bounce, not a long hold -- and a bounce
        # registers as motion, which re-arms below. So the same letter repeats
        # only after the hand has actually left it.
        if letter == self.last_letter and not self.armed:
            self.state = "HELD"
            return None

        # Retraction. Even with dwell and lockout, a transition that genuinely
        # pauses can still commit. The tell comes AFTERWARDS: the letter the hand
        # was actually going to sits there far longer than the one it brushed. So
        # a letter committed on a near-floor dwell, immediately followed by one
        # that held much longer, is taken back out of the word.
        #
        # The ratio is what keeps honest fast spelling safe -- evenly paced letters
        # all dwell about the same, so nothing ever retracts anything.
        dwell = self.dwell
        retract = False
        if T.retract and self._last_em:
            pl, pt_, pd = self._last_em
            if (pl != letter and t - pt_ <= T.retract_sec
                    and pd < T.solid_sec and dwell >= pd * 1.6):
                retract = True
                self.stats["retracted"] += 1

        self.armed = False
        self.last_letter, self.last_t = letter, t
        self.state = "EMIT"
        self.stats["emitted"] += 1
        self._last_em = (letter, t, dwell)
        n = len(self.buf)
        # Start the next letter from a clean buffer. Otherwise the frames that
        # produced THIS letter linger and fight the next one for agreement, which
        # is what made fast spelling drop letters in the middle of a word.
        self.buf.clear()
        self._cand_since = t              # a committed letter starts dwelling again
        return Emission(letter, float(P[top]), t, n, dwell=dwell, retract_prev=retract)

    def close(self):
        self.lm.close()
        self.lm_r.close()


# ------------------------------------------------------------------ replay

def replay(recdir, model_path, preset="normal"):
    eng = Engine(model_path, tuning=Tuning(**PRESETS[preset]), distortion_dir=recdir
                 if os.path.exists(os.path.join(recdir, "distortion_L.txt")) else None)
    ids = sorted({f.split("_")[0] for f in os.listdir(recdir) if f.endswith("_L.pgm")},
                 key=lambda s: int(s[1:]))
    if not ids:      # h2h-style single-image dirs
        ids = sorted([f[:-4] for f in os.listdir(recdir) if f.endswith(".pgm")],
                     key=lambda s: int(s[1:]))

    out = []
    for i, fid in enumerate(ids):
        pl = os.path.join(recdir, f"{fid}_L.pgm")
        if not os.path.exists(pl):
            pl = os.path.join(recdir, f"{fid}.pgm")
        gl = cv2.imread(pl, cv2.IMREAD_GRAYSCALE)
        pr = os.path.join(recdir, f"{fid}_R.pgm")
        gr = cv2.imread(pr, cv2.IMREAD_GRAYSCALE) if os.path.exists(pr) else None
        if gl is None:
            continue
        ev = eng.push(gl, gr, t=i / 100.0)      # leap-rec default interval is 10ms
        print(f"  {fid:>8}  state={eng.state:<9} {ev if ev else ''}")
        if ev:
            out.append(ev)

    s = eng.stats
    print(f"\n  frames {s['frames']}  no-hand {s['no_hand']}  "
          f"geometry-gated {s['gated']}  voted {s['voted']}  emitted {s['emitted']}")
    print(f"  letters: {' '.join(e.letter for e in out) if out else '(none)'}")
    eng.close()
    return out


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--replay"]
    recdir = args[0] if args else os.path.expanduser("~/leap/stereo")
    model = args[1] if len(args) > 1 else os.path.expanduser("~/leap/asl/model_cisco26b.pt")
    replay(recdir, model)
