"""Streaming fingerspelling: frames in, letters out.

Single-frame classification is not a reader. A reader needs three things the
classifier does not have:

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

# ---- tuning. Every one of these is a guess until scored against a real signer.
WINDOW = 24            # frames voted over (~0.2s at 115fps)
MIN_VOTES = 8          # usable frames required before emitting anything
ENTROPY_MAX = 2.10     # nats. Uniform over 24 letters is ln(24) = 3.18
MARGIN_MIN = 0.12      # top prob minus runner-up, averaged over the window
STILL_MAX = 9.0        # px/frame of landmark drift that still counts as "held"
COOLDOWN = 0.45        # seconds before the same letter may repeat
MOTION_MIN = 14.0      # px/frame that counts as a deliberate stroke (J/Z)


class Emission:
    """A letter the reader is willing to commit to."""

    def __init__(self, letter, conf, t, votes, kind="static"):
        self.letter, self.conf, self.t = letter, conf, t
        self.votes, self.kind = votes, kind

    def __repr__(self):
        return (f"<{self.letter} {self.conf*100:.0f}% "
                f"{self.kind} n={self.votes}>")


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

    common = max(set(shapes), key=shapes.count) if shapes else None

    pinky = P[:, PINKY_TIP]
    index = P[:, INDEX_TIP]
    pinky_travel = np.linalg.norm(np.diff(pinky, axis=0), axis=1).sum() / span
    index_travel = np.linalg.norm(np.diff(index, axis=0), axis=1).sum() / span

    # Z: index leads, and it zigzags horizontally
    if (index_travel > 1.4 and index_travel > pinky_travel
            and _reversals(index, 0, jitter) >= 2):
        return "Z"

    # J: pinky leads, one hook, and the shape was reading as I throughout
    if (common == "I" and pinky_travel > 1.0 and pinky_travel > index_travel
            and _reversals(pinky, 1, jitter) >= 1):
        return "J"
    return None


# ------------------------------------------------------------------ engine

class Engine:
    def __init__(self, model_path, distortion_dir=None, device=None):
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = Net().to(self.dev)
        self.net.load_state_dict(torch.load(model_path, map_location=self.dev))
        self.net.eval()

        opts = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_TASK),
            running_mode=vision.RunningMode.IMAGE, num_hands=1,
            min_hand_detection_confidence=0.1, min_hand_presence_confidence=0.1)
        self.lm = vision.HandLandmarker.create_from_options(opts)

        self.inv = None
        if distortion_dir:
            self.inv = (build_inverse(load_distortion(os.path.join(distortion_dir, "distortion_L.txt"))),
                        build_inverse(load_distortion(os.path.join(distortion_dir, "distortion_R.txt"))))

        self.buf = deque(maxlen=WINDOW)       # (t, probs, pts_px, shape_letter)
        self.last_letter, self.last_t = None, -1e9
        self.state = "IDLE"
        self.stats = dict(frames=0, no_hand=0, gated=0, voted=0, emitted=0)

    # -- perception -------------------------------------------------------
    def _landmarks(self, gray):
        c = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
        up = cv2.resize(c, (640, 480), interpolation=cv2.INTER_CUBIC)
        rot = cv2.rotate(up, cv2.ROTATE_180)
        rgb = np.ascontiguousarray(cv2.cvtColor(rot, cv2.COLOR_GRAY2RGB))
        res = self.lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
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

        pL, rotL = self._landmarks(gray_L)
        if pL is None:
            self.stats["no_hand"] += 1
            self.buf.clear()                  # silence resets the vote
            self.state = "IDLE"
            return None

        pR = None
        if gray_R is not None:
            pR, _ = self._landmarks(gray_R)
        good, _kn = self._metric_ok(pL, pR)
        if not good:
            self.stats["gated"] += 1
            return None                       # drop the frame, keep the window

        tile = self._silhouette(rotL, pL)
        if tile is None:
            return None
        x = torch.tensor(np.stack([tile, tile])[None], dtype=torch.float32).to(self.dev)
        with torch.no_grad():
            probs = torch.softmax(self.net(x), 1)[0].cpu().numpy()

        self.buf.append((t, probs, pL, LETTERS[int(probs.argmax())]))
        return self._vote(t)

    def _vote(self, t):
        if len(self.buf) < MIN_VOTES:
            self.state = "WATCHING"
            return None

        pts = np.array([b[2] for b in self.buf])
        drift = np.linalg.norm(np.diff(pts, axis=0), axis=2).mean()

        # a deliberate stroke means J or Z, not a held handshape
        if drift > MOTION_MIN:
            self.state = "MOVING"
            mot = classify_motion([b[2] for b in self.buf], [b[3] for b in self.buf])
            if mot and (mot != self.last_letter or t - self.last_t > COOLDOWN):
                self.buf.clear()
                self.last_letter, self.last_t = mot, t
                self.stats["emitted"] += 1
                return Emission(mot, 0.60, t, len(pts), kind="motion")
            return None

        if drift > STILL_MAX:
            self.state = "SETTLING"
            return None

        P = np.array([b[1] for b in self.buf]).mean(axis=0)
        order = P.argsort()[::-1]
        top, second = order[0], order[1]
        entropy = float(-(P * np.log(P + 1e-9)).sum())
        margin = float(P[top] - P[second])
        self.stats["voted"] += 1

        # Rejection. A softmax is never uncertain on its own -- it will hand back
        # a confident letter for a coffee mug. Entropy and margin are what make
        # "I don't know" expressible.
        if entropy > ENTROPY_MAX or margin < MARGIN_MIN:
            self.state = "UNSURE"
            return None

        letter = LETTERS[top]
        if letter == self.last_letter and t - self.last_t < COOLDOWN:
            self.state = "HELD"
            return None

        self.last_letter, self.last_t = letter, t
        self.state = "EMIT"
        self.stats["emitted"] += 1
        return Emission(letter, float(P[top]), t, len(self.buf))

    def close(self):
        self.lm.close()


# ------------------------------------------------------------------ replay

def replay(recdir, model_path):
    eng = Engine(model_path, distortion_dir=recdir
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
        ev = eng.push(gl, gr, t=i / 115.0)
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
    model = args[1] if len(args) > 1 else os.path.expanduser("~/leap/asl/model_loso0.pt")
    replay(recdir, model)
