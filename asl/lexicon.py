"""Sign memory: teach a new sign by showing it, not by retraining.

Why this is not a classifier
----------------------------
The 26 letters are a closed set -- nobody is going to invent a 27th -- so a
trained network is the right tool there, and retraining is a one-time cost.

Vocabulary is the opposite. It is open and it grows forever. A trained
classifier would need retraining for every word added, which makes adding a word
an engineering task instead of a thirty-second recording. So signs are stored as
*examples* and matched by similarity. Adding a sign is writing a file.

This is the same structure as an ordinary memory: a name, an example, and a
lookup. Show it three times, it is learnable immediately, and nothing else in
the system has to change.

Matching uses dynamic time warping, which compares two sequences that run at
different speeds. That matters because nobody signs the same word twice at the
same rate -- a signer who is tired, emphatic, or talking to a child stretches
and compresses the same sign, and DTW treats those as the same thing while a
fixed-length comparison would not.

  lexicon.py add HELLO  ~/leap/asl/data/raw/interp1/signs/HELLO
  lexicon.py list
  lexicon.py match ~/leap/some-capture
"""
import argparse
import json
import os
import sys

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

LEX = os.path.expanduser("~/leap/asl/lexicon")
RESAMPLE = 32                 # frames every sequence is resampled to
IDX_MCP, PINKY_MCP, WRIST = 5, 17, 0
# DTW distance past which we refuse to name a sign.
#
# This must be tight. On synthetic sequences, repeats of a taught sign land at
# 0.00-0.012 and a *different* taught sign at 0.589 -- but a gesture that was
# never taught at all landed at 0.351, i.e. comfortably inside a loose threshold
# and wrongly accepted. An interpreter signing a word we do not know must get
# "unknown", never the nearest thing in the lexicon; a confident wrong word is
# worse than a blank, because the reader cannot tell it was a guess.
#
# CALIBRATE THIS on real repeats: record the same sign twice from the same person
# and set the threshold above that spread, not from these synthetic numbers.
UNKNOWN_ABOVE = 0.15


# ---------------------------------------------------------------- normalize

def normalize_frame(pts):
    """21x2 pixels -> 21x2 invariant to where the hand is, how big, how rotated.

    Wrist to the origin, knuckle row to unit length and to the x-axis. What
    survives is the shape of the hand, which is what identifies a sign. Where the
    signer stood and how close they leaned do not.
    """
    p = np.asarray(pts, np.float64) - pts[WRIST]
    v = p[PINKY_MCP] - p[IDX_MCP]
    n = np.linalg.norm(v)
    if n < 1e-6:
        return None
    p = p / n
    c, s = v[0] / n, v[1] / n
    R = np.array([[c, s], [-s, c]])       # rotate knuckle row onto +x
    return (R @ p.T).T


def normalize_seq(frames):
    """List of 21x2 -> (T, 42), dropping frames that cannot be normalized."""
    out = [normalize_frame(f) for f in frames]
    out = [o.ravel() for o in out if o is not None]
    return np.array(out) if out else None


def resample(seq, n=RESAMPLE):
    """Linear resample to a fixed length so sequences are comparable."""
    if len(seq) == n:
        return seq
    idx = np.linspace(0, len(seq) - 1, n)
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, len(seq) - 1)
    w = (idx - lo)[:, None]
    return seq[lo] * (1 - w) + seq[hi] * w


# ---------------------------------------------------------------- matching

def dtw(a, b, band=None):
    """Dynamic time warping distance between two (T, D) sequences.

    A plain Euclidean comparison would demand the two recordings line up frame
    for frame. DTW allows one to stretch against the other, which is the whole
    point: the same sign done slowly and quickly should match.

    `band` is a Sakoe-Chiba constraint -- it forbids matches that warp time too
    far, which both speeds this up and stops pathological alignments where a
    single frame absorbs half the other sequence.
    """
    n, m = len(a), len(b)
    band = band if band is not None else max(4, int(0.25 * max(n, m)))
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    # pairwise frame distances up front -- this is the expensive part
    cost = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    for i in range(1, n + 1):
        lo = max(1, i - band)
        hi = min(m, i + band)
        c = cost[i - 1, lo - 1:hi]
        prev = D[i - 1, lo - 1:hi]
        prev_diag = D[i - 1, lo - 2:hi - 1] if lo > 1 else np.concatenate(
            [[D[i - 1, 0]], D[i - 1, lo - 1:hi - 1]])
        best = np.minimum(prev, prev_diag)
        for j in range(lo, hi + 1):
            k = j - lo
            D[i, j] = c[k] + min(best[k], D[i, j - 1])
    return D[n, m] / (n + m)


# ---------------------------------------------------------------- lexicon

class Lexicon:
    def __init__(self, path=LEX):
        self.path = path
        os.makedirs(path, exist_ok=True)
        self.entries = {}          # gloss -> list of (T, D) arrays
        self.load()

    def load(self):
        idx = os.path.join(self.path, "index.json")
        if not os.path.exists(idx):
            return
        for gloss, files in json.load(open(idx)).items():
            seqs = []
            for f in files:
                p = os.path.join(self.path, f)
                if os.path.exists(p):
                    seqs.append(np.load(p))
            if seqs:
                self.entries[gloss] = seqs

    def save(self):
        index = {}
        for gloss, seqs in self.entries.items():
            files = []
            for i, s in enumerate(seqs):
                fn = f"{gloss}__{i}.npy"
                np.save(os.path.join(self.path, fn), s)
                files.append(fn)
            index[gloss] = files
        json.dump(index, open(os.path.join(self.path, "index.json"), "w"), indent=1)

    def add(self, gloss, seq):
        """One more example of a sign. Three or so is usually enough to start."""
        self.entries.setdefault(gloss, []).append(resample(seq))
        self.save()
        return len(self.entries[gloss])

    def match(self, seq, top=3):
        """Nearest examples by DTW. Returns [(gloss, distance), ...] best first."""
        if not self.entries:
            return []
        q = resample(seq)
        scored = []
        for gloss, seqs in self.entries.items():
            d = min(dtw(q, s) for s in seqs)     # best of that sign's examples
            scored.append((gloss, float(d)))
        scored.sort(key=lambda x: x[1])
        return scored[:top]

    def read(self, seq):
        """Match, but willing to say nothing -- an unknown sign must not be forced
        onto the nearest known one."""
        r = self.match(seq, top=2)
        if not r or r[0][1] > UNKNOWN_ABOVE:
            return None, r
        return r[0], r


# ---------------------------------------------------------------- capture io

def seq_from_dir(recdir, video=True):
    """Recorded frames -> a normalized sequence, using the live preprocessing."""
    from stream import Engine
    eng = Engine.__new__(Engine)
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    from leap_pose import MODEL_TASK
    mk = lambda m: vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_TASK),
            running_mode=m, num_hands=1,
            min_hand_detection_confidence=0.1, min_hand_presence_confidence=0.1))
    eng.lm = mk(vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE)
    eng.lm_r, eng.video, eng._ts = eng.lm, video, 0

    ids = sorted({f.split("_")[0] for f in os.listdir(recdir) if f.endswith("_L.pgm")},
                 key=lambda s: int(s[1:]))
    pts = []
    for fid in ids:
        g = cv2.imread(os.path.join(recdir, f"{fid}_L.pgm"), cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        p, _ = eng._landmarks(g)
        if p is not None:
            pts.append(p)
    eng.lm.close()
    if len(pts) < 4:
        return None
    return normalize_seq(pts)


def main():
    ap = argparse.ArgumentParser(description="Sign memory -- teach by showing")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add");   a.add_argument("gloss"); a.add_argument("recdir")
    sub.add_parser("list")
    m = sub.add_parser("match"); m.add_argument("recdir")
    d = sub.add_parser("drop");  d.add_argument("gloss")
    args = ap.parse_args()

    lex = Lexicon()
    if args.cmd == "list":
        if not lex.entries:
            print("  lexicon is empty")
        for gloss, seqs in sorted(lex.entries.items()):
            print(f"  {gloss:<18} {len(seqs)} example(s)")
        return
    if args.cmd == "drop":
        lex.entries.pop(args.gloss, None); lex.save()
        print(f"  dropped {args.gloss}")
        return

    seq = seq_from_dir(os.path.expanduser(args.recdir))
    if seq is None:
        sys.exit("  no usable hand frames in that recording")

    if args.cmd == "add":
        n = lex.add(args.gloss, seq)
        print(f"  {args.gloss}: {len(seq)} frames stored, {n} example(s) total")
    else:
        hit, all_r = lex.read(seq)
        print(f"  {len(seq)} frames")
        for gloss, dist in all_r:
            print(f"    {gloss:<18} distance {dist:.3f}")
        print(f"\n  -> {hit[0] if hit else 'unknown (nothing close enough)'}")


if __name__ == "__main__":
    main()
