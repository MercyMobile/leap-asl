"""Word-level regression: how fast can the reader be spelled at, and what does
it invent along the way?

Two sources, and the difference between them matters:

  --synth   words assembled from posed captures, one letter's frames butted
            straight against the next. There is NO transition in this signal --
            the hand teleports between shapes. So it measures one thing only:
            whether short letters still get READ at a given pace. It cannot
            measure spurious letters, because there is nothing between the
            letters to misread.

  --clip    a real recording made through reader.py's record button, with the
            word that was actually signed in its name. This is the only source
            that contains transitions, so it is the only one that can measure
            letters the reader invented while the hand was on its way somewhere.

    words.py --synth --hold 200
    words.py --synth --sweep
    words.py --clip data/raw/live/HILLY-1786551000
    words.py --clip data/raw/live/HILLY-1786551000 --sweep
"""
import argparse
import csv
import os
import sys

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stream import Engine, Tuning, PRESETS

RAW = os.path.expanduser("~/leap/asl/data/raw")
POSED = os.path.join(RAW, "cisco_full", "posed")
DEFAULT_MODEL = os.path.expanduser("~/leap/asl/model_cisco26b.pt")
WORDS = ["CAB", "HILLY", "SAD", "BIT", "CISCO", "MERCY", "GRACE", "FIB", "TAB", "MILK"]


# ---------------------------------------------------------------- sources

def posed_frames(letter):
    d = os.path.join(POSED, letter)
    if not os.path.isdir(d):
        return []
    ids = sorted({f.split("_")[0] for f in os.listdir(d) if f.endswith("_L.pgm")},
                 key=lambda s: int(s[1:]))
    return [(os.path.join(d, f"{i}_L.pgm"), os.path.join(d, f"{i}_R.pgm")) for i in ids]


def synth(word, hold_ms, dt_ms=10):
    """Frames for a word, `hold_ms` per letter, taken from the settled middle of
    each posed burst. Cached per letter so a sweep does not re-read the disk."""
    n = max(1, int(hold_ms / dt_ms))
    seq = []
    for ch in word:
        fr = synth.cache.get(ch)
        if fr is None:
            fr = synth.cache[ch] = posed_frames(ch)
        if not fr:
            return None
        mid = len(fr) // 2
        take = fr[max(0, mid - n // 2): max(0, mid - n // 2) + n]
        while len(take) < n:                      # short burst: hold the last frame
            take.append(fr[-1])
        seq += take
    return seq


synth.cache = {}
OVERRIDE = {}


def clip_frames(d):
    """A recorded live clip: frames plus the real timestamps they arrived at."""
    stamps = {}
    sp = os.path.join(d, "stamps.csv")
    if os.path.exists(sp):
        with open(sp) as f:
            for row in csv.reader(f):
                if len(row) >= 2 and row[0] != "frame":
                    stamps[row[0]] = float(row[1])
    ids = sorted({f.split("_")[0] for f in os.listdir(d) if f.endswith("_L.pgm")},
                 key=lambda s: int(s[1:]))
    out = []
    for i, fid in enumerate(ids):
        r = os.path.join(d, f"{fid}_R.pgm")
        out.append((os.path.join(d, f"{fid}_L.pgm"), r if os.path.exists(r) else None,
                    stamps.get(fid, i * 0.01)))
    return out


# ---------------------------------------------------------------- running

def run(eng, frames, dt=0.01, warm=15):
    """Feed frames, apply retractions, return the word the reader produced.

    The first `warm` frames are fed twice: once to give MediaPipe's VIDEO tracker
    something to lock onto, then again as part of the word. Live, the hand is
    already tracked before a word starts; cold-starting the tracker on the first
    letter is an artefact of the harness, and a bad one -- a cold start on a
    sideways H reads G on 400 frames out of 400."""
    eng.buf.clear()
    eng.last_letter, eng.last_t = None, -1e9
    eng._cand, eng._cand_since, eng._last_em = None, None, None
    eng._prev_pts, eng._prev_t, eng._prev_frame = None, None, None
    eng._moving_until = -1e9
    for item in frames[:warm]:              # warm the tracker, discard the answers
        g = cv2.imread(item[0], cv2.IMREAD_GRAYSCALE)
        if g is not None:
            eng.push(g, None, t=-1.0 + 0.01 * warm)
    eng.buf.clear(); eng._cand, eng._cand_since = None, None
    eng._prev_pts, eng._last_em = None, None
    out, detail = [], []
    for i, item in enumerate(frames):
        lp, rp = item[0], item[1]
        t = item[2] if len(item) > 2 else i * dt
        gl = cv2.imread(lp, cv2.IMREAD_GRAYSCALE)
        gr = cv2.imread(rp, cv2.IMREAD_GRAYSCALE) if rp and os.path.exists(rp) else None
        if gl is None:
            continue
        ev = eng.push(gl, gr, t=t)
        if ev:
            if ev.retract_prev and out:
                detail.append(f"-{out[-1]}")
                out.pop()
            out.append(ev.letter)
            detail.append(f"{ev.letter}@{ev.dwell*1000:.0f}ms")
    return "".join(out), detail


def edit(a, b):
    """Levenshtein, so a spurious letter and a missing one cost the same."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def report(name, want, got, detail=None):
    ok = "OK  " if got == want else "MISS"
    print(f"  {ok} {name:>8}  want {want:<10} got {got:<14} "
          f"edits {edit(want, got)}" + (f"   {' '.join(detail)}" if detail else ""))
    return got == want, edit(want, got)


# ---------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--synth", action="store_true")
    p.add_argument("--clip", help="a recorded live clip directory")
    p.add_argument("--sweep", action="store_true", help="try every preset")
    p.add_argument("--hold", type=int, default=200, help="synth: ms per letter")
    p.add_argument("--words", nargs="*", default=WORDS)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--preset", default="normal", choices=list(PRESETS))
    p.add_argument("--set", nargs="*", default=[], metavar="K=V",
                   help="override any Tuning field, e.g. --set still_shape=8 dwell_sec=0.14")
    a = p.parse_args()

    if not a.synth and not a.clip:
        a.synth = True

    over = {}
    for kv in a.set:
        k, _, v = kv.partition("=")
        over[k] = type(getattr(Tuning(), k))(v)
    OVERRIDE.update(over)
    eng = Engine(a.model, tuning=Tuning(**PRESETS[a.preset], **OVERRIDE))

    if a.clip:
        clips = [a.clip] if os.path.isdir(os.path.join(a.clip)) and \
            any(f.endswith("_L.pgm") for f in os.listdir(a.clip)) else \
            [os.path.join(a.clip, d) for d in sorted(os.listdir(a.clip))]
        presets = list(PRESETS) if a.sweep else [a.preset]
        for pr in presets:
            eng.tune = Tuning(**PRESETS[pr], **OVERRIDE)
            print(f"\n  preset {pr}  (dwell {eng.tune.dwell_sec*1000:.0f}ms  "
                  f"settle {eng.tune.settle_sec*1000:.0f}ms  agree {eng.tune.agree_min})")
            tot = bad = 0
            for c in clips:
                want = os.path.basename(c).split("-")[0].upper()
                got, detail = run(eng, clip_frames(c))
                _, e = report(os.path.basename(c), want, got, detail)
                tot += len(want); bad += e
            print(f"       {bad} edits over {tot} letters")
        eng.close()
        return

    holds = [300, 250, 200, 160, 130, 100] if a.sweep else [a.hold]
    presets = list(PRESETS) if a.sweep else [a.preset]
    print(f"  synthetic words -- no transitions in this signal, so this measures\n"
          f"  whether short letters still get READ, not whether extra ones appear.\n")
    for pr in presets:
        eng.tune = Tuning(**PRESETS[pr], **OVERRIDE)
        print(f"\n  preset {pr}  (dwell {eng.tune.dwell_sec*1000:.0f}ms)")
        for h in holds:
            got_all = 0
            for w in a.words:
                fr = synth(w, h)
                if fr is None:
                    continue
                got, detail = run(eng, fr)
                got_all += (got == w)
                if not a.sweep:
                    report(w, w, got, detail)
            print(f"    {h:>4}ms/letter ({1000/h:4.1f} letters/sec)   "
                  f"{got_all}/{len(a.words)} words exact")
    eng.close()


if __name__ == "__main__":
    main()
