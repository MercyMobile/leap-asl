"""Score the reader against labeled capture. This is what the interpreter hour buys.

Reports three different numbers, because they answer three different questions:

  per-frame     -- how often a single frame is right. Comparable to the 81.11%
                   leave-one-subject-out figure from ASLA-Leap, so it tells us
                   how much accuracy was lost crossing from their rig to ours.
  after voting  -- how often a committed letter is right. This is the number a
                   user actually experiences, and it should beat per-frame,
                   because voting is there to discard bad frames.
  coverage      -- what fraction of held letters produced any answer at all.
                   A reader that is always right and never speaks is useless,
                   so accuracy without coverage is a vanity metric.

  score.py --subject interp1              # posed
  score.py --subject interp1 --mode speed # co-articulated
"""
import argparse
import csv
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import LETTERS
from stream import Engine

RAW = os.path.expanduser("~/leap/asl/data/raw")


def frames_in(d):
    ids = sorted({f.split("_")[0] for f in os.listdir(d) if f.endswith("_L.pgm")},
                 key=lambda s: int(s[1:]))
    for fid in ids:
        gl = cv2.imread(os.path.join(d, f"{fid}_L.pgm"), cv2.IMREAD_GRAYSCALE)
        rp = os.path.join(d, f"{fid}_R.pgm")
        gr = cv2.imread(rp, cv2.IMREAD_GRAYSCALE) if os.path.exists(rp) else None
        if gl is not None:
            yield fid, gl, gr


def score_posed(subject, model, mode="posed"):
    root = os.path.join(RAW, subject, mode)
    if not os.path.isdir(root):
        sys.exit(f"no capture at {root} -- run capture.py first")

    labels = sorted(os.listdir(root))
    per_frame = Counter()
    per_frame_tot = Counter()
    emissions = defaultdict(list)
    silent = []

    for label in labels:
        d = os.path.join(root, label)
        if not os.path.isdir(d):
            continue
        dist = d if os.path.exists(os.path.join(d, "distortion_L.txt")) else None
        eng = Engine(model, distortion_dir=dist)
        got = []
        for i, (fid, gl, gr) in enumerate(frames_in(d)):
            ev = eng.push(gl, gr, t=i / 100.0)
            if eng.buf:
                pred = eng.buf[-1][3]
                per_frame_tot[label] += 1
                if pred == label:
                    per_frame[label] += 1
            if ev:
                got.append(ev.letter)
        eng.close()
        emissions[label] = got
        if not got:
            silent.append(label)

    # ---- report
    print(f"\n  subject {subject}  mode {mode}  model {os.path.basename(model)}\n")
    print(f"  {'letter':<7}{'frames':>8}{'per-frame':>11}{'emitted':>9}{'correct':>9}")
    print("  " + "-" * 44)

    fr_ok = fr_tot = em_ok = em_tot = 0
    rows = []
    for label in labels:
        if not per_frame_tot[label] and not emissions[label]:
            continue
        tot = per_frame_tot[label]
        acc = per_frame[label] / tot * 100 if tot else float("nan")
        em = emissions[label]
        ok = sum(1 for e in em if e == label)
        fr_ok += per_frame[label]; fr_tot += tot
        em_ok += ok; em_tot += len(em)
        rows.append((label, acc, len(em), ok))
        flag = "" if ok else ("  <- silent" if not em else "  <- all wrong")
        print(f"  {label:<7}{tot:>8}{acc:>10.1f}%{len(em):>9}{ok:>9}{flag}")

    print("  " + "-" * 44)
    pf = fr_ok / fr_tot * 100 if fr_tot else float("nan")
    ev = em_ok / em_tot * 100 if em_tot else float("nan")
    cov = (len(labels) - len(silent)) / max(len(labels), 1) * 100
    print(f"  per-frame accuracy   {pf:5.1f}%   ({fr_ok}/{fr_tot})")
    print(f"  after voting         {ev:5.1f}%   ({em_ok}/{em_tot} committed letters)")
    print(f"  coverage             {cov:5.1f}%   ({len(labels)-len(silent)}/{len(labels)} letters produced an answer)")
    if silent:
        print(f"  never answered: {' '.join(silent)}")

    # ---- confusions, only the ones that actually happen
    conf = Counter()
    for label, em in emissions.items():
        for e in em:
            if e != label:
                conf[(label, e)] += 1
    if conf:
        print("\n  top confusions (signed -> read as):")
        for (a, b), n in conf.most_common(12):
            print(f"    {a} -> {b}   {n}")
    return dict(per_frame=pf, voted=ev, coverage=cov)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subject", required=True)
    p.add_argument("--mode", default="posed", choices=["posed", "speed"])
    p.add_argument("--model", default=os.path.expanduser("~/leap/asl/model_loso0.pt"))
    a = p.parse_args()
    score_posed(a.subject, a.model, a.mode)


if __name__ == "__main__":
    main()
