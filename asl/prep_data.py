"""Load ASLA-Leap into arrays and cache as .npy.

54,000 samples, 24 static ASL letters (J and Z excluded -- they are the dynamic
letters), 5 subjects, stereo 32x32 IR hand crops.
Row i of labels.csv corresponds to image i.jpg in both left/ and right/.
"""
import os, csv, sys
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.expanduser("~/leap/ASLA-Leap/dataset")
OUT = os.path.expanduser("~/leap/asl/data")
os.makedirs(OUT, exist_ok=True)

rows = list(csv.DictReader(open(os.path.join(ROOT, "labels.csv"))))
n = len(rows)
print(f"{n} samples")

subj = np.array([int(r["subject_id"]) for r in rows], dtype=np.int8)
sign = np.array([int(r["sign_id"]) for r in rows], dtype=np.int8)

X = np.zeros((n, 2, 32, 32), dtype=np.uint8)

def load(i):
    for c, eye in enumerate(("left", "right")):
        p = os.path.join(ROOT, eye, eye, f"{i}.jpg")
        X[i, c] = np.asarray(Image.open(p).convert("L"), dtype=np.uint8)

with ThreadPoolExecutor(max_workers=16) as ex:
    for k, _ in enumerate(ex.map(load, range(n))):
        if k % 10000 == 0:
            print(f"  {k}/{n}", flush=True)

np.save(os.path.join(OUT, "X.npy"), X)
np.save(os.path.join(OUT, "y.npy"), sign)
np.save(os.path.join(OUT, "subj.npy"), subj)
print("saved", X.shape, X.dtype, "labels", np.bincount(sign).tolist())
print("subjects", np.bincount(subj).tolist())
