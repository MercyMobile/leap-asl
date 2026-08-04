"""Adapt the ASLA-Leap classifier to our rig and our signers.

The model currently in hand learned from someone else's Leap, someone else's
segmentation, and five people who are not us. `live_infer2` reshapes our frames
to *approximate* that training domain, and that approximation is the largest
untested assumption in the stack. Fine-tuning on frames captured here removes it.

Deliberately keeps 24 output classes. J and Z stay with the motion rule in
stream.py rather than becoming CNN classes -- a static network cannot represent
a trajectory, and giving it two labels it can only guess at would corrupt the
22 letters it does know. Nothing is gained by pretending otherwise.

  finetune.py --subjects interp1 cisco          # build tiles, fine-tune, report
  finetune.py --subjects interp1 --tiles-only   # just cache the tiles
"""
import argparse
import os
import sys
import time

import numpy as np
import cv2
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import Net, LETTERS, augment
from stream import Engine

RAW = os.path.expanduser("~/leap/asl/data/raw")
CACHE = os.path.expanduser("~/leap/asl/data/tiles")
LETTER_IDX = {c: i for i, c in enumerate(LETTERS)}


def build_tiles(subjects, mode="posed", verbose=True):
    """Captured PGMs -> (N,2,32,32) tiles, labels, subject ids.

    Uses the exact preprocessing the live reader uses, so what the model trains
    on is what it will be shown at runtime. Any mismatch here reintroduces the
    domain gap this script exists to close.
    """
    os.makedirs(CACHE, exist_ok=True)
    X, y, s = [], [], []
    # Borrow the reader's preprocessing without loading the CNN -- the whole point
    # is that tiles are built exactly the way the live path builds them.
    eng = Engine.__new__(Engine)
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    from leap_pose import MODEL_TASK
    opts = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_TASK),
        running_mode=vision.RunningMode.IMAGE, num_hands=1,
        min_hand_detection_confidence=0.1, min_hand_presence_confidence=0.1)
    eng.lm = vision.HandLandmarker.create_from_options(opts)
    eng.lm_r, eng.video, eng._ts = eng.lm, False, 0

    for si, subj in enumerate(subjects):
        root = os.path.join(RAW, subj, mode)
        if not os.path.isdir(root):
            print(f"  ! no capture for {subj} at {root}")
            continue
        for label in sorted(os.listdir(root)):
            if label not in LETTER_IDX:
                continue                    # J and Z live in the motion rule
            d = os.path.join(root, label)
            if not os.path.isdir(d):
                continue
            n0 = len(X)
            ids = sorted({f.split("_")[0] for f in os.listdir(d) if f.endswith("_L.pgm")},
                         key=lambda t: int(t[1:]))
            for fid in ids:
                gl = cv2.imread(os.path.join(d, f"{fid}_L.pgm"), cv2.IMREAD_GRAYSCALE)
                rp = os.path.join(d, f"{fid}_R.pgm")
                gr = cv2.imread(rp, cv2.IMREAD_GRAYSCALE) if os.path.exists(rp) else None
                if gl is None:
                    continue
                pL, rotL = eng._landmarks(gl)
                if pL is None:
                    continue
                tl = eng._silhouette(rotL, pL)
                if tl is None:
                    continue
                tr = tl
                if gr is not None:
                    pR, rotR = eng._landmarks(gr)
                    if pR is not None:
                        t2 = eng._silhouette(rotR, pR)
                        if t2 is not None:
                            tr = t2
                X.append(np.stack([tl, tr]))
                y.append(LETTER_IDX[label])
                s.append(si)
            if verbose:
                print(f"  {subj}/{label}: {len(X)-n0} tiles")
    eng.lm.close()

    if not X:
        sys.exit("no tiles built -- is there any capture data?")
    X = (np.stack(X) * 255).astype(np.uint8)
    y = np.array(y, np.int64)
    s = np.array(s, np.int64)
    np.save(f"{CACHE}/X.npy", X); np.save(f"{CACHE}/y.npy", y); np.save(f"{CACHE}/s.npy", s)
    print(f"\n  cached {X.shape} to {CACHE}")
    return X, y, s


def finetune(X, y, base_model, epochs=12, lr=3e-4, holdout=0.2, seed=0):
    """Low learning rate, whole network. The features transfer; the calibration does not."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    cut = int((1 - holdout) * len(idx))
    tr, te = idx[:cut], idx[cut:]

    Xt = torch.tensor(X[tr], dtype=torch.float32).div_(255.)
    yt = torch.tensor(y[tr])
    Xe = torch.tensor(X[te], dtype=torch.float32).div_(255.).to(dev)
    ye = torch.tensor(y[te]).to(dev)

    net = Net().to(dev)
    net.load_state_dict(torch.load(base_model, map_location=dev))

    # baseline before touching anything -- this is the transfer penalty
    net.eval()
    with torch.no_grad():
        base_acc = (net(Xe).argmax(1) == ye).float().mean().item()
    print(f"\n  before fine-tuning (pure transfer from ASLA-Leap): {base_acc*100:.2f}%")

    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    best = base_acc
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(tr))
        tot = 0.0
        for i in range(0, len(perm), 128):
            b = perm[i:i + 128]
            xb = augment(Xt[b].to(dev))
            loss = F.cross_entropy(net(xb), yt[b].to(dev), label_smoothing=0.05)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(b)
        net.eval()
        with torch.no_grad():
            acc = (net(Xe).argmax(1) == ye).float().mean().item()
        best = max(best, acc)
        print(f"    epoch {ep+1:2d}/{epochs}  loss {tot/len(perm):.4f}  held-out {acc*100:5.2f}%")

    out = os.path.expanduser("~/leap/asl/model_tuned.pt")
    torch.save(net.state_dict(), out)
    print(f"\n  best {best*100:.2f}%  (transfer baseline was {base_acc*100:.2f}%)")
    print(f"  saved {out}")
    return best, base_acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subjects", nargs="+", required=True)
    p.add_argument("--mode", default="posed")
    p.add_argument("--base", default=os.path.expanduser("~/leap/asl/model_loso0.pt"))
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--tiles-only", action="store_true")
    p.add_argument("--use-cache", action="store_true")
    a = p.parse_args()

    if a.use_cache and os.path.exists(f"{CACHE}/X.npy"):
        X, y = np.load(f"{CACHE}/X.npy"), np.load(f"{CACHE}/y.npy")
        print(f"  loaded cached {X.shape}")
    else:
        X, y, _ = build_tiles(a.subjects, a.mode)
    if a.tiles_only:
        return
    finetune(X, y, a.base, epochs=a.epochs)


if __name__ == "__main__":
    main()
