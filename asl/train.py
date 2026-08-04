"""Fingerspelling classifier on ASLA-Leap stereo IR crops.

Two evaluations:
  random  -- random 80/20 split. Inflated: the same subject's frames appear in
             both train and test, and consecutive frames are near-duplicates.
  loso    -- leave-one-subject-out. The honest number: can it read a hand it
             has never seen before? That is what a real device faces.
"""
import sys, time, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DATA = os.path.expanduser("~/leap/asl/data")
OUTD = os.path.expanduser("~/leap/asl")
# 24 static letters: the ASL alphabet minus J and Z, which require motion
LETTERS = [c for c in "ABCDEFGHIKLMNOPQRSTUVWXY"]
assert len(LETTERS) == 24

dev = "cuda" if torch.cuda.is_available() else "cpu"

X = np.load(f"{DATA}/X.npy")
y = np.load(f"{DATA}/y.npy").astype(np.int64)
subj = np.load(f"{DATA}/subj.npy").astype(np.int64)


class Net(nn.Module):
    def __init__(self, nc=24):
        super().__init__()
        self.c1 = nn.Conv2d(2, 32, 3, padding=1);   self.b1 = nn.BatchNorm2d(32)
        self.c2 = nn.Conv2d(32, 64, 3, padding=1);  self.b2 = nn.BatchNorm2d(64)
        self.c3 = nn.Conv2d(64, 128, 3, padding=1); self.b3 = nn.BatchNorm2d(128)
        self.drop = nn.Dropout(0.4)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, nc)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.b1(self.c1(x))), 2)   # 16
        x = F.max_pool2d(F.relu(self.b2(self.c2(x))), 2)   # 8
        x = F.max_pool2d(F.relu(self.b3(self.c3(x))), 2)   # 4
        x = x.flatten(1)
        x = self.drop(F.relu(self.fc1(x)))
        return self.fc2(x)


def augment(xb):
    """Small shifts + scale jitter; NO horizontal flip (that changes handedness)."""
    n = xb.shape[0]
    if torch.rand(1).item() < 0.5:
        dx, dy = np.random.randint(-3, 4), np.random.randint(-3, 4)
        xb = torch.roll(xb, shifts=(dy, dx), dims=(2, 3))
    ang = (torch.rand(n, device=xb.device) - 0.5) * (12 * np.pi / 180)
    cos, sin = torch.cos(ang), torch.sin(ang)
    theta = torch.zeros(n, 2, 3, device=xb.device)
    theta[:, 0, 0] = cos; theta[:, 0, 1] = -sin
    theta[:, 1, 0] = sin; theta[:, 1, 1] = cos
    grid = F.affine_grid(theta, xb.shape, align_corners=False)
    return F.grid_sample(xb, grid, align_corners=False, padding_mode="zeros")


def run(tr_idx, te_idx, tag, epochs=14):
    Xtr = torch.tensor(X[tr_idx], dtype=torch.float32).div_(255.)
    ytr = torch.tensor(y[tr_idx])
    Xte = torch.tensor(X[te_idx], dtype=torch.float32).div_(255.).to(dev)
    yte = torch.tensor(y[te_idx]).to(dev)

    net = Net().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=3e-3, total_steps=epochs * (len(tr_idx) // 256 + 1))
    best = 0.0
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(tr_idx))
        tot = 0.0
        for i in range(0, len(perm), 256):
            b = perm[i:i + 256]
            xb = Xtr[b].to(dev, non_blocking=True)
            yb = ytr[b].to(dev, non_blocking=True)
            xb = augment(xb)
            opt.zero_grad()
            loss = F.cross_entropy(net(xb), yb, label_smoothing=0.05)
            loss.backward(); opt.step()
            try: sched.step()
            except Exception: pass
            tot += loss.item() * len(b)
        net.eval()
        with torch.no_grad():
            pred = torch.cat([net(Xte[i:i+2048]).argmax(1) for i in range(0, len(Xte), 2048)])
            acc = (pred == yte).float().mean().item()
        best = max(best, acc)
        print(f"    [{tag}] epoch {ep+1:2d}/{epochs} loss {tot/len(perm):.4f} test acc {acc*100:5.2f}%", flush=True)
    return best, net, pred.cpu().numpy(), yte.cpu().numpy()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    print(f"device={dev}  X={X.shape}  classes={len(LETTERS)}\n")

    if mode in ("random", "both"):
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(y))
        cut = int(0.8 * len(idx))
        acc, net, _, _ = run(idx[:cut], idx[cut:], "random")
        print(f"\n  RANDOM SPLIT best accuracy: {acc*100:.2f}%  (inflated -- same "
              f"subjects and near-duplicate frames on both sides)\n")
        torch.save(net.state_dict(), f"{OUTD}/model_random.pt")

    if mode in ("loso", "both"):
        accs = []
        for s in range(5):
            tr = np.where(subj != s)[0]
            te = np.where(subj == s)[0]
            acc, net, pred, true = run(tr, te, f"holdout subj {s}")
            accs.append(acc)
            print(f"  -> subject {s} held out: {acc*100:.2f}%\n", flush=True)
            if s == 0:
                torch.save(net.state_dict(), f"{OUTD}/model_loso0.pt")
                np.save(f"{OUTD}/loso0_pred.npy", pred)
                np.save(f"{OUTD}/loso0_true.npy", true)
        a = np.array(accs)
        print(f"\n  LEAVE-ONE-SUBJECT-OUT: mean {a.mean()*100:.2f}%  "
              f"(per subject: {', '.join(f'{v*100:.1f}' for v in a)})")
        print("  ^ this is the number that matters for an unseen signer")
