"""Live bridge: Leap stereo IR frame -> MediaPipe hand box -> ASLA-Leap 32x32 -> letter.

The dataset's samples are tight, contrast-normalized hand crops. Our raw frames are
wide views with a body in them, so MediaPipe supplies the hand box that the original
capture rig got from close-range depth segmentation.

Usage: live_infer.py <model.pt> <frame_L.pgm> [frame_R.pgm]
"""
import sys, os
import numpy as np, cv2, torch
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import Net, LETTERS

SCRATCH = "/tmp/claude-1000/-home-cisco/ed237c20-5857-4892-8b42-ac36a8bc4c81/scratchpad"
MODEL_TASK = os.path.join(SCRATCH, "hand_landmarker.task")


def prep_for_mp(g):
    c = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(g)
    up = cv2.resize(c, (640, 480), interpolation=cv2.INTER_CUBIC)
    return cv2.rotate(up, cv2.ROTATE_180)


def hand_box(g, conf=0.1, pad=0.35):
    """Return (x0,y0,x1,y1) in ORIGINAL pixel coords, or None."""
    rgb = np.ascontiguousarray(cv2.cvtColor(prep_for_mp(g), cv2.COLOR_GRAY2RGB))
    opts = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_TASK),
        running_mode=vision.RunningMode.IMAGE, num_hands=1,
        min_hand_detection_confidence=conf, min_hand_presence_confidence=conf)
    with vision.HandLandmarker.create_from_options(opts) as lm:
        res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not res.hand_landmarks:
        return None
    pts = np.array([[1.0 - p.x, 1.0 - p.y] for p in res.hand_landmarks[0]])  # undo rot180
    h, w = g.shape[:2]
    xs, ys = pts[:, 0] * w, pts[:, 1] * h
    cx, cy = xs.mean(), ys.mean()
    half = max(xs.max() - xs.min(), (ys.max() - ys.min())) * (0.5 + pad)
    half = max(half, 12)
    return (int(max(0, cx - half)), int(max(0, cy - half)),
            int(min(w, cx + half)), int(min(h, cy + half)))


def to_tile(g, box):
    """Crop -> contrast normalize -> square -> 32x32, matching the dataset's look."""
    x0, y0, x1, y1 = box
    crop = g[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    crop = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4)).apply(crop)
    # dataset crops are hand-bright-on-black; suppress the dim background
    crop = crop.astype(np.float32)
    lo, hi = np.percentile(crop, 45), crop.max()
    crop = np.clip((crop - lo) / max(hi - lo, 1e-3), 0, 1)
    h, w = crop.shape
    s = max(h, w)
    sq = np.zeros((s, s), np.float32)
    sq[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = crop
    # our frames are 2:1 squashed horizontally -- undo before resizing
    sq = cv2.resize(sq, (s, s // 2 if False else s), interpolation=cv2.INTER_AREA)
    return cv2.resize(sq, (32, 32), interpolation=cv2.INTER_AREA)


def main():
    model_path, lpath = sys.argv[1], sys.argv[2]
    rpath = sys.argv[3] if len(sys.argv) > 3 else lpath.replace("_L.pgm", "_R.pgm")

    gl = cv2.imread(lpath, cv2.IMREAD_GRAYSCALE)
    gr = cv2.imread(rpath, cv2.IMREAD_GRAYSCALE)
    if gl is None:
        sys.exit("no left image")
    # stretch the 2:1 squash so the hand has human proportions, like the dataset
    gl = cv2.resize(gl, (640, 480), interpolation=cv2.INTER_CUBIC)
    gr = cv2.resize(gr, (640, 480), interpolation=cv2.INTER_CUBIC) if gr is not None else gl

    bl = hand_box(cv2.resize(gl, (640, 240), interpolation=cv2.INTER_AREA))
    if bl is None:
        print(f"{os.path.basename(lpath)}: no hand found")
        return
    bl = (bl[0], bl[1] * 2, bl[2], bl[3] * 2)      # box was found on the 640x240 view
    br = bl

    tl, tr = to_tile(gl, bl), to_tile(gr, br)
    if tl is None:
        print("crop failed"); return

    x = torch.tensor(np.stack([tl, tr])[None], dtype=torch.float32)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = Net().to(dev)
    net.load_state_dict(torch.load(model_path, map_location=dev))
    net.eval()
    with torch.no_grad():
        p = torch.softmax(net(x.to(dev)), 1)[0].cpu().numpy()
    top = p.argsort()[::-1][:3]
    print(f"{os.path.basename(lpath)}: " +
          "  ".join(f"{LETTERS[i]}={p[i]*100:.1f}%" for i in top))
    # save the tile so the preprocessing can be eyeballed against the dataset
    out = os.path.expanduser("~/leap/shots/tile_" + os.path.basename(lpath).replace(".pgm", ".png"))
    cv2.imwrite(out, np.hstack([tl, tr]) * 255)
    return out


if __name__ == "__main__":
    main()
