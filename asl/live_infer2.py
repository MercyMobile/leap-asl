"""Leap stereo IR -> ASLA-Leap-style silhouette -> letter.

v2 closes the domain gap found in v1. The dataset's tiles are clean white
hand silhouettes on black, upright and tightly cropped -- produced by close-range
depth segmentation. Our frames are wide, textured, arbitrary-angle views. So:

  1. MediaPipe landmarks give the hand.
  2. Rotate so wrist -> middle-finger-MCP points up  (matches dataset orientation).
  3. Mask to the dilated convex hull of the landmarks (kills arm + background).
  4. Normalize to bright-hand-on-black, tight-crop, 32x32.

Usage: live_infer2.py <model.pt> <frame_L.pgm> [--save]
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
WRIST, MID_MCP = 0, 9


def landmarks(g, conf=0.1):
    """21 landmarks in pixel coords of the aspect-corrected (640x480) image."""
    c = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(g)
    up = cv2.resize(c, (640, 480), interpolation=cv2.INTER_CUBIC)
    rot = cv2.rotate(up, cv2.ROTATE_180)
    rgb = np.ascontiguousarray(cv2.cvtColor(rot, cv2.COLOR_GRAY2RGB))
    opts = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_TASK),
        running_mode=vision.RunningMode.IMAGE, num_hands=1,
        min_hand_detection_confidence=conf, min_hand_presence_confidence=conf)
    with vision.HandLandmarker.create_from_options(opts) as lm:
        res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not res.hand_landmarks:
        return None, None
    pts = np.array([[p.x * 640, p.y * 480] for p in res.hand_landmarks[0]], np.float32)
    return pts, rot          # landmarks are in the ROTATED image's frame


def silhouette(img, pts):
    """Upright, masked, bright-on-black, tight 32x32 tile."""
    # --- rotate so the hand points up
    v = pts[MID_MCP] - pts[WRIST]
    ang = np.degrees(np.arctan2(v[0], -v[1]))          # 0 == already pointing up
    c = (float(pts[WRIST][0]), float(pts[WRIST][1]))
    M = cv2.getRotationMatrix2D(c, -ang, 1.0)
    rimg = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    rp = (M[:, :2] @ pts.T).T + M[:, 2]

    # --- convex-hull mask, dilated to include finger thickness
    mask = np.zeros(rimg.shape[:2], np.uint8)
    hull = cv2.convexHull(rp.astype(np.int32))
    cv2.fillConvexPoly(mask, hull, 255)
    span = np.linalg.norm(rp.max(0) - rp.min(0))
    k = max(3, int(span * 0.16) | 1)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))

    # --- bright hand on black
    f = rimg.astype(np.float32)
    inside = f[mask > 0]
    if inside.size < 50:
        return None
    lo, hi = np.percentile(inside, 25), np.percentile(inside, 99)
    f = np.clip((f - lo) / max(hi - lo, 1e-3), 0, 1)
    f[mask == 0] = 0.0

    ys, xs = np.where(mask > 0)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = f[y0:y1, x0:x1]
    h, w = crop.shape
    s = max(h, w)
    sq = np.zeros((s, s), np.float32)
    sq[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = crop
    return cv2.resize(sq, (32, 32), interpolation=cv2.INTER_AREA)


def main():
    model_path, lpath = sys.argv[1], sys.argv[2]
    rpath = lpath.replace("_L.pgm", "_R.pgm")
    gl = cv2.imread(lpath, cv2.IMREAD_GRAYSCALE)
    gr = cv2.imread(rpath, cv2.IMREAD_GRAYSCALE)
    if gl is None:
        sys.exit("no left image")

    pl, rl = landmarks(gl)
    if pl is None:
        print(f"{os.path.basename(lpath)}: no hand"); return
    tl = silhouette(rl, pl)
    pr, rr = landmarks(gr) if gr is not None else (None, None)
    tr = silhouette(rr, pr) if pr is not None else tl
    if tl is None or tr is None:
        print(f"{os.path.basename(lpath)}: mask failed"); return

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
    if "--save" in sys.argv:
        out = os.path.expanduser("~/leap/shots/sil_" +
                                 os.path.basename(lpath).replace(".pgm", ".png"))
        cv2.imwrite(out, (np.hstack([tl, tr]) * 255).astype(np.uint8))


if __name__ == "__main__":
    main()
