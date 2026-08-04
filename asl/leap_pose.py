"""Leap stereo IR  ->  metric 3D hand landmarks  ->  .pose (pose-format holistic schema).

Why this file exists
--------------------
The open ASL toolchain (sign-language-processing/spoken-to-signed-translation,
sign.mt) is lexicon-driven: text -> gloss -> *pose sequence* -> avatar. The pose
sequences it stitches are `.pose` files in the `pose_format` container, and the
schema everything speaks is MediaPipe Holistic.

So the bridge from our rig to that world is exactly this: emit .pose files whose
header declares the holistic components. Then our Leap recordings are lexicon
entries, and their renderer/retargeter consumes them unmodified.

One real difference, and it is the interesting one
--------------------------------------------------
MediaPipe Holistic's `z` is *pseudo*-depth: roughly image-normalized, relative to
the wrist, with no metric meaning and no stable scale between frames. Ours is
triangulated from two physical cameras with a measured 60mm baseline, so it is
true millimetres. Same container, strictly better contents. We record that in the
header's `depth` dimension (mm) so a consumer can tell.

Geometry
--------
The Leap ships a 64x64 distortion grid per eye (`LEAP_IMAGE.distortion_matrix`)
mapping ray slope, spanning [-4,4] in both axes, to normalized image coordinate.
We need the other direction, so we invert it numerically (scipy griddata).
Then, for a landmark seen in both eyes:

    z = baseline / (slope_x_L - slope_x_R)
    x = slope_x_L * z        y = slope_y_L * z      (origin at the left camera)

An epipolar gate is mandatory: MediaPipe is matching independently in each eye and
will happily lock onto different objects. Rectified cameras must agree in y, so
|slope_y_L - slope_y_R| > EPI_TOL means the correspondence is bogus. Without this
gate reconstructed hand spans blow out to 420mm.
"""
import os
import sys

import numpy as np
import cv2
from scipy.interpolate import griddata

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_TASK = os.path.expanduser("~/leap/models/hand_landmarker.task")

N = 64                       # distortion grid is 64x64
SLOPE_MIN, SLOPE_MAX = -4.0, 4.0
BASELINE_MM = 60.0           # MEASURED on this unit, not the 40mm commonly quoted
EPI_TOL = 0.15               # max |slope_y_L - slope_y_R| for a trusted match

# Rigid-bone gate. The knuckle row (index MCP -> pinky MCP) is metacarpal bone: it
# reads the same whether the hand is a fist or flat, so a correct reconstruction
# must return a constant. Adult hand breadth at the metacarpals is ~80-95mm.
#
# This is a FILTER, not a validation of accuracy -- it discards frames using known
# anatomy, so it cannot then be cited as evidence that the anatomy came out right.
# What it does honestly show is *self-consistency*: surviving frames agree with
# each other to ~8%, which per-landmark epipolar gating alone does not deliver.
IDX_MCP, PINKY_MCP = 5, 17
KNUCKLE_MIN_MM, KNUCKLE_MAX_MM = 72.0, 105.0

# MediaPipe hand landmark names, in index order. Hardcoded because mediapipe 1.0
# removed `mp.solutions`, which is where pose_format reads them from -- so
# pose_format.utils.holistic cannot even be imported against a current mediapipe.
HAND_POINTS = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
]
HAND_LIMBS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]


# ---------------------------------------------------------------- distortion

def load_distortion(path):
    """64x64x2 grid: [row, col] -> (x, y) normalized image coord."""
    vals = []
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            vals.extend(float(v) for v in line.split())
    a = np.array(vals, np.float64)
    assert a.size == N * N * 2, f"{path}: expected {N*N*2} values, got {a.size}"
    return a.reshape(N, N, 2)


def build_inverse(grid):
    """Return f(pixel_uv_normalized) -> ray slope (sx, sy), by inverting the grid.

    The forward grid is slope -> image coord. We hand griddata the forward pairs
    with the roles swapped, which gives a scattered interpolant in the direction
    we need.
    """
    us = np.linspace(SLOPE_MIN, SLOPE_MAX, N)
    sx, sy = np.meshgrid(us, us)                       # slope samples
    img_pts = grid.reshape(-1, 2)                      # where each slope lands
    slopes = np.stack([sx.ravel(), sy.ravel()], axis=1)

    # Drop grid entries that fall far outside the sensor; they are extrapolation
    # garbage and they drag the triangulation around.
    keep = ((img_pts[:, 0] > -0.5) & (img_pts[:, 0] < 1.5) &
            (img_pts[:, 1] > -0.5) & (img_pts[:, 1] < 1.5))
    img_pts, slopes = img_pts[keep], slopes[keep]

    def f(uv):
        return griddata(img_pts, slopes, uv, method="linear")

    return f


# ---------------------------------------------------------------- mediapipe

def _landmarker(conf=0.1):
    opts = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_TASK),
        running_mode=vision.RunningMode.IMAGE, num_hands=1,
        min_hand_detection_confidence=conf, min_hand_presence_confidence=conf)
    return vision.HandLandmarker.create_from_options(opts)


def detect(gray, lm):
    """21 landmarks as normalized coords in the RAW frame, plus a score.

    The preprocessing here is not optional and was found empirically: rotate 180
    (raw Leap frames are upside down), CLAHE for local contrast, and stretch
    640x240 -> 640x480 to undo the 2:1 vertical squash. Six other variants failed
    outright; the aspect correction is the one that unlocks it.
    """
    c = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
    up = cv2.resize(c, (640, 480), interpolation=cv2.INTER_CUBIC)
    rot = cv2.rotate(up, cv2.ROTATE_180)
    rgb = np.ascontiguousarray(cv2.cvtColor(rot, cv2.COLOR_GRAY2RGB))
    res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not res.hand_landmarks:
        return None, 0.0
    # undo the 180 rotation to get back into raw-frame normalized coords
    pts = np.array([[1.0 - p.x, 1.0 - p.y] for p in res.hand_landmarks[0]], np.float64)
    score = float(res.handedness[0][0].score) if res.handedness else 1.0
    return pts, score


# ---------------------------------------------------------------- triangulate

def triangulate(ptsL, ptsR, invL, invR):
    """(21,3) metric mm in the left-camera frame, and a per-point valid mask."""
    sL = invL(ptsL)
    sR = invR(ptsR)

    dy = np.abs(sL[:, 1] - sR[:, 1])
    disp = sL[:, 0] - sR[:, 0]
    ok = (np.isfinite(dy) & np.isfinite(disp) & (dy < EPI_TOL) & (np.abs(disp) > 1e-6))

    z = np.full(len(ptsL), np.nan)
    z[ok] = BASELINE_MM / disp[ok]
    ok &= np.isfinite(z) & (z > 0)          # behind the camera is not a hand

    xyz = np.stack([sL[:, 0] * z, sL[:, 1] * z, z], axis=1)
    xyz[~ok] = 0.0
    return xyz, ok


# ---------------------------------------------------------------- .pose output

def holistic_header(width, height):
    """A pose_format header declaring the MediaPipe Holistic schema.

    We build it by hand rather than calling pose_format.utils.holistic, which
    imports mp.solutions and therefore cannot load against mediapipe >= 1.0.
    Body and face components are declared but left empty -- consumers detect the
    format by component name, and downstream code selects the components it wants.
    """
    from pose_format.pose_header import (PoseHeader, PoseHeaderComponent,
                                         PoseHeaderDimensions)

    body_points = [
        "NOSE", "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER", "RIGHT_EYE_INNER",
        "RIGHT_EYE", "RIGHT_EYE_OUTER", "LEFT_EAR", "RIGHT_EAR", "MOUTH_LEFT",
        "MOUTH_RIGHT", "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW",
        "LEFT_WRIST", "RIGHT_WRIST", "LEFT_PINKY", "RIGHT_PINKY", "LEFT_INDEX",
        "RIGHT_INDEX", "LEFT_THUMB", "RIGHT_THUMB", "LEFT_HIP", "RIGHT_HIP",
        "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_HEEL",
        "RIGHT_HEEL", "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX",
    ]
    hand_colors = [(0, 0, 255)] * len(HAND_LIMBS)

    def hand(name):
        return PoseHeaderComponent(name=name, points=HAND_POINTS, limbs=HAND_LIMBS,
                                   colors=hand_colors, point_format="XYZC")

    components = [
        PoseHeaderComponent(name="POSE_LANDMARKS", points=body_points, limbs=[],
                            colors=[(255, 0, 0)], point_format="XYZC"),
        hand("LEFT_HAND_LANDMARKS"),
        hand("RIGHT_HAND_LANDMARKS"),
    ]
    # depth is in millimetres here -- ours is real metric depth, unlike holistic's
    dims = PoseHeaderDimensions(width=width, height=height, depth=1000)
    return PoseHeader(version=0.2, dimensions=dims, components=components)


def write_pose(path, frames_xyz, frames_ok, fps, side="RIGHT", width=640, height=240):
    """frames_xyz: (T,21,3) mm.  frames_ok: (T,21) bool."""
    from pose_format import Pose
    from pose_format.numpy import NumPyPoseBody

    header = holistic_header(width, height)
    total = header.total_points()
    T = len(frames_xyz)

    data = np.zeros((T, 1, total, 3), np.float32)
    conf = np.zeros((T, 1, total), np.float32)

    # component order in the flat point array follows header component order
    offset = 0
    spans = {}
    for c in header.components:
        spans[c.name] = (offset, offset + len(c.points))
        offset += len(c.points)

    lo, hi = spans[f"{side}_HAND_LANDMARKS"]
    data[:, 0, lo:hi, :] = frames_xyz
    conf[:, 0, lo:hi] = frames_ok.astype(np.float32)

    body = NumPyPoseBody(fps=fps, data=data, confidence=conf)
    pose = Pose(header, body)
    with open(path, "wb") as f:
        pose.write(f)
    return pose


def normalize_knuckles(pose, side="RIGHT"):
    """Scale-normalize a Leap pose, anchored on the rigid knuckle row.

    The standard holistic normalization (`pose_normalization_info`) anchors on
    shoulder landmarks 11/12 in POSE_LANDMARKS. We have no body camera, so those
    are zeros -- and the normalizer does NOT raise on a zero-length anchor, it
    silently returns finite nonsense. Anything downstream that calls the default
    path on a Leap-only .pose is quietly wrong.

    So anchor inside the hand instead, on the metacarpal row, which is rigid bone
    and therefore a fixed reference regardless of handshape. Residual scatter
    after this is triangulation noise (~8% on our recordings).
    """
    from pose_format.pose_header import PoseNormalizationInfo

    offset = 0
    for c in pose.header.components:
        if c.name == f"{side}_HAND_LANDMARKS":
            break
        offset += len(c.points)
    else:
        raise ValueError(f"no {side}_HAND_LANDMARKS component")

    pose.normalize(PoseNormalizationInfo(p1=offset + IDX_MCP, p2=offset + PINKY_MCP))
    return pose


# ---------------------------------------------------------------- driver

def convert_dir(recdir, out_pose=None, fps=115.0, side="RIGHT", verbose=True):
    invL = build_inverse(load_distortion(os.path.join(recdir, "distortion_L.txt")))
    invR = build_inverse(load_distortion(os.path.join(recdir, "distortion_R.txt")))

    ids = sorted({f.split("_")[0] for f in os.listdir(recdir) if f.endswith("_L.pgm")},
                 key=lambda s: int(s[1:]))

    xyzs, oks, kept = [], [], []
    with _landmarker() as lm:
        for fid in ids:
            gl = cv2.imread(os.path.join(recdir, f"{fid}_L.pgm"), cv2.IMREAD_GRAYSCALE)
            gr = cv2.imread(os.path.join(recdir, f"{fid}_R.pgm"), cv2.IMREAD_GRAYSCALE)
            if gl is None or gr is None:
                continue
            pL, _ = detect(gl, lm)
            pR, _ = detect(gr, lm)
            if pL is None or pR is None:
                if verbose:
                    print(f"  {fid}: hand in both eyes? no")
                continue
            xyz, ok = triangulate(pL, pR, invL, invR)
            if ok.sum() < 8:
                if verbose:
                    print(f"  {fid}: only {ok.sum()}/21 survived the epipolar gate -- dropped")
                continue
            knuckle = (np.linalg.norm(xyz[IDX_MCP] - xyz[PINKY_MCP])
                       if ok[IDX_MCP] and ok[PINKY_MCP] else float("nan"))
            if not (KNUCKLE_MIN_MM <= knuckle <= KNUCKLE_MAX_MM):
                if verbose:
                    print(f"  {fid}: knuckle row {knuckle:.1f}mm is not a hand -- dropped")
                continue
            xyzs.append(xyz)
            oks.append(ok)
            kept.append(fid)
            if verbose:
                z = xyz[ok][:, 2]
                span = np.linalg.norm(xyz[12] - xyz[0]) if ok[12] and ok[0] else float("nan")
                print(f"  {fid}: {ok.sum():2d}/21 pts  depth {np.median(z):6.1f}mm  "
                      f"wrist->mid-tip {span:5.1f}mm")

    if not xyzs:
        print("no usable frames")
        return None

    X = np.stack(xyzs).astype(np.float32)
    O = np.stack(oks)
    if out_pose:
        write_pose(out_pose, X, O, fps=fps, side=side)
    return X, O, kept


if __name__ == "__main__":
    recdir = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/leap/stereo")
    out = sys.argv[2] if len(sys.argv) > 2 else None
    r = convert_dir(recdir, out)
    if r:
        X, O, kept = r
        z = X[..., 2][O]
        print(f"\n{len(kept)} frames  |  median depth {np.median(z):.1f}mm  "
              f"| {O.sum()}/{O.size} landmarks passed the epipolar gate")
        if out:
            print(f"wrote {out} ({os.path.getsize(out)} bytes)")
