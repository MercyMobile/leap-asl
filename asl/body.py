"""Body and face from the webcam, fused with hands from the Leap.

An avatar cannot be driven by a hand. Where a sign is made carries meaning, and
ASL grammar -- question, negation, intensity, topic -- lives on the face, not the
hands. So a signing avatar needs body and face or it is not signing, it is
gesturing.

The Leap cannot supply either. It sees a small volume just above itself. A plain
webcam sees the whole person, and MediaPipe reads body and face from it well.
So each sensor does what it is actually good at:

    webcam  -> 33 body landmarks + 478 face landmarks
    Leap    -> 21 hand landmarks, with true metric depth from two cameras

MediaPipe 1.0 removed the combined Holistic model along with `mp.solutions`, so
this composes the equivalent from the separate Pose and Face landmarkers and
emits the same component names. Downstream tooling cannot tell the difference,
and we get to choose better models per part.

    body.py --check                 # what can the webcam see right now
    body.py --out session.pose      # record body+face to a .pose
"""
import argparse
import os
import sys
import time

import numpy as np
import cv2

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODELS = os.path.expanduser("~/leap/models")
POSE_TASK = os.path.join(MODELS, "pose_landmarker_full.task")
FACE_TASK = os.path.join(MODELS, "face_landmarker.task")

# Landmarks that matter for signing space: where the hands sit relative to these
# is what "location" means as an ASL parameter.
KEY = {11: "L shoulder", 12: "R shoulder", 13: "L elbow", 14: "R elbow",
       15: "L wrist", 16: "R wrist", 0: "nose"}


def open_cam(dev=0, w=1280, h=720):
    cap = cv2.VideoCapture(dev)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    if not cap.isOpened():
        sys.exit(f"could not open /dev/video{dev}")
    return cap


def make_pose(video=True):
    return vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=POSE_TASK),
            running_mode=vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE,
            num_poses=1, min_pose_detection_confidence=0.4))


def make_face(video=True):
    return vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=FACE_TASK),
            running_mode=vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE,
            num_faces=1, output_face_blendshapes=True,
            min_face_detection_confidence=0.4))


def read(frame, pose_lm, face_lm, ts):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    pr = pose_lm.detect_for_video(img, ts)
    fr = face_lm.detect_for_video(img, ts)
    body = None
    if pr.pose_landmarks:
        body = np.array([[p.x, p.y, p.z, p.visibility] for p in pr.pose_landmarks[0]])
    face = None
    if fr.face_landmarks:
        face = np.array([[p.x, p.y, p.z] for p in fr.face_landmarks[0]])
    blend = None
    if getattr(fr, "face_blendshapes", None):
        blend = {b.category_name: b.score for b in fr.face_blendshapes[0]}
    return body, face, blend


def check(dev, secs=4.0):
    """Look through the webcam and report what is actually visible."""
    cap = open_cam(dev)
    pose_lm, face_lm = make_pose(), make_face()
    t0 = time.time()
    n = nb = nf = 0
    last = (None, None, None)
    while time.time() - t0 < secs:
        ok, frame = cap.read()
        if not ok:
            continue
        b, f, bl = read(frame, pose_lm, face_lm, int((time.time() - t0) * 1000) + n)
        n += 1
        nb += b is not None
        nf += f is not None
        if b is not None:
            last = (b, f, bl)
    cap.release(); pose_lm.close(); face_lm.close()

    print(f"  {n} frames in {secs:.0f}s ({n/secs:.0f} fps)")
    print(f"  body found  {nb}/{n}   face found  {nf}/{n}")
    b, f, bl = last
    if b is None:
        print("\n  No body detected. Step back so your head and shoulders are in frame.")
        return
    print("\n  key landmarks (x, y normalized; visibility):")
    for i, name in KEY.items():
        print(f"    {name:<12} ({b[i][0]:.2f}, {b[i][1]:.2f})   vis {b[i][3]:.2f}")
    sw = np.linalg.norm(b[11][:2] - b[12][:2])
    print(f"\n  shoulder width {sw:.3f} of frame width "
          f"-- this is the scale reference the standard pose normalizer wants")
    if f is not None:
        print(f"  face: {len(f)} landmarks")
    if bl:
        top = sorted(bl.items(), key=lambda kv: -kv[1])[:5]
        print("  strongest expressions: " + ", ".join(f"{k} {v:.2f}" for k, v in top))
        print("  ^ these are the non-manual markers ASL grammar rides on")


# ---------------------------------------------------------------- .pose out

def holistic_header(width, height, n_face):
    """Full holistic schema: body, face, both hands. Built by hand because
    pose_format.utils.holistic imports mp.solutions and will not load against
    MediaPipe 1.0."""
    from pose_format.pose_header import (PoseHeader, PoseHeaderComponent,
                                         PoseHeaderDimensions)
    from leap_pose import HAND_POINTS, HAND_LIMBS

    body_points = [
        "NOSE", "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER", "RIGHT_EYE_INNER",
        "RIGHT_EYE", "RIGHT_EYE_OUTER", "LEFT_EAR", "RIGHT_EAR", "MOUTH_LEFT",
        "MOUTH_RIGHT", "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW",
        "LEFT_WRIST", "RIGHT_WRIST", "LEFT_PINKY", "RIGHT_PINKY", "LEFT_INDEX",
        "RIGHT_INDEX", "LEFT_THUMB", "RIGHT_THUMB", "LEFT_HIP", "RIGHT_HIP",
        "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_HEEL",
        "RIGHT_HEEL", "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX",
    ]
    hc = [(0, 0, 255)] * len(HAND_LIMBS)
    hand = lambda n: PoseHeaderComponent(name=n, points=HAND_POINTS, limbs=HAND_LIMBS,
                                         colors=hc, point_format="XYZC")
    comps = [
        PoseHeaderComponent(name="POSE_LANDMARKS", points=body_points, limbs=[],
                            colors=[(255, 0, 0)], point_format="XYZC"),
        PoseHeaderComponent(name="FACE_LANDMARKS", points=[str(i) for i in range(n_face)],
                            limbs=[], colors=[(128, 0, 0)], point_format="XYZC"),
        hand("LEFT_HAND_LANDMARKS"),
        hand("RIGHT_HAND_LANDMARKS"),
    ]
    return PoseHeader(version=0.2,
                      dimensions=PoseHeaderDimensions(width=width, height=height, depth=1000),
                      components=comps)


def record(dev, secs, out, n_face=478):
    from pose_format import Pose
    from pose_format.numpy import NumPyPoseBody

    cap = open_cam(dev)
    pose_lm, face_lm = make_pose(), make_face()
    frames = []
    t0 = time.time(); n = 0
    print(f"  recording {secs:.0f}s from /dev/video{dev} ...")
    while time.time() - t0 < secs:
        ok, frame = cap.read()
        if not ok:
            continue
        b, f, _ = read(frame, pose_lm, face_lm, int((time.time() - t0) * 1000) + n)
        frames.append((b, f)); n += 1
    h, w = frame.shape[:2]
    cap.release(); pose_lm.close(); face_lm.close()

    header = holistic_header(w, h, n_face)
    total = header.total_points()
    T = len(frames)
    data = np.zeros((T, 1, total, 3), np.float32)
    conf = np.zeros((T, 1, total), np.float32)
    off = 0; span = {}
    for c in header.components:
        span[c.name] = (off, off + len(c.points)); off += len(c.points)

    bl, bh = span["POSE_LANDMARKS"]; fl, fh = span["FACE_LANDMARKS"]
    for t, (b, f) in enumerate(frames):
        if b is not None:
            data[t, 0, bl:bh] = b[:, :3] * [w, h, w]
            conf[t, 0, bl:bh] = b[:, 3]
        if f is not None:
            k = min(len(f), fh - fl)
            data[t, 0, fl:fl + k] = f[:k] * [w, h, w]
            conf[t, 0, fl:fl + k] = 1.0

    pose = Pose(header, NumPyPoseBody(fps=T / secs, data=data, confidence=conf))
    with open(out, "wb") as fh_:
        pose.write(fh_)
    got_b = sum(1 for b, _ in frames if b is not None)
    got_f = sum(1 for _, f in frames if f is not None)
    print(f"  {T} frames, body {got_b}, face {got_f}  ->  {out} ({os.path.getsize(out)} bytes)")
    print(f"  hands are left empty here -- the Leap fills those, see leap_pose.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--secs", type=float, default=4.0)
    ap.add_argument("--out")
    a = ap.parse_args()
    for m in (POSE_TASK, FACE_TASK):
        if not os.path.exists(m):
            sys.exit(f"missing model {m}")
    if a.out:
        record(a.device, a.secs, a.out)
    else:
        check(a.device, a.secs)


if __name__ == "__main__":
    main()
