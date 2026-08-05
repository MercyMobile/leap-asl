"""Direct camera access to the Leap, with manual exposure, gain and LEDs.

Why bypass the service
----------------------
The factory spec is 25mm-600mm of tracking range. We measured detection dying
past about 250mm -- under half. The cause is not the sensor: `leapd` runs
auto-exposure tuned for the product's real use case, a hand hovering six inches
over a desk puck for VR. That is the wrong target for a person signing.

We already ignore leapd's tracker (it found hands in 2.8% of signing frames
against MediaPipe's 30.6%) and we already have the distortion grids saved to
disk, so the only thing leapd still provides is raw frames -- which the camera
will hand over directly, with settings we choose.

The control mapping, from Leap's own leapuvc example
----------------------------------------------------
Not guessable. Standard UVC properties are reused for unrelated purposes:

    exposure (microseconds, 10-32222)  ->  CAP_PROP_ZOOM
    LEDs                               ->  CAP_PROP_CONTRAST, encoded
                                           (selector | value<<6), sel 2/3/4
                                           for left/centre/right
    analog gain (16-63)                ->  CAP_PROP_GAIN
    digital gain (0-16)                ->  CAP_PROP_BRIGHTNESS

The LEDs are why a plain `ffmpeg` capture hangs forever: nothing turns them on,
so the sensor sits in the dark and never produces a usable frame.

Left and right eyes arrive interleaved by COLUMN in a single frame:
frame[:, ::2] is left, frame[:, 1::2] is right.

    leapcam.py probe            # what settings does this device report
    leapcam.py sweep            # measure detection against exposure and LED
    leapcam.py grab out.png     # one stereo pair at chosen settings

The service is stopped while this runs and ALWAYS restarted afterwards, even on
crash or Ctrl-C -- a dark Leap is a broken Leap from the user's side.
"""
import argparse
import atexit
import os
import subprocess
import sys
import time

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SERVICE = "ultraleap-hand-tracking-service"
DEV = "/dev/video4"

# OpenCV property aliases for what they actually control on this device
P_EXPOSURE = cv2.CAP_PROP_ZOOM
P_LED = cv2.CAP_PROP_CONTRAST
P_AGAIN = cv2.CAP_PROP_GAIN
P_DGAIN = cv2.CAP_PROP_BRIGHTNESS

EXPOSURE_MAX = 32222
_service_was_running = False


def _svc(action):
    subprocess.run(["sudo", "-n", "systemctl", action, SERVICE],
                   capture_output=True, timeout=60)


def _restore():
    global _service_was_running
    if _service_was_running:
        _svc("start")
        _service_was_running = False


def take_device():
    """Stop the service so the camera node appears. Restoration is registered
    immediately, before anything can fail."""
    global _service_was_running
    r = subprocess.run(["systemctl", "is-active", SERVICE], capture_output=True, text=True)
    if r.stdout.strip() == "active":
        _service_was_running = True
        atexit.register(_restore)
        _svc("stop")
        for _ in range(20):
            time.sleep(0.25)
            if os.path.exists(DEV):
                break
    if not os.path.exists(DEV):
        _restore()
        sys.exit(f"{DEV} never appeared after stopping {SERVICE}")


def open_leap(exposure=None, leds=True, again=None, dgain=None, width=640, height=480):
    cap = cv2.VideoCapture(DEV, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"could not open {DEV}")
    # The device speaks YUYV only. Ask for it explicitly and BEFORE the size --
    # an ffmpeg attempt with -input_format gray hung forever waiting on a stream
    # that never started.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y', 'U', 'Y', 'V'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
    set_leds(cap, leds)
    if exposure is not None:
        cap.set(P_EXPOSURE, max(10, min(EXPOSURE_MAX, int(exposure))))
    if again is not None:
        cap.set(P_AGAIN, max(16, min(63, int(again))))
    if dgain is not None:
        cap.set(P_DGAIN, max(0, min(16, int(dgain))))
    return cap


def set_leds(cap, on):
    """Left, centre and right are separate; the value is packed into contrast."""
    v = 1 if on else 0
    for sel in (2, 3, 4):
        cap.set(P_LED, float(sel | (v << 6)))


def split(frame):
    """Columns are interleaved: even = left eye, odd = right eye."""
    if frame.ndim == 3:
        frame = frame[:, :, 0]
    return frame[:, ::2].copy(), frame[:, 1::2].copy()


def read_pair(cap, warm=6):
    """Settings take a few frames to take effect, so throw the first ones away."""
    f = None
    for _ in range(warm):
        ok, f = cap.read()
        if not ok:
            return None, None
    if f is None:
        return None, None
    return split(f)


# ---------------------------------------------------------------- commands

def probe():
    take_device()
    cap = open_leap()
    print(f"  {DEV} open")
    l, r = read_pair(cap)
    if l is None:
        print("  no frames"); cap.release(); return
    print(f"  raw frame splits to {l.shape[1]}x{l.shape[0]} per eye")
    for name, p in (("exposure(zoom)", P_EXPOSURE), ("led(contrast)", P_LED),
                    ("analog gain", P_AGAIN), ("digital gain", P_DGAIN)):
        print(f"    {name:<16} {cap.get(p)}")
    print(f"  left  eye: mean {l.mean():.1f}  max {l.max()}  px>100 {(l>100).mean()*100:.2f}%")
    print(f"  right eye: mean {r.mean():.1f}  max {r.max()}  px>100 {(r>100).mean()*100:.2f}%")
    cap.release()


def sweep(args):
    """Measure hand detection against exposure, with LEDs on and off."""
    from stream import Engine
    take_device()
    eng = Engine(os.path.expanduser("~/leap/asl/model_cisco6.pt"), video=False)

    print(f"  {'LEDs':<6}{'exposure':>10}{'a.gain':>8}{'mean':>7}{'px>100':>9}{'hand':>7}")
    print("  " + "-" * 48)
    best = None
    for leds in (True, False) if args.compare_leds else (True,):
        cap = open_leap(leds=leds)
        for exp in args.exposures:
            for ag in args.gains:
                cap.set(P_EXPOSURE, max(10, min(EXPOSURE_MAX, exp)))
                cap.set(P_AGAIN, ag)
                l, r = read_pair(cap, warm=8)
                if l is None:
                    print(f"  {'on' if leds else 'off':<6}{exp:>10}{ag:>8}    no frame")
                    continue
                pts, _ = eng._landmarks(l)
                hit = pts is not None
                print(f"  {'on' if leds else 'off':<6}{exp:>10}{ag:>8}{l.mean():>7.1f}"
                      f"{(l>100).mean()*100:>8.2f}%{'YES' if hit else '-':>7}")
                if hit and (best is None or l.mean() > best[3]):
                    best = (leds, exp, ag, l.mean())
        cap.release()
    eng.close()
    if best:
        print(f"\n  best detection: LEDs {'on' if best[0] else 'off'}, "
              f"exposure {best[1]}us, analog gain {best[2]} (mean {best[3]:.1f})")
    else:
        print("\n  no hand detected at any setting -- is a hand over the device?")


def grab(args):
    take_device()
    cap = open_leap(exposure=args.exposure, again=args.gain, dgain=args.dgain)
    l, r = read_pair(cap, warm=10)
    cap.release()
    if l is None:
        sys.exit("no frame")
    cv2.imwrite(args.out, np.hstack([l, r]))
    print(f"  {args.out}  ({l.shape[1]}x{l.shape[0]} per eye)  "
          f"left mean {l.mean():.1f}  right mean {r.mean():.1f}")


def main():
    ap = argparse.ArgumentParser(description="Direct Leap camera control")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("probe")
    s = sub.add_parser("sweep")
    s.add_argument("--exposures", type=int, nargs="+",
                   default=[500, 2000, 6000, 12000, 24000, 32222])
    s.add_argument("--gains", type=int, nargs="+", default=[16, 40, 63])
    s.add_argument("--compare-leds", action="store_true")
    g = sub.add_parser("grab")
    g.add_argument("out")
    g.add_argument("--exposure", type=int, default=12000)
    g.add_argument("--gain", type=int, default=40)
    g.add_argument("--dgain", type=int, default=4)
    a = ap.parse_args()

    try:
        if a.cmd == "probe":
            probe()
        elif a.cmd == "sweep":
            sweep(a)
        else:
            grab(a)
    finally:
        _restore()
        print(f"  [{SERVICE} restored]")


if __name__ == "__main__":
    main()
