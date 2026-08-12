"""Live reader: Leap -> letters on screen, served to any browser on the network.

This is the session tool. It runs the engine over a continuous capture and serves
the picture-in-picture interface -- signer's own hand tracked in the corner so
they can see they are being picked up, letters accumulating underneath.

    reader.py                       # http://localhost:8770
    reader.py --port 8770 --model ~/leap/asl/model_tuned.pt

Capture strategy: `leap-rec` writes PGM pairs into a scratch directory and this
watches that directory, so nothing new has to be written in C. Frames are deleted
after they are read -- an hour at 100fps would otherwise be about a hundred
thousand files.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stream import Tuning, PRESETS      # no torch pulled in by this import

LEAP_REC = os.path.expanduser("~/leap/leap-rec")
PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reader.html")

STATE = {
    "tracking": False, "state": "IDLE", "letters": [], "text": "",
    "hand": None, "conf": None, "fps": 0.0,
    "frames": 0, "no_hand": 0, "gated": 0, "emitted": 0,
    "retracted": 0, "relocks": 0,
    "travel": 0.0, "shape": 0.0, "dwell": 0.0,
    "preset": "normal", "tune": {}, "recording": None,
    "device": "waiting for frames",
}
LOCK = threading.Lock()
STOP = threading.Event()
TEXT = []          # the authoritative letter list; /clear must empty THIS

# A live clip is the only recording that contains TRANSITIONS -- the hand on its
# way from one letter to the next. Posed captures do not, which is why spurious
# letters could never be reproduced offline and every fix had to be judged by
# feel. Recording a word here turns "it added a Q" into a file that can be
# replayed against any setting.
TUNING = Tuning()
SCRATCH = [None]
REC = {"on": False, "dir": None, "word": "", "n": 0, "fh": None, "failed": 0}
LIVE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw", "live")


def capture_loop(scratch, interval_ms):
    """Keep leap-rec running; restart it if the device hiccups."""
    while not STOP.is_set():
        try:
            p = subprocess.Popen([LEAP_REC, scratch, "1000000", str(interval_ms)],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            with LOCK:
                STATE["device"] = "capturing"
            while p.poll() is None and not STOP.is_set():
                time.sleep(0.3)
            p.terminate()
        except FileNotFoundError:
            with LOCK:
                STATE["device"] = f"{LEAP_REC} missing -- build it from src/Recorder.c"
            return
        if not STOP.is_set():
            with LOCK:
                STATE["device"] = "capture stopped, restarting"
            time.sleep(1.0)


def rec_start(word, scratch):
    """Begin keeping frames instead of deleting them."""
    rec_stop()
    word = "".join(c for c in word.upper() if c.isalpha()) or "CLIP"
    d = os.path.join(LIVE, f"{word}-{int(time.time())}")
    os.makedirs(d, exist_ok=True)
    for f in ("distortion_L.txt", "distortion_R.txt"):
        src = os.path.join(scratch, f)
        if os.path.exists(src):
            shutil.copy(src, d)               # replay must see the same geometry gate
    REC.update(on=True, dir=d, word=word, n=0, failed=0,
               fh=open(os.path.join(d, "stamps.csv"), "w"))
    REC["fh"].write("frame,t\n")
    return d


def rec_stop():
    if REC["fh"]:
        REC["fh"].close()
    d, n, bad = REC["dir"], REC["n"], REC["failed"]
    REC.update(on=False, fh=None)
    return d, n, bad


def engine_loop(scratch, model, tuning):
    from stream import Engine                     # imported here so --help needs no torch
    eng = Engine(model, tuning=tuning, distortion_dir=scratch
                 if os.path.exists(os.path.join(scratch, "distortion_L.txt")) else None)
    seen = set()
    times = []

    while not STOP.is_set():
        try:
            files = [f for f in os.listdir(scratch) if f.endswith("_L.pgm")]
        except FileNotFoundError:
            time.sleep(0.2); continue
        ids = sorted({f.split("_")[0] for f in files}, key=lambda s: int(s[1:]))
        ids = [i for i in ids if i not in seen]
        if not ids:
            time.sleep(0.02); continue

        # if we have fallen behind, skip to the newest -- latency matters more
        # than completeness for a live reader
        if len(ids) > 6:
            for old in ids[:-6]:
                seen.add(old)
                for e in ("L", "R"):
                    try: os.remove(os.path.join(scratch, f"{old}_{e}.pgm"))
                    except OSError: pass
            ids = ids[-6:]

        for fid in ids:
            seen.add(fid)
            lp = os.path.join(scratch, f"{fid}_L.pgm")
            rp = os.path.join(scratch, f"{fid}_R.pgm")
            gl = cv2.imread(lp, cv2.IMREAD_GRAYSCALE)
            gr = cv2.imread(rp, cv2.IMREAD_GRAYSCALE) if os.path.exists(rp) else None
            t = time.time()
            if REC["on"] and gl is not None:
                # shutil.move, not os.replace. The scratch directory is a tmpfs and
                # the clip directory is on disk, so a rename across them fails with
                # EXDEV -- which the first version swallowed, counted as success,
                # and reported "saved 362 frames" over an empty directory.
                ok = 0
                for f in (lp, rp):
                    if not os.path.exists(f):
                        continue
                    try:
                        shutil.move(f, os.path.join(REC["dir"], os.path.basename(f)))
                        ok += 1
                    except (OSError, shutil.Error):
                        REC["failed"] += 1
                REC["n"] += bool(ok)
                try:
                    REC["fh"].write(f"{fid},{t:.4f}\n")
                except (OSError, ValueError): pass
            else:
                for f in (lp, rp):
                    try: os.remove(f)
                    except OSError: pass
            if gl is None:
                continue

            ev = eng.push(gl, gr, t=t)
            times.append(t)
            times[:] = [x for x in times if t - x < 2.0]

            hand = None
            if eng.buf:
                pts = eng.buf[-1][2]
                hand = np.round(pts, 1).tolist()

            with LOCK:
                STATE["state"] = eng.state
                STATE["tracking"] = bool(eng.buf)
                STATE["hand"] = hand
                STATE["fps"] = round(len(times) / 2.0, 1)
                STATE["travel"] = round(eng.travel_speed, 2)
                STATE["shape"] = round(eng.shape_speed, 2)
                STATE["dwell"] = round(eng.dwell, 3)
                STATE["tune"] = eng.tune.as_dict()
                STATE["recording"] = ({"word": REC["word"], "frames": REC["n"],
                                       "failed": REC["failed"]}
                                      if REC["on"] else None)
                STATE.update({k: eng.stats[k] for k in
                              ("frames", "no_hand", "gated", "voted", "emitted",
                               "retracted", "relocks")})
                if ev:
                    # A retraction takes back the letter before this one: the hand
                    # brushed a shape on its way here and the reader believed it.
                    if ev.retract_prev and TEXT:
                        TEXT.pop()
                    TEXT.append(ev.letter)
                    del TEXT[:-40]
                    STATE["letters"] = [{"l": e, "t": round(t, 2)} for e in TEXT]
                    STATE["text"] = "".join(TEXT)
    eng.close()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.startswith("/state"):
            with LOCK:
                body = json.dumps(STATE).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path.startswith("/tune"):
            q = parse_qs(urlparse(self.path).query)
            with LOCK:
                if "preset" in q and q["preset"][0] in PRESETS:
                    STATE["preset"] = q["preset"][0]
                    for k, v in PRESETS[q["preset"][0]].items():
                        setattr(TUNING, k, v)
                for k, v in q.items():
                    if k != "preset" and hasattr(TUNING, k):
                        try: setattr(TUNING, k, type(getattr(TUNING, k))(v[0]))
                        except (TypeError, ValueError): pass
                body = json.dumps(TUNING.as_dict()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path.startswith("/record"):
            q = parse_qs(urlparse(self.path).query)
            if "stop" in q:
                d, n, bad = rec_stop()
                real = len([f for f in os.listdir(d)
                            if f.endswith("_L.pgm")]) if d and os.path.isdir(d) else 0
                # Count what is ON DISK, not what the loop thinks it wrote.
                msg = (f"saved {real} frames to {d}"
                       + (f"  ({bad} writes FAILED)" if bad else "")) if d else "not recording"
            else:
                d = rec_start(q.get("word", ["CLIP"])[0], SCRATCH[0])
                msg = f"recording to {d}"
            print(f"  {msg}")
            body = json.dumps({"msg": msg}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path.startswith("/clear"):
            with LOCK:
                TEXT.clear()
                STATE["letters"] = []; STATE["text"] = ""
            self.send_response(204); self.end_headers()
            return
        try:
            body = open(PAGE, "rb").read()
        except FileNotFoundError:
            body = b"<h1>reader.html missing</h1>"
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8770)
    # model_loso0 -- the ASLA-Leap transfer model -- was the default here, and it
    # scores 37.8% per frame on Cisco's own posed set against 99.5% for
    # model_cisco26b. Anyone following the README got the 37.8% one, which alone
    # produces letters nobody signed.
    ap.add_argument("--model", default=os.path.expanduser("~/leap/asl/model_cisco26b.pt"))
    ap.add_argument("--preset", default="normal", choices=list(PRESETS),
                    help="how long a letter must be held before it is committed")
    ap.add_argument("--interval", type=int, default=10, help="ms between frames")
    ap.add_argument("--scratch", default=None)
    a = ap.parse_args()

    scratch = a.scratch or tempfile.mkdtemp(prefix="leapread-")
    os.makedirs(scratch, exist_ok=True)
    print(f"  scratch  {scratch}")
    print(f"  model    {a.model}")
    print(f"  preset   {a.preset}  (dwell {TUNING.dwell_sec*1000:.0f}ms)")
    print(f"  open     http://localhost:{a.port}/")

    threading.Thread(target=capture_loop, args=(scratch, a.interval), daemon=True).start()
    SCRATCH[0] = scratch
    STATE["preset"] = a.preset
    for k, v in PRESETS[a.preset].items():
        setattr(TUNING, k, v)
    threading.Thread(target=engine_loop, args=(scratch, a.model, TUNING), daemon=True).start()

    srv = ThreadingHTTPServer(("0.0.0.0", a.port), Handler)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopping")
    finally:
        STOP.set()
        time.sleep(0.4)
        if not a.scratch:
            shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    main()
