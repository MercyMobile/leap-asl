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

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

LEAP_REC = os.path.expanduser("~/leap/leap-rec")
PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reader.html")

STATE = {
    "tracking": False, "state": "IDLE", "letters": [], "text": "",
    "hand": None, "conf": None, "fps": 0.0,
    "frames": 0, "no_hand": 0, "gated": 0, "emitted": 0,
    "device": "waiting for frames",
}
LOCK = threading.Lock()
STOP = threading.Event()


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


def engine_loop(scratch, model):
    from stream import Engine                     # imported here so --help needs no torch
    eng = Engine(model, distortion_dir=scratch
                 if os.path.exists(os.path.join(scratch, "distortion_L.txt")) else None)
    seen = set()
    times = []
    text = []

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
            for f in (lp, rp):
                try: os.remove(f)
                except OSError: pass
            if gl is None:
                continue

            t = time.time()
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
                STATE.update({k: eng.stats[k] for k in ("frames", "no_hand", "gated", "voted", "emitted")})
                if ev:
                    text.append(ev.letter)
                    STATE["letters"] = [{"l": e, "t": round(t, 2)} for e in text[-40:]]
                    STATE["text"] = "".join(text[-40:])
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
        if self.path.startswith("/clear"):
            with LOCK:
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
    ap.add_argument("--model", default=os.path.expanduser("~/leap/asl/model_loso0.pt"))
    ap.add_argument("--interval", type=int, default=10, help="ms between frames")
    ap.add_argument("--scratch", default=None)
    a = ap.parse_args()

    scratch = a.scratch or tempfile.mkdtemp(prefix="leapread-")
    os.makedirs(scratch, exist_ok=True)
    print(f"  scratch  {scratch}")
    print(f"  model    {a.model}")
    print(f"  open     http://localhost:{a.port}/")

    threading.Thread(target=capture_loop, args=(scratch, a.interval), daemon=True).start()
    threading.Thread(target=engine_loop, args=(scratch, a.model), daemon=True).start()

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
