"""Self-paced capture prompter. Built for sitting down with a fluent signer.

The terminal version works but assumes the signer can see a terminal. A signer
sitting at the sensor cannot, and asking them to lean over and read a console
between letters wastes the one thing that is genuinely scarce here: their time.

So the prompt goes to a browser -- phone, tablet, whatever is to hand. One big
letter, the handshape in plain words, and a button. They control the pace; there
is no countdown racing them, and nothing is lost if they need to stop and think
or answer a question.

    prompter.py --subject sarah          # http://<box>:8771

Recording still goes through leap-rec into data/raw/<subject>/posed/<LETTER>/,
identical to capture.py, so everything downstream is unchanged.
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from capture import ALPHABET, CUE, DYNAMIC, RAW, LEAP_REC, log, already_done

PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompter.html")
STATE = {"letter": None, "index": 0, "total": 0, "cue": "", "dynamic": False,
         "status": "ready", "captured": {}, "subject": "", "message": ""}
LOCK = threading.Lock()
CFG = {}


def outdir(letter):
    return os.path.join(RAW, CFG["subject"], "posed", letter)


def refresh():
    with LOCK:
        L = CFG["letters"][STATE["index"]]
        n = int(CFG["seconds"] * (2 if L in DYNAMIC else 1) * 1000 / CFG["interval"])
        STATE.update(letter=L, cue=CUE.get(L, ""), dynamic=L in DYNAMIC,
                     total=len(CFG["letters"]), subject=CFG["subject"],
                     captured={l: already_done(outdir(l), 0) for l in CFG["letters"]},
                     want=n)


def do_record():
    L = STATE["letter"]
    d = outdir(L)
    os.makedirs(d, exist_ok=True)
    for f in os.listdir(d):
        if f.endswith(".pgm"):
            os.remove(os.path.join(d, f))
    n = STATE["want"]
    with LOCK:
        STATE["status"] = "recording"
        STATE["message"] = ""
    try:
        subprocess.run([LEAP_REC, d, str(n), str(CFG["interval"])],
                       capture_output=True, timeout=n * CFG["interval"] / 1000 + 30)
    except subprocess.TimeoutExpired:
        pass
    got = already_done(d, n)
    ok = got >= n * 0.6
    if ok:
        log(CFG["subject"], "posed", L, d, got, CFG["hand"])
    with LOCK:
        STATE["status"] = "done" if ok else "failed"
        STATE["message"] = (f"{got} frames captured" if ok
                            else f"only {got} frames - is the hand over the sensor?")
        STATE["captured"][L] = got
    if ok and STATE["index"] < len(CFG["letters"]) - 1:
        time.sleep(0.9)
        with LOCK:
            STATE["index"] += 1
        refresh()
        with LOCK:
            STATE["status"] = "ready"


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj):
        b = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        p = self.path.split("?")[0]
        if p == "/state":
            with LOCK:
                return self._json(dict(STATE))
        if p == "/record":
            if STATE["status"] != "recording":
                threading.Thread(target=do_record, daemon=True).start()
            return self._json({"ok": True})
        if p in ("/next", "/prev", "/redo"):
            with LOCK:
                if p == "/next":
                    STATE["index"] = min(STATE["index"] + 1, len(CFG["letters"]) - 1)
                elif p == "/prev":
                    STATE["index"] = max(STATE["index"] - 1, 0)
                STATE["status"] = "ready"; STATE["message"] = ""
            refresh()
            return self._json({"ok": True})
        try:
            body = open(PAGE, "rb").read()
        except FileNotFoundError:
            body = b"<h1>prompter.html missing</h1>"
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--hand", default="right", choices=["right", "left"])
    ap.add_argument("--letters", nargs="*", default=None)
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--interval", type=int, default=10)
    ap.add_argument("--port", type=int, default=8771)
    a = ap.parse_args()

    CFG.update(subject=a.subject, hand=a.hand, seconds=a.seconds, interval=a.interval,
               letters=[c.upper() for c in (a.letters or ALPHABET)])
    refresh()
    print(f"  subject  {a.subject}   {len(CFG['letters'])} letters")
    print(f"  open     http://localhost:{a.port}/   (or the box's LAN/Tailscale address)")
    ThreadingHTTPServer(("0.0.0.0", a.port), H).serve_forever()


if __name__ == "__main__":
    main()
