"""Labeled fingerspelling capture. Built for one hour with a fluent signer.

That hour is the scarce resource, so this is written to waste none of it:
it is resumable, it never asks a question mid-session, and every burst lands
on disk labeled the moment it is recorded.

  capture.py posed  --subject interp1              # all 26, held
  capture.py speed  --subject interp1 --words CISCO MERCY MOBILE
  capture.py posed  --subject cisco --only A S T M N   # drill the fist family
  capture.py posed  --subject test --dry-run       # no device needed

Two modes, because they are two different problems:

  posed  -- each letter held still. This is what the classifier was trained on
            and what it will score well against.
  speed  -- whole words at natural pace. Letters blend into each other here
            (co-articulation), which is the real-world case and much harder.
            Captured as one continuous burst, segmented later, because a fluent
            signer cannot pause between letters without destroying the thing
            we are trying to measure.

Output:
  data/raw/<subject>/<mode>/<label>/  f*_L.pgm f*_R.pgm native.csv distortion_*.txt
  data/raw/manifest.csv
"""
import argparse
import csv
import os
import subprocess
import sys
import time

LEAP_REC = os.path.expanduser("~/leap/leap-rec")
RAW = os.path.expanduser("~/leap/asl/data/raw")

ALPHABET = [c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
DYNAMIC = {"J", "Z"}          # motion letters -- need a longer burst

# Plain-language handshape cues. For the operator rehearsing solo; a fluent
# signer needs none of this. Descriptions are the standard ASL manual alphabet.
CUE = {
    "A": "fist, thumb alongside the index finger",
    "B": "flat hand, fingers together and up, thumb folded across the palm",
    "C": "hand curved into a C",
    "D": "index up, other fingertips meet the thumb",
    "E": "fingertips curled down onto the thumb",
    "F": "index and thumb touch in a circle, other three up",
    "G": "index and thumb extended flat, pointing sideways",
    "H": "index and middle extended together, pointing sideways",
    "I": "pinky up, rest in a fist",
    "J": "pinky up, then draw a J hook downward  [MOTION]",
    "K": "index and middle up in a V, thumb between them",
    "L": "index up, thumb out -- an L",
    "M": "thumb tucked under three fingers",
    "N": "thumb tucked under two fingers",
    "O": "fingertips and thumb meet in an O",
    "P": "like K, rotated to point downward",
    "Q": "like G, rotated to point downward",
    "R": "index and middle crossed",
    "S": "fist, thumb across the front of the fingers",
    "T": "thumb tucked under the index finger",
    "U": "index and middle up, held together",
    "V": "index and middle up, apart",
    "W": "index, middle, ring up and spread",
    "X": "index crooked into a hook",
    "Y": "thumb and pinky out, rest folded",
    "Z": "index finger draws a Z in the air  [MOTION]",
}

FONT = {
    "A": ["  ###  ", " #   # ", " ##### ", " #   # ", " #   # "],
    "B": [" ####  ", " #   # ", " ####  ", " #   # ", " ####  "],
    "C": ["  #### ", " #     ", " #     ", " #     ", "  #### "],
    "D": [" ####  ", " #   # ", " #   # ", " #   # ", " ####  "],
    "E": [" ##### ", " #     ", " ####  ", " #     ", " ##### "],
    "F": [" ##### ", " #     ", " ####  ", " #     ", " #     "],
    "G": ["  #### ", " #     ", " #  ## ", " #   # ", "  #### "],
    "H": [" #   # ", " #   # ", " ##### ", " #   # ", " #   # "],
    "I": [" ##### ", "   #   ", "   #   ", "   #   ", " ##### "],
    "J": [" ##### ", "    #  ", "    #  ", " #  #  ", "  ##   "],
    "K": [" #   # ", " #  #  ", " ###   ", " #  #  ", " #   # "],
    "L": [" #     ", " #     ", " #     ", " #     ", " ##### "],
    "M": [" #   # ", " ## ## ", " # # # ", " #   # ", " #   # "],
    "N": [" #   # ", " ##  # ", " # # # ", " #  ## ", " #   # "],
    "O": ["  ###  ", " #   # ", " #   # ", " #   # ", "  ###  "],
    "P": [" ####  ", " #   # ", " ####  ", " #     ", " #     "],
    "Q": ["  ###  ", " #   # ", " #   # ", " #  #  ", "  ## # "],
    "R": [" ####  ", " #   # ", " ####  ", " #  #  ", " #   # "],
    "S": ["  #### ", " #     ", "  ###  ", "     # ", " ####  "],
    "T": [" ##### ", "   #   ", "   #   ", "   #   ", "   #   "],
    "U": [" #   # ", " #   # ", " #   # ", " #   # ", "  ###  "],
    "V": [" #   # ", " #   # ", " #   # ", "  # #  ", "   #   "],
    "W": [" #   # ", " #   # ", " # # # ", " ## ## ", " #   # "],
    "X": [" #   # ", "  # #  ", "   #   ", "  # #  ", " #   # "],
    "Y": [" #   # ", "  # #  ", "   #   ", "   #   ", "   #   "],
    "Z": [" ##### ", "    #  ", "   #   ", "  #    ", " ##### "],
}

BIG = "\033[1m"
DIM = "\033[2m"
GRN = "\033[32m"
YEL = "\033[33m"
RST = "\033[0m"


def banner(text, color=GRN):
    """Render a word in block letters, big enough to read across a table."""
    rows = ["", "", "", "", ""]
    for ch in text.upper():
        glyph = FONT.get(ch, ["       "] * 5)
        for i in range(5):
            rows[i] += glyph[i] + " "
    return color + BIG + "\n".join(rows) + RST


def clear():
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()


def countdown(secs, label):
    for i in range(secs, 0, -1):
        sys.stdout.write(f"\r  {YEL}{label} in {i}...{RST}   ")
        sys.stdout.flush()
        time.sleep(1)
    sys.stdout.write(f"\r  {GRN}{BIG}RECORDING{RST}            \n")
    sys.stdout.flush()


def record(outdir, n, interval_ms, dry_run=False):
    os.makedirs(outdir, exist_ok=True)
    if dry_run:
        time.sleep(0.4)
        return n, True
    try:
        p = subprocess.run([LEAP_REC, outdir, str(n), str(interval_ms)],
                           capture_output=True, text=True, timeout=n * interval_ms / 1000 + 30)
    except subprocess.TimeoutExpired:
        return 0, False
    got = len({f.split("_")[0] for f in os.listdir(outdir) if f.endswith("_L.pgm")})
    return got, p.returncode == 0 and got > 0


def already_done(outdir, want):
    if not os.path.isdir(outdir):
        return 0
    return len({f.split("_")[0] for f in os.listdir(outdir) if f.endswith("_L.pgm")})


def log(subject, mode, label, outdir, n, hand):
    os.makedirs(RAW, exist_ok=True)
    path = os.path.join(RAW, "manifest.csv")
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["timestamp", "subject", "mode", "label", "hand", "frames", "dir"])
        w.writerow([time.strftime("%Y-%m-%dT%H:%M:%S"), subject, mode, label,
                    hand, n, os.path.relpath(outdir, RAW)])


def run(args):
    mode = args.mode
    targets = args.words if mode == "speed" else (args.only or ALPHABET)
    targets = [t.upper() for t in targets]

    total = len(targets)
    done = []
    clear()
    print(f"{BIG}Fingerspelling capture{RST}   subject={args.subject}  mode={mode}  "
          f"hand={args.hand}{'  [DRY RUN]' if args.dry_run else ''}")
    print(f"{DIM}{total} items. Ctrl-C to stop -- everything recorded so far is already saved.{RST}\n")
    time.sleep(1.5)

    for i, label in enumerate(targets, 1):
        secs = args.seconds * (2 if label in DYNAMIC and mode == "posed" else 1)
        n = int(secs * 1000 / args.interval)

        # Resume against what was actually asked for. A fixed threshold would
        # treat a burst interrupted at 50 of 300 frames as finished.
        outdir = os.path.join(RAW, args.subject, mode, label)
        have = already_done(outdir, n)
        if have >= n * 0.8 and not args.redo:
            print(f"  {DIM}[{i}/{total}] {label}: have {have}/{n} frames, skipping{RST}")
            continue
        if have:
            print(f"  {DIM}[{i}/{total}] {label}: only {have}/{n} frames, recapturing{RST}")
            for f in os.listdir(outdir):
                if f.endswith(".pgm"):
                    os.remove(os.path.join(outdir, f))

        clear()
        print(f"{DIM}[{i}/{total}]  subject {args.subject}  ·  {args.hand} hand{RST}\n")
        print(banner(label))
        if mode == "posed":
            print(f"\n  {CUE.get(label, '')}\n")
            if label in DYNAMIC:
                print(f"  {YEL}Motion letter -- sign it repeatedly for the whole window.{RST}\n")
        else:
            print(f"\n  Spell it at your natural speed. Do not pause between letters.\n")

        countdown(args.lead, "hold" if mode == "posed" else "spell")
        got, ok = record(outdir, n, args.interval, args.dry_run)

        if ok:
            log(args.subject, mode, label, outdir, got, args.hand)
            done.append(label)
            print(f"  {GRN}captured {got} frames{RST}")
        else:
            print(f"  {YEL}FAILED -- got {got} frames. Is the Leap plugged in?{RST}")
        time.sleep(args.rest)

    clear()
    print(f"{BIG}Done.{RST}  captured {len(done)}/{total}: {' '.join(done) if done else '(none)'}")
    print(f"{DIM}manifest: {os.path.join(RAW, 'manifest.csv')}{RST}")


def main():
    p = argparse.ArgumentParser(description="Labeled ASL fingerspelling capture")
    p.add_argument("mode", choices=["posed", "speed"])
    p.add_argument("--subject", required=True, help="signer id, e.g. interp1")
    p.add_argument("--hand", default="right", choices=["right", "left"])
    p.add_argument("--only", nargs="*", help="posed: capture just these letters")
    p.add_argument("--words", nargs="*", default=["HELLO"], help="speed: words to spell")
    p.add_argument("--seconds", type=float, default=3.0, help="burst length per item")
    p.add_argument("--interval", type=int, default=10, help="ms between frames (10 = 100fps)")
    p.add_argument("--lead", type=int, default=3, help="countdown seconds")
    p.add_argument("--rest", type=float, default=1.2, help="pause between items")
    p.add_argument("--frames", type=int, default=30, help="frames that count as already captured")
    p.add_argument("--redo", action="store_true", help="recapture even if data exists")
    p.add_argument("--dry-run", action="store_true", help="exercise the flow with no device")
    args = p.parse_args()

    if not args.dry_run and not os.path.exists(LEAP_REC):
        sys.exit(f"{LEAP_REC} not found -- build it from src/Recorder.c first")
    try:
        run(args)
    except KeyboardInterrupt:
        print(f"\n\n{YEL}Stopped. Everything captured so far is saved.{RST}")


if __name__ == "__main__":
    main()
