#!/usr/bin/env bash
# Start the ASL reader and open it. Safe to run twice -- if it is already up,
# this just opens the page.
#
# Built for demonstrating: the failure mode that matters at a demo is a black
# window with no explanation, so every exit path here says what happened and
# waits for a keypress instead of vanishing.

set -uo pipefail
PORT="${LEAP_READER_PORT:-8770}"
ROOT="$HOME/leap"
PY="$ROOT/venv/bin/python"
LOG="$HOME/.local/state/leap-reader.log"
URL="http://localhost:$PORT/"

mkdir -p "$(dirname "$LOG")"

hold() {          # never let the terminal close on an error nobody read
  echo
  read -r -n 1 -p "Press any key to close. " _ || true
  echo
}

open_page() {
  [ -n "${LEAP_READER_NOOPEN:-}" ] && { echo "  (not opening a browser: $URL)"; return 0; }
  for b in xdg-open gio firefox google-chrome chromium; do
    if command -v "$b" >/dev/null 2>&1; then
      { "$b" "$URL" >/dev/null 2>&1 & } ; return 0
    fi
  done
  echo "  No browser launcher found -- open this yourself:  $URL"
}

up() { curl -sf -m 2 "http://localhost:$PORT/state" >/dev/null 2>&1; }

echo "  ASL reader"
echo "  ----------"

if up; then
  echo "  Already running on port $PORT."
  open_page
  exit 0
fi

# Only ONE reader may run: they each spawn leap-rec, and two of them fighting
# over the controller drops the frame rate to a crawl (measured: 68fps alone,
# 5fps with a second instance). If one is already up on another port, use it
# rather than starting a rival.
other="$(pgrep -af 'asl/reader\.py' | grep -v "$$" | head -1 || true)"
if [ -n "$other" ]; then
  oport="$(sed -n 's/.*--port \([0-9]*\).*/\1/p' <<<"$other")"
  oport="${oport:-8770}"
  if curl -sf -m 2 "http://localhost:$oport/state" >/dev/null 2>&1; then
    PORT="$oport"; URL="http://localhost:$PORT/"
    echo "  A reader is already running on port $PORT -- using that one."
    open_page
    exit 0
  fi
  echo "  A reader process exists but is not answering. Clearing it."
  pkill -f 'asl/reader\.py' || true
  sleep 2
fi

if [ ! -x "$PY" ]; then
  echo "  MISSING: $PY"
  echo "  Rebuild it:  uv venv --python 3.12 $ROOT/venv"
  hold; exit 1
fi

if ! lsusb 2>/dev/null | grep -qi "leap motion"; then
  echo "  WARNING: no Leap Motion controller on USB. Starting anyway --"
  echo "  the page will come up and say 'waiting for frames'."
fi

echo "  Starting... (first run loads the model, give it ~15 seconds)"
setsid nohup "$PY" "$ROOT/asl/reader.py" --port "$PORT" >>"$LOG" 2>&1 < /dev/null &

for _ in $(seq 1 40); do
  sleep 1
  up && break
done

if up; then
  fps=$(curl -sf -m 2 "http://localhost:$PORT/state" \
        | sed -n 's/.*"fps": \([0-9.]*\).*/\1/p')
  echo "  Up on $URL  (${fps:-?} fps)"
  echo "  Log: $LOG"
  open_page
  exit 0
fi

echo "  FAILED to come up within 40 seconds. Last lines of the log:"
echo
tail -n 15 "$LOG" | sed 's/^/    /'
hold
exit 1
