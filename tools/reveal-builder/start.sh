#!/usr/bin/env bash
set -euo pipefail
# start.sh — build + serve Reveal POC
# Default behavior: run the server in the foreground so it receives SIGINT and can be stopped with Ctrl-C.
# Use `--bg` to explicitly start the server in the background.

# Compute repository root (two levels up from tools/reveal-builder)
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

MODE="fg"
if [ "${1-}" = "--bg" ] || [ "${1-}" = "-b" ]; then
  MODE="bg"
fi

if [ "${1-}" = "--help" ] || [ "${1-}" = "-h" ]; then
  cat <<'USAGE'
Usage: start.sh [--bg|-b]    # start server in background
       start.sh [--fg|-f]    # (default) start server in foreground
       start.sh --help       # show this help
USAGE
  exit 0
fi

echo "Building Reveal POC..."
node tools/reveal-builder/build.js

# determine dist directory
if [ -d "dist" ]; then
  DIST_DIR="dist"
elif [ -d "tools/dist" ]; then
  DIST_DIR="tools/dist"
else
  DIST_DIR="dist"
fi

serve_with_httpserver() {
  if [ "$MODE" = "bg" ]; then
    echo "Starting http-server in background at http://localhost:8000 serving ${DIST_DIR}"
    npx http-server "${DIST_DIR}" -p 8000 >/dev/null 2>&1 &
    echo "Started (pid $!)"
    exit 0
  else
    echo "Starting http-server in foreground at http://localhost:8000 serving ${DIST_DIR}"
    # exec replaces shell so signals go directly to the server process
    exec npx http-server "${DIST_DIR}" -p 8000
  fi
}

serve_with_python() {
  if [ "$MODE" = "bg" ]; then
    echo "Starting python -m http.server in background at http://localhost:8000 serving ${DIST_DIR}"
    python3 -m http.server 8000 -d "${DIST_DIR}" >/dev/null 2>&1 &
    echo "Started (pid $!)"
    exit 0
  else
    echo "Starting python -m http.server in foreground at http://localhost:8000 serving ${DIST_DIR}"
    exec python3 -m http.server 8000 -d "${DIST_DIR}"
  fi
}

# serve using http-server if available, else fallback to python http
if command -v http-server >/dev/null 2>&1; then
  serve_with_httpserver
else
  serve_with_python
fi
