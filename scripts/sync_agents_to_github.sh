#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<USAGE
Usage: $(basename "$0") [--check]

Sync source agent files from agents/ to .github/agents/.

Options:
  --check   verify mirror is in sync; exit non-zero if drift exists
USAGE
}

MODE="sync"
if [[ ${1:-} == "--check" ]]; then
  MODE="check"
elif [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
  usage
  exit 0
elif [[ $# -gt 0 ]]; then
  echo "Unknown argument: $1" >&2
  usage
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC_DIR="$REPO_ROOT/agents"
DST_DIR="$REPO_ROOT/.github/agents"

if [[ ! -d "$SRC_DIR" ]]; then
  echo "Missing source directory: $SRC_DIR" >&2
  exit 1
fi
mkdir -p "$DST_DIR"

sync_once() {
  # Remove stale mirrored files.
  find "$DST_DIR" -mindepth 1 -maxdepth 1 -name '*.agent.md' -type f | while IFS= read -r existing; do
    base="$(basename "$existing")"
    if [[ ! -f "$SRC_DIR/$base" ]]; then
      rm -f "$existing"
    fi
  done

  # Copy all source files.
  find "$SRC_DIR" -mindepth 1 -maxdepth 1 -name '*.agent.md' -type f | while IFS= read -r src; do
    cp "$src" "$DST_DIR/$(basename "$src")"
  done
}

if [[ "$MODE" == "sync" ]]; then
  sync_once
  echo "Mirrored agents: $SRC_DIR -> $DST_DIR"
  exit 0
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT
EXPECTED="$TMP_DIR/expected"
mkdir -p "$EXPECTED"

find "$SRC_DIR" -mindepth 1 -maxdepth 1 -name '*.agent.md' -type f | while IFS= read -r src; do
  cp "$src" "$EXPECTED/$(basename "$src")"
done

# Compare by file set + content
if ! diff -qr "$EXPECTED" "$DST_DIR" >/dev/null; then
  echo "Agent mirror drift detected between agents/ and .github/agents/." >&2
  echo "Run: bash scripts/sync_agents_to_github.sh" >&2
  diff -qr "$EXPECTED" "$DST_DIR" || true
  exit 1
fi

echo "Agent mirror check passed."
