#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<USAGE
Usage: $(basename "$0") [--check] [--target-dir PATH]

Sync source content to GitHub-compatible mirror directories:
- agents/  -> .github/agents/
- skills/  -> .github/skills/
- prompts/ -> .github/instructions/

Options:
  --check   verify mirror is in sync; exit non-zero if drift exists
USAGE
}

MODE="sync"
TARGET_DIR=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --check)
      MODE="check"
      shift
      ;;
    --target-dir)
      TARGET_DIR="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TARGET_ROOT="$REPO_ROOT"
if [[ -n "$TARGET_DIR" ]]; then
  if [[ ! -d "$TARGET_DIR" ]]; then
    echo "Target project directory does not exist: $TARGET_DIR" >&2
    echo "Please provide an existing directory with --target-dir PATH." >&2
    exit 1
  fi
  TARGET_ROOT="$(cd "$TARGET_DIR" && pwd)"
fi
declare -a PAIRS=(
  "agents:.github/agents"
  "skills:.github/skills"
  "prompts:.github/instructions"
)

sync_dir() {
  local src="$1"
  local dst="$2"

  if [[ ! -d "$src" ]]; then
    echo "Missing source directory: $src" >&2
    exit 1
  fi

  mkdir -p "$dst"

  # Remove stale paths in destination.
  while IFS= read -r -d '' existing; do
    local rel="${existing#"$dst"/}"
    if [[ ! -e "$src/$rel" ]]; then
      rm -rf "$existing"
    fi
  done < <(find "$dst" -mindepth 1 -depth -print0)

  # Copy source content recursively.
  while IFS= read -r -d '' item; do
    local rel="${item#"$src"/}"
    local target="$dst/$rel"
    if [[ -d "$item" ]]; then
      mkdir -p "$target"
    elif [[ -f "$item" ]]; then
      mkdir -p "$(dirname "$target")"
      cp "$item" "$target"
    fi
  done < <(find "$src" -mindepth 1 -print0)
}

check_dir() {
  local src="$1"
  local dst="$2"
  local label="$3"

  if [[ ! -d "$src" ]]; then
    echo "Missing source directory: $src" >&2
    exit 1
  fi
  mkdir -p "$dst"

  if ! diff -qr "$src" "$dst" >/dev/null; then
    echo "Mirror drift detected for $label." >&2
    echo "Run: bash scripts/sync_agents_to_github.sh" >&2
    diff -qr "$src" "$dst" || true
    return 1
  fi
  echo "Mirror check passed for $label."
  return 0
}

status=0
for pair in "${PAIRS[@]}"; do
  src_rel="${pair%%:*}"
  dst_rel="${pair#*:}"
  src_dir="$REPO_ROOT/$src_rel"
  dst_dir="$TARGET_ROOT/$dst_rel"

  if [[ "$MODE" == "sync" ]]; then
    sync_dir "$src_dir" "$dst_dir"
    echo "Mirrored $src_rel: $src_dir -> $dst_dir"
  else
    check_dir "$src_dir" "$dst_dir" "$src_rel" || status=1
  fi
done

exit $status
