#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<USAGE
Usage: $(basename "$0") [--plugin codex|cline|copilot|kimi|all] [--scope project|global] [--target-dir PATH] [--skills all|none|a,b] [--agents all|none|a,b] [--skip-doctor]
USAGE
}

PLUGIN="codex"
SCOPE="project"
SKILLS_ARG="all"
AGENTS_ARG="all"
SKIP_DOCTOR="false"
TARGET_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --plugin)
      PLUGIN="${2:-}"
      shift 2
      ;;
    --scope)
      SCOPE="${2:-}"
      shift 2
      ;;
    --target-dir)
      TARGET_DIR="${2:-}"
      shift 2
      ;;
    --skills)
      SKILLS_ARG="${2:-}"
      shift 2
      ;;
    --agents)
      AGENTS_ARG="${2:-}"
      shift 2
      ;;
    --skip-doctor)
      SKIP_DOCTOR="true"
      shift
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

TARGET_DIR_ABS=""
if [[ "$SCOPE" == "project" ]]; then
  if [[ -z "$TARGET_DIR" ]]; then
    TARGET_DIR_ABS="$REPO_ROOT"
  else
    if [[ ! -d "$TARGET_DIR" ]]; then
      echo "Target project directory does not exist: $TARGET_DIR" >&2
      echo "Please provide an existing directory with --target-dir PATH." >&2
      exit 1
    fi
    TARGET_DIR_ABS="$(cd "$TARGET_DIR" && pwd)"
  fi
fi

SKIP_CORE="false"
if [[ "$SCOPE" == "global" && "$PLUGIN" == "cline" ]]; then
  SKIP_CORE="true"
  echo "Skipping core install for cline global scope (no global prompts/agents/skills)."
fi

if [[ "$SKIP_CORE" == "false" ]]; then
  "$SCRIPT_DIR/install_core.sh" --scope "$SCOPE" --target-dir "$TARGET_DIR_ABS" --skills "$SKILLS_ARG" --agents "$AGENTS_ARG"
fi
"$SCRIPT_DIR/install_adapter.sh" --plugin "$PLUGIN" --scope "$SCOPE" --target-dir "$TARGET_DIR_ABS"

if [[ "$SKIP_DOCTOR" == "false" ]]; then
  "$SCRIPT_DIR/doctor.sh" --plugin "$PLUGIN" --scope "$SCOPE" --target-dir "$TARGET_DIR_ABS"
fi

if [[ "$SCOPE" == "project" ]]; then
  echo "Setup completed (plugin=$PLUGIN, scope=$SCOPE, target=$TARGET_DIR_ABS, skills=$SKILLS_ARG, agents=$AGENTS_ARG)."
else
  echo "Setup completed (plugin=$PLUGIN, scope=$SCOPE, skills=$SKILLS_ARG, agents=$AGENTS_ARG)."
fi
