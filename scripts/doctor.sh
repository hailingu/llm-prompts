#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<USAGE
Usage: $(basename "$0") [--plugin codex|cline|copilot|kimi|all] [--scope project|global] [--target-dir PATH]
USAGE
}

PLUGIN="all"
SCOPE="project"
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
CODEX_HOME="${CODEX_HOME:-$HOME/.codex}"
COPILOT_HOME="${COPILOT_HOME:-$HOME/.copilot}"
TARGET_ROOT="$REPO_ROOT"
if [[ "$SCOPE" == "project" && -n "$TARGET_DIR" ]]; then
  if [[ ! -d "$TARGET_DIR" ]]; then
    echo "Target project directory does not exist: $TARGET_DIR" >&2
    echo "Please provide an existing directory with --target-dir PATH." >&2
    exit 1
  fi
  TARGET_ROOT="$TARGET_DIR"
fi

check_file() {
  local file="$1"
  if [[ -f "$file" ]]; then
    echo "PASS: $file"
    return 0
  fi
  echo "FAIL: $file"
  return 1
}

status=0
check_file "$REPO_ROOT/skills/rss-reader/SKILL.md" || status=1
check_file "$REPO_ROOT/agents/python-coder-specialist.agent.md" || status=1

check_plugin() {
  local p="$1"
  case "$p" in
    codex)
      if [[ "$SCOPE" == "project" ]]; then
        check_file "$TARGET_ROOT/AGENTS.md" || status=1
      else
        check_file "$CODEX_HOME/AGENTS.md" || status=1
      fi
      ;;
    cline)
      if [[ "$SCOPE" == "project" ]]; then
        check_file "$TARGET_ROOT/.cline/mcp_settings.json" || status=1
        check_file "$TARGET_ROOT/.clinerules/python-coder-specialist.agent.md" || status=1
        check_file "$TARGET_ROOT/.clinerules/task/task-breakdown.prompt.md" || status=1
        check_file "$TARGET_ROOT/.cline/skills/rss-reader/SKILL.md" || status=1
      else
        check_file "$HOME/.cline/mcp_settings.json" || status=1
      fi
      ;;
    copilot)
      if [[ "$SCOPE" == "project" ]]; then
        check_file "$TARGET_ROOT/.github/copilot-instructions.md" || status=1
        check_file "$TARGET_ROOT/.github/agents/python-coder-specialist.agent.md" || status=1
        check_file "$TARGET_ROOT/.github/skills/rss-reader/SKILL.md" || status=1
        check_file "$TARGET_ROOT/.github/instructions/task/task-breakdown.prompt.md" || status=1
      else
        check_file "$COPILOT_HOME/copilot-instructions.md" || status=1
      fi
      ;;
    kimi)
      if [[ "$SCOPE" == "project" ]]; then
        check_file "$TARGET_ROOT/.kimi/config.json" || status=1
      else
        check_file "$HOME/.kimi/config.json" || status=1
      fi
      ;;
    *)
      echo "Unknown plugin: $p" >&2
      status=1
      ;;
  esac
}

if [[ "$PLUGIN" == "all" ]]; then
  for p in codex cline copilot kimi; do
    check_plugin "$p"
  done
else
  check_plugin "$PLUGIN"
fi

if [[ $status -eq 0 ]]; then
  echo "Doctor checks passed."
else
  echo "Doctor found missing items."
fi

exit $status
