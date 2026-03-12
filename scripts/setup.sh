#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<USAGE
Usage: $(basename "$0") [--plugin codex|cline|copilot|kimi|all] [--scope project|global] [--skills all|none|a,b] [--agents all|none|a,b] [--skip-doctor]
USAGE
}

PLUGIN="codex"
SCOPE="project"
SKILLS_ARG="all"
AGENTS_ARG="all"
SKIP_DOCTOR="false"

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

"$SCRIPT_DIR/install_core.sh" --scope "$SCOPE" --skills "$SKILLS_ARG" --agents "$AGENTS_ARG"
"$SCRIPT_DIR/install_adapter.sh" --plugin "$PLUGIN" --scope "$SCOPE"

if [[ "$SKIP_DOCTOR" == "false" ]]; then
  "$SCRIPT_DIR/doctor.sh" --plugin "$PLUGIN" --scope "$SCOPE"
fi

echo "Setup completed (plugin=$PLUGIN, scope=$SCOPE, skills=$SKILLS_ARG, agents=$AGENTS_ARG)."
