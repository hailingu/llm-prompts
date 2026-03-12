#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<USAGE
Usage: $(basename "$0") --plugin codex|cline|copilot|kimi|all [--scope project|global]
USAGE
}

PLUGIN=""
SCOPE="project"

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

if [[ -z "$PLUGIN" ]]; then
  echo "--plugin is required" >&2
  usage
  exit 1
fi

if [[ "$SCOPE" != "project" && "$SCOPE" != "global" ]]; then
  echo "Invalid --scope: $SCOPE (expected project|global)" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CODEX_HOME="${CODEX_HOME:-$HOME/.codex}"

render_template() {
  local tpl="$1"
  local out="$2"
  local root_value="$3"
  local escaped_root
  escaped_root=$(printf '%s' "$root_value" | sed 's/[\\&/]/\\&/g')
  mkdir -p "$(dirname "$out")"
  sed "s#__REPO_ROOT__#${escaped_root}#g" "$tpl" > "$out"
}

install_one() {
  local p="$1"
  case "$p" in
    codex)
      local tpl="$REPO_ROOT/adapters/codex/AGENTS.md.tpl"
      if [[ "$SCOPE" == "project" ]]; then
        render_template "$tpl" "$REPO_ROOT/AGENTS.md" "."
        echo "Installed codex adapter: $REPO_ROOT/AGENTS.md"
      else
        mkdir -p "$CODEX_HOME"
        render_template "$tpl" "$CODEX_HOME/AGENTS.md" "$REPO_ROOT"
        echo "Installed codex adapter: $CODEX_HOME/AGENTS.md"
      fi
      ;;
    cline)
      local tpl="$REPO_ROOT/adapters/cline/cline_mcp_settings.json.tpl"
      if [[ "$SCOPE" == "project" ]]; then
        render_template "$tpl" "$REPO_ROOT/.cline/mcp_settings.json" "."
        echo "Installed cline adapter: $REPO_ROOT/.cline/mcp_settings.json"
      else
        render_template "$tpl" "$HOME/.cline/mcp_settings.json" "$REPO_ROOT"
        echo "Installed cline adapter: $HOME/.cline/mcp_settings.json"
      fi
      ;;
    copilot)
      local tpl="$REPO_ROOT/adapters/copilot/instructions.md.tpl"
      if [[ "$SCOPE" == "project" ]]; then
        render_template "$tpl" "$REPO_ROOT/.github/copilot-instructions.md" "."
        echo "Installed copilot adapter: $REPO_ROOT/.github/copilot-instructions.md"
      else
        render_template "$tpl" "$HOME/.copilot/instructions.md" "$REPO_ROOT"
        echo "Installed copilot adapter: $HOME/.copilot/instructions.md"
      fi
      ;;
    kimi)
      local tpl="$REPO_ROOT/adapters/kimi/config.json.tpl"
      if [[ "$SCOPE" == "project" ]]; then
        render_template "$tpl" "$REPO_ROOT/.kimi/config.json" "."
        echo "Installed kimi adapter: $REPO_ROOT/.kimi/config.json"
      else
        render_template "$tpl" "$HOME/.kimi/config.json" "$REPO_ROOT"
        echo "Installed kimi adapter: $HOME/.kimi/config.json"
      fi
      ;;
    *)
      echo "Unsupported plugin: $p" >&2
      exit 1
      ;;
  esac
}

if [[ "$PLUGIN" == "all" ]]; then
  for p in codex cline copilot kimi; do
    install_one "$p"
  done
else
  install_one "$PLUGIN"
fi

# Reduce repo duplication by default:
# only materialize .github/agents when GitHub adapter is requested in project scope.
if [[ "$SCOPE" == "project" && ( "$PLUGIN" == "copilot" || "$PLUGIN" == "all" ) ]]; then
  "$SCRIPT_DIR/sync_agents_to_github.sh"
fi
