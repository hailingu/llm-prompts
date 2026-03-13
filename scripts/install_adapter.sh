#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<USAGE
Usage: $(basename "$0") --plugin codex|cline|copilot|kimi|all [--scope project|global] [--target-dir PATH]
USAGE
}

PLUGIN=""
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
COPILOT_HOME="${COPILOT_HOME:-$HOME/.copilot}"
TARGET_ROOT="$REPO_ROOT"
if [[ "$SCOPE" == "project" && -n "$TARGET_DIR" ]]; then
  if [[ ! -d "$TARGET_DIR" ]]; then
    echo "Target project directory does not exist: $TARGET_DIR" >&2
    echo "Please provide an existing directory with --target-dir PATH." >&2
    exit 1
  fi
  TARGET_ROOT="$(cd "$TARGET_DIR" && pwd)"
fi

render_template() {
  local tpl="$1"
  local out="$2"
  local root_value="$3"
  local escaped_root
  escaped_root=$(printf '%s' "$root_value" | sed 's/[\\&/]/\\&/g')
  mkdir -p "$(dirname "$out")"
  sed "s#__REPO_ROOT__#${escaped_root}#g" "$tpl" > "$out"
}

sync_dir_recursive() {
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

sync_cline_content() {
  local tmp
  tmp="$(mktemp -d)"
  mkdir -p "$tmp"

  # Merge agents into .clinerules root.
  while IFS= read -r -d '' item; do
    local rel="${item#"$REPO_ROOT/agents"/}"
    local target="$tmp/$rel"
    if [[ -d "$item" ]]; then
      mkdir -p "$target"
    elif [[ -f "$item" ]]; then
      mkdir -p "$(dirname "$target")"
      cp "$item" "$target"
    fi
  done < <(find "$REPO_ROOT/agents" -mindepth 1 -maxdepth 1 -print0)

  # Merge prompts into .clinerules root.
  while IFS= read -r -d '' item; do
    local rel="${item#"$REPO_ROOT/prompts"/}"
    local target="$tmp/$rel"
    if [[ -d "$item" ]]; then
      mkdir -p "$target"
    elif [[ -f "$item" ]]; then
      mkdir -p "$(dirname "$target")"
      cp "$item" "$target"
    fi
  done < <(find "$REPO_ROOT/prompts" -mindepth 1 -print0)

  sync_dir_recursive "$tmp" "$TARGET_ROOT/.clinerules"
  rm -rf "$tmp"
  echo "Mirrored agents+prompts: $REPO_ROOT/{agents,prompts} -> $TARGET_ROOT/.clinerules"
  sync_dir_recursive "$REPO_ROOT/skills" "$TARGET_ROOT/.cline/skills"
  echo "Mirrored skills: $REPO_ROOT/skills -> $TARGET_ROOT/.cline/skills"
}

install_one() {
  local p="$1"
  local project_root_value="."
  if [[ "$TARGET_ROOT" != "$REPO_ROOT" ]]; then
    project_root_value="$REPO_ROOT"
  fi
  case "$p" in
    codex)
      local tpl="$REPO_ROOT/adapters/codex/AGENTS.md.tpl"
      if [[ "$SCOPE" == "project" ]]; then
        render_template "$tpl" "$TARGET_ROOT/AGENTS.md" "$project_root_value"
        echo "Installed codex adapter: $TARGET_ROOT/AGENTS.md"
      else
        mkdir -p "$CODEX_HOME"
        render_template "$tpl" "$CODEX_HOME/AGENTS.md" "$REPO_ROOT"
        echo "Installed codex adapter: $CODEX_HOME/AGENTS.md"
      fi
      ;;
    cline)
      local tpl="$REPO_ROOT/adapters/cline/cline_mcp_settings.json.tpl"
      if [[ "$SCOPE" == "project" ]]; then
        render_template "$tpl" "$TARGET_ROOT/.cline/mcp_settings.json" "$project_root_value"
        echo "Installed cline adapter: $TARGET_ROOT/.cline/mcp_settings.json"
      else
        render_template "$tpl" "$HOME/.cline/mcp_settings.json" "$REPO_ROOT"
        echo "Installed cline adapter: $HOME/.cline/mcp_settings.json"
      fi
      ;;
    copilot)
      local tpl="$REPO_ROOT/adapters/copilot/instructions.md.tpl"
      if [[ "$SCOPE" == "project" ]]; then
        render_template "$tpl" "$TARGET_ROOT/.github/copilot-instructions.md" "$project_root_value"
        echo "Installed copilot adapter: $TARGET_ROOT/.github/copilot-instructions.md"
      else
        render_template "$tpl" "$COPILOT_HOME/copilot-instructions.md" "$REPO_ROOT"
        echo "Installed copilot adapter: $COPILOT_HOME/copilot-instructions.md"
      fi
      ;;
    kimi)
      local tpl="$REPO_ROOT/adapters/kimi/config.json.tpl"
      if [[ "$SCOPE" == "project" ]]; then
        render_template "$tpl" "$TARGET_ROOT/.kimi/config.json" "$project_root_value"
        echo "Installed kimi adapter: $TARGET_ROOT/.kimi/config.json"
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

# Only materialize GitHub compatibility mirrors when GitHub adapter is requested in project scope.
if [[ "$SCOPE" == "project" && ( "$PLUGIN" == "copilot" || "$PLUGIN" == "all" ) ]]; then
  "$SCRIPT_DIR/sync_agents_to_github.sh" --target-dir "$TARGET_ROOT"
fi

# Materialize Cline compatibility mirrors for project scope.
if [[ "$SCOPE" == "project" && ( "$PLUGIN" == "cline" || "$PLUGIN" == "all" ) ]]; then
  sync_cline_content
fi
