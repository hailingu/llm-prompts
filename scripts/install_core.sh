#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<USAGE
Usage: $(basename "$0") [--scope project|global] [--skills all|none|a,b] [--agents all|none|a,b]
USAGE
}

SCOPE="project"
SKILLS_ARG="all"
AGENTS_ARG="all"
while [[ $# -gt 0 ]]; do
  case "$1" in
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

if [[ "$SCOPE" != "project" && "$SCOPE" != "global" ]]; then
  echo "Invalid --scope: $SCOPE (expected project|global)" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SKILLS_DIR="$REPO_ROOT/skills"
AGENTS_DIR="$REPO_ROOT/agents"

if [[ ! -d "$SKILLS_DIR" || ! -d "$AGENTS_DIR" ]]; then
  echo "Missing required directories: skills/ or agents/" >&2
  exit 1
fi

to_lines() {
  local value="$1"
  if [[ "$value" == "all" || "$value" == "none" ]]; then
    printf '%s\n' "$value"
    return
  fi
  printf '%s\n' "$value" | tr ',' '\n' | sed '/^$/d'
}

select_skills() {
  if [[ "$SKILLS_ARG" == "all" ]]; then
    find "$SKILLS_DIR" -mindepth 1 -maxdepth 1 -type d -exec basename {} \; | sort
    return
  fi
  if [[ "$SKILLS_ARG" == "none" ]]; then
    return
  fi
  to_lines "$SKILLS_ARG"
}

select_agents() {
  if [[ "$AGENTS_ARG" == "all" ]]; then
    find "$AGENTS_DIR" -mindepth 1 -maxdepth 1 -name '*.agent.md' -type f -exec basename {} \; | sort
    return
  fi
  if [[ "$AGENTS_ARG" == "none" ]]; then
    return
  fi
  to_lines "$AGENTS_ARG" | while IFS= read -r item; do
    if [[ "$item" == *.agent.md ]]; then
      printf '%s\n' "$item"
    else
      printf '%s.agent.md\n' "$item"
    fi
  done
}

validate_selected() {
  local missing=0
  while IFS= read -r skill; do
    [[ -z "$skill" ]] && continue
    if [[ ! -d "$SKILLS_DIR/$skill" ]]; then
      echo "Unknown skill: $skill" >&2
      missing=1
    fi
  done < <(select_skills)

  while IFS= read -r agent; do
    [[ -z "$agent" ]] && continue
    if [[ ! -f "$AGENTS_DIR/$agent" ]]; then
      echo "Unknown agent: $agent" >&2
      missing=1
    fi
  done < <(select_agents)

  if [[ $missing -ne 0 ]]; then
    exit 1
  fi
}

validate_selected

if [[ "$SCOPE" == "global" ]]; then
  CODEX_HOME="${CODEX_HOME:-$HOME/.codex}"
  mkdir -p "$CODEX_HOME/skills" "$CODEX_HOME/agents"

  while IFS= read -r skill; do
    [[ -z "$skill" ]] && continue
    ln -sfn "$SKILLS_DIR/$skill" "$CODEX_HOME/skills/$skill"
  done < <(select_skills)

  while IFS= read -r agent; do
    [[ -z "$agent" ]] && continue
    ln -sfn "$AGENTS_DIR/$agent" "$CODEX_HOME/agents/$agent"
  done < <(select_agents)

  echo "Core installed globally: $CODEX_HOME (skills=$SKILLS_ARG, agents=$AGENTS_ARG)"
else
  echo "Core installed for project scope: $REPO_ROOT (skills=$SKILLS_ARG, agents=$AGENTS_ARG)"
fi
