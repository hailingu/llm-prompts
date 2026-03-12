# AGENTS.md

## Workspace Policy
- Treat this repository as the source of truth.
- Prefer repository-local assets before any global assets.
- Use these local directories:
  - `__REPO_ROOT__/agents`
  - `__REPO_ROOT__/skills`
  - `__REPO_ROOT__/prompts`
  - `__REPO_ROOT__/knowledge/standards/common`
  - `__REPO_ROOT__/knowledge/standards/engineering/java`
  - `__REPO_ROOT__/knowledge/standards/engineering/go`
  - `__REPO_ROOT__/knowledge/standards/engineering/python`
  - `__REPO_ROOT__/knowledge/standards/data-science`
  - `__REPO_ROOT__/knowledge/templates`

## Skill Discovery
- Only load skills from `__REPO_ROOT__/skills/*/SKILL.md` unless user explicitly asks for global skills.
- Resolve skill-relative paths from each skill directory.

## Agent Discovery
- Use `__REPO_ROOT__/agents/*.agent.md` as agent definitions.
