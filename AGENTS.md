# AGENTS.md

## Workspace Policy
- Treat this repository as the source of truth.
- Prefer repository-local assets before any global assets.
- Use these local directories:
  - `./agents`
  - `./skills`
  - `./prompts`
  - `./knowledge/standards/common`
  - `./knowledge/standards/engineering/java`
  - `./knowledge/standards/engineering/go`
  - `./knowledge/standards/engineering/python`
  - `./knowledge/standards/data-science`
  - `./knowledge/templates`

## Skill Discovery
- Only load skills from `./skills/*/SKILL.md` unless user explicitly asks for global skills.
- Resolve skill-relative paths from each skill directory.

## Agent Discovery
- Use `./agents/*.agent.md` as agent definitions.
