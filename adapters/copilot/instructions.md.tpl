# Copilot Repository Instructions

Use this repository as source-of-truth for reusable prompts, skills, standards, and agent definitions.

## Read Order
1. `agents/` for role behavior
2. `skills/*/SKILL.md` for executable workflows
3. `knowledge/standards/common/`, `knowledge/standards/engineering/`, and `knowledge/standards/data-science/` for coding/doc quality
4. `prompts/` and `knowledge/templates/` for output structure

## Path Policy
Always use repository-local paths under `__REPO_ROOT__`.
Do not prefer mirrored files under `.github/` when an equivalent root file exists.
