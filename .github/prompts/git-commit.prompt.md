---
agent: "agent"
model: "Raptor mini (Preview)"
description: "You are an AI agent designed to prepare and execute an open-source-friendly git commit: review changes, update docs/changelog when needed, and produce a detailed Chinese commit message that follows the repository .gitmessage template."
---

Generate and finally commit a detailed git commit message (in Chinese) based on the actual code changes, following a typical open-source contribution workflow. 

Instructions:

Workflow (do these in order):

1) Inspect changes
- Use `git status` and `git diff` (and `git diff --staged` if needed) to understand exactly what changed.
- Identify scope: what is user-facing, what is internal, and what is purely formatting.

2) Safety / hygiene checks (must)
- Do NOT include secrets, tokens, private keys, credentials, or large binaries.
- If the change touches dependencies/config, confirm no accidental environment-specific paths are committed.

3) Quality checks (as applicable)
- Run available tests / linters / formatters relevant to the change.
- If there are no tests, describe manual verification steps under the “测试” section.

4) Update project docs (only when needed)
- Update `README.md` when behavior/usage/setup changes.
- Update `CHANGELOG.md` when the change is user-visible or affects behavior.
- Keep entries concise and in Chinese.

5) Update CHANGELOG (when applicable)
- When you are making a commit in this workflow, add a new entry to `CHANGELOG.md` under `## Unreleased` for this change if it's user-visible or affects behavior.
- Keep the entry concise and in Chinese. Format: `- <change summary>`.
- Optionally, you may include the commit id if it's already available (e.g., when referencing an existing commit). Format: `- <change summary>（abc1234）`.
- Note: Commit IDs can be added retroactively in batch updates, so don't create a second commit just to add the ID.

6) Write the commit message (required)
- The commit message MUST be in Chinese.
- Follow the commit message template at `.gitmessage` (type(scope): summary + body + optional footer).
- Summary guidance: <= 50 chars (as per template). Body lines: wrap at <= 72 chars (as per template).
- Explain WHAT changed and WHY; include HOW only when it is necessary for future maintainers.
- Include a “测试” section stating how the change was validated.
- If breaking changes exist, add a footer line: `BREAKING CHANGE: ...` (English keyword, Chinese description is ok).
- If relevant, reference issues/PRs in the footer (e.g., `Issue: #123`, `PR: #456`).

7) Commit
- Stage only the relevant files (avoid unrelated noise).
- Commit to the current branch.

Output requirements:
- Output ONLY the final commit message text (no extra explanation), formatted to match the `.gitmessage` sections.
