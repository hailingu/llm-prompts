# markdown-formatter Skill

## Purpose

This skill provides automated validation, reporting, and fixing for Markdown
documents. It is intended to be invoked by agents (notably
`markdown-writer-specialist`) and via CLI to ensure consistent, lint-compliant
Markdown using these tools:

- `markdownlint-cli`
- `prettier`
- `md_table_tool.py`

## Core features

- Format validation (lint-only run)
- Automatic fixes for fixable markdownlint rules
- Table alignment fixes using `md_table_tool.py`
- Machine-friendly outputs for agent workflows and automation

## Usage

CLI commands (preferred for automation):

```bash
# Lint (report only)
npx markdownlint-cli <file_or_directory>

# Auto-fix markdownlint fixable issues
npx markdownlint-cli --fix <file_or_directory>

# Reflow paragraphs using prettier (recommended for MD013)
npx prettier --write --prose-wrap always <file>

# Detect table issues (md_table_tool.py)
python3 skills/md-table-fixer/bin/md_table_tool.py detect <file_or_directory>

# Fix table alignment
python3 skills/md-table-fixer/bin/md_table_tool.py fix <file_or_directory>
```

Agent workflow (example: `markdown-writer-specialist`)

1. Agent generates or updates document to `draft.md`.
2. Run `lint` command. If exit_code != 0 then:
   - Run `fix` command.
   - Run `table_fix` to correct tables.
   - Re-run `lint` to gather remaining issues.
3. If issues remain, include `remaining_errors` in a `partial` response to the
   user.

## Supported rules

Auto-fixable (via `--fix`): MD009, MD010, MD012, MD022, MD032, MD047.

Manual or tool-aided: MD001 (heading levels), MD013 (line length — use
`prettier`), MD060 (table alignment — `md_table_tool.py`).

## Common issues & fixes

Examples and fixes are included in `examples.yml` and demonstrate typical
failing and passing cases (MD022, MD032, MD013, MD060).

## Toolchain & configuration

- Node.js and `markdownlint-cli` (npm package)
- `prettier` (optional but recommended)
- Python 3.8+ and `tools/md_table_tool.py`

Use `.markdownlint.json` in repo root to control lint rules (example provided in
the repo).

## Return payloads

The skill returns machine-friendly YAML/JSON payloads matching the schemas in
`manifest.yml` and `commands.yml` (status: `success|partial|error`).

## Troubleshooting & limitations

- Not all markdownlint rules are auto-fixable; complex structural issues require
  human review.
- `prettier` reflow may alter semantics in edge cases — review changes before
  committing.
- Table alignment for CJK needs display-width aware tools (provided by
  `md_table_tool.py`).

## Related files

- `manifest.yml` — machine-readable manifest
- `commands.yml` — structured command definitions
- `examples.yml` — testable examples

## Changelog

- 2026-02-05 — v1.0.0 — Split into directory, added manifest, commands and
  examples.
