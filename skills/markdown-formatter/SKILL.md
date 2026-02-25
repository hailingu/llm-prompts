---
name: markdown-formatter
description: Automated validation, reporting, and fixing for Markdown documents using markdownlint, prettier and md_table_tool.
metadata:
  version: 1.0.0
  author: markdown-writer-specialist
---

# Markdown Formatter

## Overview

This skill provides automated validation, reporting, and fixing for Markdown documents. It is intended to be invoked by agents (notably `markdown-writer-specialist`) and via CLI to ensure consistent, lint-compliant Markdown.

## When to Use This Skill

- Validate Markdown files for syntax and style issues
- Auto-fix fixable markdownlint rules
- Fix table alignment issues
- Ensure consistent formatting across documentation

## Supported Tools

| Tool | Purpose | Command |
|------|---------|---------|
| `markdownlint-cli` | Lint and auto-fix Markdown | `npx markdownlint-cli` |
| `prettier` | Reflow paragraphs and format | `npx prettier --write` |
| `md_table_tool.py` | Fix table alignment | `python3 skills/md-table-fixer/bin/md_table_tool.py` |

## Usage

### CLI Commands

```bash
# Lint (report only)
npx markdownlint-cli <file_or_directory>

# Auto-fix markdownlint fixable issues
npx markdownlint-cli --fix <file_or_directory>

# Reflow paragraphs using prettier (recommended for MD013)
npx prettier --write --prose-wrap always <file>

# Detect table issues
python3 skills/md-table-fixer/bin/md_table_tool.py detect <file_or_directory>

# Fix table alignment
python3 skills/md-table-fixer/bin/md_table_tool.py fix <file_or_directory>
```

### Agent Workflow

1. Agent generates or updates document to `draft.md`
2. Run `lint` command. If exit_code != 0 then:
   - Run `fix` command
   - Run `table_fix` to correct tables
   - Re-run `lint` to gather remaining issues
3. If issues remain, include `remaining_errors` in a `partial` response to the user

## Supported Rules

### Auto-fixable (via `--fix`)

- MD009: Trailing spaces
- MD010: Hard tabs
- MD012: Multiple consecutive blank lines
- MD022: Headers should be surrounded by blank lines
- MD032: Lists should be surrounded by blank lines
- MD047: Files should end with a single newline character

### Manual or Tool-aided

- MD001: Heading levels
- MD013: Line length — use `prettier`
- MD060: Table alignment — use `md_table_tool.py`

## Common Issues & Fixes

### MD022: Headers should be surrounded by blank lines

**Before:**
```markdown
# Title
Content here
```

**After:**
```markdown
# Title

Content here
```

### MD032: Lists should be surrounded by blank lines

**Before:**
```markdown
Text
- Item 1
- Item 2
More text
```

**After:**
```markdown
Text

- Item 1
- Item 2

More text
```

### MD013: Line length

Use prettier to reflow:
```bash
npx prettier --write --prose-wrap always <file>
```

### MD060: Table alignment

Use md_table_tool.py:
```bash
python3 skills/md-table-fixer/bin/md_table_tool.py fix <file>
```

## Return Payloads

The skill returns machine-friendly YAML/JSON payloads matching the following schemas:

### Success
```yaml
status: success
errors_before: 8
errors_after: 0
actions_taken:
  - "markdownlint --fix"
```

### Partial
```yaml
status: partial
errors_before: 12
errors_after: 4
remaining_errors:
  - line: 10
    rule: MD013
    message: "Line length: 200 (max 120)"
actions_taken:
  - "markdownlint --fix"
```

### Error
```yaml
status: error
error_message: "Configuration file not found"
hint: "Check .markdownlint.json exists"
```

## Troubleshooting & Limitations

- Not all markdownlint rules are auto-fixable; complex structural issues require human review
- `prettier` reflow may alter semantics in edge cases — review changes before committing
- Table alignment for CJK needs display-width aware tools (provided by `md_table_tool.py`)

## Integration

This skill is primarily used by `markdown-writer-specialist` to validate and fix generated Markdown documents.

## Dependencies

- Node.js and `markdownlint-cli` (npm package)
- `prettier` (optional but recommended)
- Python 3.8+ and `skills/md-table-fixer/bin/md_table_tool.py`

## Configuration

Use `.markdownlint.json` in repo root to control lint rules.

## Changelog

- **2026-02-05** — v1.0.0 — Initial release with manifest, commands and examples
