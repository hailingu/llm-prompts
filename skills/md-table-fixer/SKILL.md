---
name: md-table-fixer
description: Detect and fix Markdown table alignment issues (MD060) with display-width awareness for CJK characters.
metadata:
  version: 1.0.0
  author: markdown-writer-specialist
---

# Markdown Table Fixer

## Overview

This skill provides an automated, display-width-aware toolset for detecting and fixing Markdown table alignment issues (MD060). It handles CJK characters correctly and creates `.bak` files for any modified documents.

## When to Use This Skill

- Detect misaligned Markdown tables
- Fix table alignment issues (MD060)
- Handle CJK characters in tables correctly
- Ensure consistent table formatting across documentation

## Supported Commands

| Command | Purpose | Exit Codes |
|---------|---------|------------|
| `detect` | Detect misaligned tables | 0: no issues, 1: issues found, 2: error |
| `fix` | Fix misaligned tables | 0: fixed/none, 1: some failed, 2: error |

## Usage

### CLI Commands

```bash
# Detect misaligned tables in a file or directory
python3 skills/md-table-fixer/scripts/md_table_tool.py detect <path>

# Fix misaligned tables (creates .bak backups)
python3 skills/md-table-fixer/scripts/md_table_tool.py fix <path>
```

### Agent Workflow

1. Run `detect` first to report problems and decide whether to `fix`
2. After `fix`, run `npx markdownlint-cli <file>` to validate other markdown rules
3. For batch runs, pass a directory path (the tool will recurse `**/*.md`)

**Example agent flow:**

```bash
# Step 1: Detect issues
python3 skills/md-table-fixer/scripts/md_table_tool.py detect templates/

# Step 2: Fix if issues found
python3 skills/md-table-fixer/scripts/md_table_tool.py fix templates/

# Step 3: Validate with markdownlint
npx markdownlint-cli templates/
```

## Behavior & Guarantees

- Detects table blocks and merges common accidental line breaks in headers/rows before analyzing
- Uses Unicode/East Asian width to compute display widths (so CJK cells are padded correctly) and rebuilds tables in an aligned style
- Creates a `.bak` backup for files that were modified
- Preserves non-table content and only modifies table blocks when necessary

## Example

### Before (Misaligned)

```markdown
| Name | Description | Status |
|------|----|----------|
| Item 1 | Short | Active |
| Item 2 | A much longer description here | Pending |
```

### After (Fixed)

```markdown
| Name   | Description                  | Status  |
|--------|------------------------------|---------|
| Item 1 | Short                        | Active  |
| Item 2 | A much longer description here | Pending |
```

## Return Payloads

### Success
```yaml
status: success
file_path: docs/example.md
tables_fixed: 2
actions_taken:
  - "md_table_tool.py fix"
```

### Partial
```yaml
status: partial
file_path: docs/example.md
tables_fixed: 1
failed_tables:
  - "table at line 45"
actions_taken:
  - "md_table_tool.py fix"
```

### Error
```yaml
status: error
error_message: "Path not found: /invalid/path"
hint: "Check the file or directory path exists"
```

## Troubleshooting & Limitations

- Manual review is recommended for unusual or ambiguous cases
- The tool aims to preserve non-table content and only modifies table blocks when necessary
- Complex nested tables may require manual adjustment

## Integration

This skill is primarily used by:
- `markdown-writer-specialist` — to fix table alignment in generated documents
- `markdown-formatter` — as part of the comprehensive markdown formatting workflow

## Dependencies

- Python 3.8+
- No external packages required (uses standard library only)

## Changelog

- **2026-02-05** — v1.0.0 — Initial release with detect and fix commands
