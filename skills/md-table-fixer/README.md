# md-table-fixer Skill

## Purpose

`md-table-fixer` provides an automated, display-width-aware toolset for
detecting and fixing Markdown table alignment issues (MD060). It handles CJK
characters correctly and creates `.bak` files for any modified documents.

## Usage

CLI commands:

```bash
# Detect misaligned tables in a file or directory
python3 skills/md-table-fixer/bin/md_table_tool.py detect <path>

# Fix misaligned tables (creates .bak backups)
python3 skills/md-table-fixer/bin/md_table_tool.py fix <path>
```

## Behavior & guarantees

- Detects table blocks and merges common accidental line breaks in headers/rows
  before analyzing.
- Uses Unicode/East Asian width to compute display widths (so CJK cells are
  padded correctly) and rebuilds tables in an aligned style.
- Creates a `.bak` backup for files that were modified.

## Agent integration guidance

- Run `detect` first to report problems and decide whether to `fix`.
- After `fix`, run `npx markdownlint-cli <file>` to validate other markdown
  rules.
- For batch runs, pass a directory path (the tool will recurse `**/*.md`).

## Example agent flow

1. `run_in_terminal: python3 skills/md-table-fixer/bin/md_table_tool.py detect templates/`
2. If issues found:
   `run_in_terminal: python3 skills/md-table-fixer/bin/md_table_tool.py fix templates/`
3. `run_in_terminal: npx markdownlint-cli templates/`

## Notes

- The tool aims to preserve non-table content and only modifies table blocks
  when necessary.
- Manual review is recommended for unusual or ambiguous cases.

## Related files

- `tools/md_table_tool.py` — implementation
- `skills/md-table-fixer/manifest.yml` — skill manifest (machine-readable)

## Changelog

- 2026-02-05 — v1.0.0 — Initial directory-based skill and docs.
