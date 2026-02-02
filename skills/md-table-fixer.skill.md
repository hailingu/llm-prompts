# MD Table Fixer Skill

**Purpose:**
Provide an automated, display-width-aware toolset for detecting and fixing
Markdown table alignment issues (MD060). This skill is intended to be used
by the `markdown-writer-specialist` agent (and other agents) to reliably
find and fix tables that are visually misaligned, including tables with
CJK (Chinese/Japanese/Korean) characters.

Commands (via CLI):
- `python3 tools/md_table_tool.py detect <path>` — detect misaligned tables
- `python3 tools/md_table_tool.py fix <path>` — fix (and backup) misaligned tables

Behavior & guarantees:
- Detects table blocks and merges common accidental line breaks in table
  headers/rows before analyzing.
- Uses Unicode/East Asian width to compute display widths (so CJK cells are
  padded correctly) and rebuilds the table in aligned style.
- Creates a `.bak` backup for files that were modified.

Agent integration guidance:
- Run `detect` first to report problems and decide whether to `fix`.
- After `fix`, run `npx markdownlint-cli <file>` to validate other markdown rules.
- For batch runs, pass a directory path (the tool will recurse `**/*.md`).

Example usage in agent flow:
1. `run_in_terminal: python3 tools/md_table_tool.py detect templates/` (collect issues)
2. If issues found: `run_in_terminal: python3 tools/md_table_tool.py fix templates/`
3. `run_in_terminal: npx markdownlint-cli templates/` (re-validate)

Notes:
- The tool aims to preserve non-table content and to only modify table
  blocks when necessary.
- For unusual cases, manual review is recommended after `fix`.
