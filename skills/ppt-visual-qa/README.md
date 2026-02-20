# ppt-visual-qa

Unified quality assurance gate system for PPT HTML slides.

## Runtime Requirements

This skill handles visual regression testing and requires a browser engine:

```bash
# Ensure you are in the virtual environment
source .venv/bin/activate

# Python dependencies
pip install playwright beautifulsoup4

# Browser binaries (required for layout verification)
playwright install chromium
```

## Overview

This skill merges the former **交付门禁** (43 items) and **质量检查清单** (~80 items) from the `ppt-html-generator` agent into a **single check matrix** (`gates.yml`) with 78 unique gates. Each gate is tagged with metadata to enable mode-aware checking.

### Key Improvements

| Problem | Solution |
|---------|----------|
| **C3**: Draft mode triggered all 43 gates → infinite retry loops | Added `draft_skip` flag; draft mode checks only ~16 structural gates |
| **R6**: Gates & checklist duplicated the same rules in 2 forms | Merged into single matrix, each rule appears once |
| **R1**: "洞察三段式" repeated 5 times across agent | Defined once as G17, referenced by ID |
| **R2**: "Left/right fill rate ≥ X%" repeated 6 times | Consolidated into G36/G37 with layout-specific variants in scope |
| **R4**: "overflow-auto handling" repeated 3 times | Consolidated into G61 |

## File Structure

```
skills/ppt-visual-qa/
├── manifest.yml      # Skill metadata
├── gates.yml         # Single source of truth: 78 gates + fallback sequence
├── run_visual_qa.py  # Executable gate-runner (HTML visual QA)
├── run_pptx_qa.py    # High-level PPTX QA scorer
├── run_auto_fix_qa.py # Auto-fix QA loop for semantic/design JSON
├── validate_pptx_and_write_report.py # MR validator for generated PPTX
└── README.md         # This file
```

## Gate Schema

Each gate in `gates.yml` has:

| Field | Values | Description |
|-------|--------|-------------|
| `id` | G01–G78 | Unique identifier |
| `condition` | string | Human-readable check condition |
| `phase` | `pre` / `during` / `post` | When to check |
| `level` | `block` / `warn` | `block` = must fix before delivery; `warn` = flag but allow |
| `draft_skip` | `true` / `false` | If `true`, skipped in draft mode |
| `scope` | `all` / layout type | Which pages this gate applies to |
| `category` | `structural` / `content` / `chart` / `layout` / `brand` / `visual` | For grouping and reporting |

## Draft vs Production

| Mode | Gates Checked | Max Retries | On Final Fail |
|------|--------------|-------------|---------------|
| **Production** (成片模式) | All 78 | 2 | Output failure reason + suggest manual fix |
| **Draft** (草稿模式) | ~16 (draft_skip: false) | 1 | Mark Draft + list failures, allow delivery |

## Auto-Fallback Sequence

When blocking gates fail in production mode, fixes are attempted in this order:

1. Data mapping (bubble radius recalc)
2. Axis correction (suggestedMax / stepSize / clip check)
3. Copy enhancement (add structured claims, three-part insight)
4. Vertical budget fix (main height / overflow strategy / footer safety)
5. Full-width local fix (reduce chart height → expand budget → compress KPI)
6. Total budget fix (reduce fixed-height items)
7. Layout reflow (chart height / card density / layout switch)
8. Max 2 retries → report failure with manual intervention points

## Usage

In the agent file, reference with:
```
> 交付门禁与质量检查见 `skills/ppt-visual-qa/gates.yml`。
> 草稿模式仅检查 `draft_skip: false` 的 gate；成片模式检查全部。
> 自动回退顺序见 `gates.yml → fallback_sequence`。
```

### Run executable gate-runner

```bash
python skills/ppt-visual-qa/run_visual_qa.py \
	--presentation-dir docs/presentations/ai-report\ Bain-style_20260216_v1 \
	--mode production \
	--strict
```

If you want strict checking while tolerating unfinished checker mappings:

```bash
python skills/ppt-visual-qa/run_visual_qa.py \
	--presentation-dir docs/presentations/ai-report\ Bain-style_20260216_v1 \
	--mode production \
	--strict \
	--allow-unimplemented
```

Output:
- `${presentation_dir}/qa/layout-runtime-report.json` (fixed path and filename)
- Per-slide, per-profile metrics: `footer_safe_gap`, `overflow_nodes`, `m01..m07`, `pass/fail`
- Per-slide gate results for all configured gates (status: `pass|fail|not_applicable|skipped_mode|not_implemented`)

Notes:
- This runner now loads the complete gate catalog from `gates.yml`.
- Gates without checker mapping are explicitly marked `not_implemented` (never silently treated as pass).

## Release Gate (Must Pass)

Production delivery is blocked unless QA passes with strict mode.

Mandatory pre-release command:

```bash
python skills/ppt-visual-qa/run_visual_qa.py \
	--presentation-dir docs/presentations/<topic>_<YYYYMMDD>_v<N> \
	--mode production \
	--strict
```

Release policy:
- Exit code must be `0`; non-zero means **do not publish**.
- Do not use `--allow-unimplemented` for release.
- Any of the following means **not releasable**: `failed_slides > 0`, any `block` gate=`fail`, or `not_implemented > 0`.

Runtime backend behavior:
- If `playwright` is installed, runs browser-semantic checks across 3 profiles: `1280x720@1x`, `1366x768@1x`, `1512x982@2x`
- If not installed, falls back to static estimator and marks backend as `static-fallback`

## QA Report Path Contract (Mandatory)

- Canonical report path is fixed to: `${presentation_dir}/qa/layout-runtime-report.json`
- `run_visual_qa.py` enforces this contract and rejects non-canonical `--report-out` values.
- Agent repair loop must read this canonical report before each fix iteration (no blind edits).
- Delivery is invalid if QA output exists only at non-canonical paths.

Example check command:

```bash
python skills/ppt-visual-qa/run_visual_qa.py \
	--presentation-dir docs/presentations/ai-report\ Bain-style_20260216_v1 \
	--mode production \
	--strict
```

Optional install for runtime backend:

```bash
pip install playwright
python -m playwright install chromium
```

### PPTX QA scripts (co-located in this skill)

```bash
python skills/ppt-visual-qa/run_pptx_qa.py --out-dir docs/presentations/mft-20260206
python skills/ppt-visual-qa/validate_pptx_and_write_report.py <pptx> <semantic.json> <design.json> <out_report.json>
python skills/ppt-visual-qa/run_auto_fix_qa.py
```

## Integration

- **Owner**: `ppt-html-generator`
- **Dependencies**: `ppt-brand-system` (for brand consistency gates G68-G70)
