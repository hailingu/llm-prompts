# ppt-visual-qa

Unified quality assurance gate system for PPT HTML slides.

## Overview

This skill merges the former **交付门禁** (43 items) and **质量检查清单** (~80 items) from the `ppt-html-generator` agent into a **single check matrix** (`gates.yml`) with 74 unique gates. Each gate is tagged with metadata to enable mode-aware checking.

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
├── gates.yml         # Single source of truth: 74 gates + fallback sequence
└── README.md         # This file
```

## Gate Schema

Each gate in `gates.yml` has:

| Field | Values | Description |
|-------|--------|-------------|
| `id` | G01–G74 | Unique identifier |
| `condition` | string | Human-readable check condition |
| `phase` | `pre` / `during` / `post` | When to check |
| `level` | `block` / `warn` | `block` = must fix before delivery; `warn` = flag but allow |
| `draft_skip` | `true` / `false` | If `true`, skipped in draft mode |
| `scope` | `all` / layout type | Which pages this gate applies to |
| `category` | `structural` / `content` / `chart` / `layout` / `brand` / `visual` | For grouping and reporting |

## Draft vs Production

| Mode | Gates Checked | Max Retries | On Final Fail |
|------|--------------|-------------|---------------|
| **Production** (成片模式) | All 74 | 2 | Output failure reason + suggest manual fix |
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

## Integration

- **Owner**: `ppt-html-generator`
- **Dependencies**: `ppt-brand-system` (for brand consistency gates G68-G70)
