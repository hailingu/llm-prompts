---
name: ppt-visual-qa
description: "Unified quality assurance gate system for PPT HTML slides — validates presentation quality with 80+ gates, supports draft/production modes, and provides automated fallback repair sequences."
metadata:
  version: 1.0.0
  author: ppt-visual-qa
---

# ppt-visual-qa

Auxiliary visual QA and runtime diagnosis system for PPT HTML slides.

## Overview

This skill provides a QA-assisted diagnosis system for validating PPT HTML presentations. It merges structural/runtime checks and visual review heuristics into a single check matrix, but it does not replace page design judgment, Thinking decisions, or layout/chart/map/component contracts.

Use this skill to surface real page risks.
Do not use it as a mechanical release gate that overrides the PPT specialist workflow.

Operational heuristics:

- Mild runtime overflow on a root `.grid` or `.h-full` wrapper can be browser rounding noise, not a real page break.
- Semantic color can be expressed through badge classes, CSS variables, or map-overlay annotations, not only utility-class tokens.
- `.connection-line` alone is not enough to infer a timeline layout; hub-and-spoke pages may use the same primitive.
- Brand-style validation should confirm the active scope used by the current slide and probe common theme locations such as `assets/slide-theme.css`.

## Key Features

| Feature | Description |
|---------|-------------|
| **80+ Gates** | Comprehensive check matrix covering structural, content, chart, layout, brand, and visual categories |
| **Draft Mode** | Fast validation (~16 gates) for draft content |
| **Production Mode** | Full validation (all gates) for deeper diagnosis |
| **Recovery Hints** | Gate output can inform repair order, but does not replace layout contracts |
| **Runtime Profiles** | Tests across 3 viewport+DPR combinations |

## Directory Structure

```
ppt-visual-qa/
├── SKILL.md              # Core skill definition (required)
├── scripts/
│   └── run_visual_qa.py  # Main executable
├── references/
│   └── reference.md      # Detailed reference docs
├── examples/
│   └── examples.md       # Test examples
└── assets/
    ├── gates.yml         # Gate definitions
    ├── metrics.yml       # Executable metrics
    ├── fallback.yml      # Recovery guidance sequence
    └── quality_rules.yml # Quality constraints
```

## Execution Commands

Run the QA script from the project root:

```bash
# Production mode (full validation - optional deeper diagnosis)
python3 skills/ppt-visual-qa/scripts/run_visual_qa.py \
    --presentation-dir <path-to-presentation> \
    --mode production \
    --strict

# Draft mode (fast validation, ~16 structural gates only)
python3 skills/ppt-visual-qa/scripts/run_visual_qa.py \
    --presentation-dir <path-to-presentation> \
    --mode draft

# Check specific slides only (incremental update)
python3 skills/ppt-visual-qa/scripts/run_visual_qa.py \
    --presentation-dir <path-to-presentation> \
    --mode production \
    --slides 4 5 6

# Check specific gates only
python3 skills/ppt-visual-qa/scripts/run_visual_qa.py \
    --presentation-dir <path-to-presentation> \
    --mode production \
    --gates G06 G07 G08 G09

# Combine slide and gate filters for targeted testing
python3 skills/ppt-visual-qa/scripts/run_visual_qa.py \
    --presentation-dir <path-to-presentation> \
    --mode production \
    --slides 4 \
    --gates G10 G11 G12
```

## CLI Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--presentation-dir` | Yes | - | Path to presentation directory containing `slide-*.html` files |
| `--mode` | No | `production` | QA mode: `draft` (fast, ~16 gates) or `production` (all gates) |
| `--strict` | No | False | Exit with code 1 only for blocking defects, and optionally blocking `not_implemented` coverage gaps |
| `--allow-unimplemented` | No | False | Do not make strict mode fail on blocking `not_implemented` coverage gaps |
| `--slides` | No | All | Specific slide numbers to check (e.g., `1 2 5`). Performs incremental report update |
| `--gates` | No | All | Specific gate IDs to check (e.g., `G01 G02 G11`). Case-insensitive |
| `--report-out` | Deprecated | - | Report path is fixed to `<presentation-dir>/qa/layout-runtime-report.json` |

## Filter Behavior

| `--slides` | `--gates` | Behavior |
|------------|-----------|----------|
| Not set | Not set | **Full scan**: All slides × all applicable gates |
| Set | Not set | **Slide filter**: Specified slides × all applicable gates |
| Not set | Set | **Gate filter**: All slides × specified gates only |
| Set | Set | **Targeted**: Specified slides × specified gates only |

## Runtime Requirements

```bash
# Ensure you are in the virtual environment
source .venv/bin/activate

# Python dependencies
pip install playwright beautifulsoup4 pyyaml

# Browser binaries (required for layout verification)
playwright install chromium
```

## Output Modes

| Mode | Gates Checked | Max Retries | Use Case |
|------|--------------|-------------|----------|
| **Production** | All 80+ | 2 | Deeper diagnosis across structure, runtime, and visual heuristics |
| **Draft** | ~16 (structural only) | 1 | Quick validation while building or revising slides |

## Operating Position

This skill is an auxiliary reviewer, not the delivery authority.

Priority order when this skill is used with the PPT system:

1. `ppt-specialist` workflow and Thinking decisions
2. `ppt-slide-layout-library` layout contract and recovery order
3. `ppt-chart-engine` chart contract and chart-level fallback
4. `ppt-map-storytelling` map narrative and overlay routing decisions
5. `ppt-component-library` semantic resolver and component skeleton rules
6. `ppt-visual-qa` gate output as diagnosis and repair guidance

If a QA hint conflicts with a verified layout contract, chart contract, or map-routing decision, prefer the upstream contract and update QA expectations rather than forcing the page into a worse design.

`block` means the page likely has a real structural, runtime, or readability defect that should usually be repaired.
`warn` means the page may still be deliverable and should be reviewed with design judgment.
`info` means the signal is advisory only.

Do not equate `--strict` failure with non-deliverability unless the failing gates describe actual user-visible breakage.

## Contract Alignment

When used with the current PPT workflow, QA should respect these upstream fields when they exist in Thinking or layout assets:

- `layout_contract_source`
- `overflow_recovery_order`
- `fallback_layouts`
- `contract_source`
- `resolver_source`

Operational rule:

1. If a page budget issue can be corrected by the chosen layout asset's `overflow_recovery_order`, do that before changing the layout because of a QA complaint.
2. If a chart can be stabilized by its declared chart contract and the layout remains valid, do that before degrading to cards or tables.
3. If a map overlay remains legible after layout-contract recovery, do not force a non-map rewrite merely because a generic density gate complains.
4. If a component block is semantically correct and the page remains stable, do not replace it just to satisfy a low-signal cosmetic gate.

## Gate Categories

- **Structural** (13 gates): HTML skeleton, chart initialization
- **Content** (4 gates): Text density, three-part insights
- **Chart** (17 gates): Chart.js/ECharts configuration
- **Layout** (25 gates): Multi-layout specific checks
- **Visual** (12 gates): Color, spacing, overflow
- **Brand-Style** (3 gates): Brand-style consistency
- **Post-generation** (11 gates): Runtime validation

Interpretation notes:

- Structural and runtime gates are the highest-signal checks in this skill.
- Content-density and deck-rhythm gates are heuristic and should not override page-specific reasoning by themselves.
- Layout-specific gates should be read together with `ppt-slide-layout-library` contracts, not in isolation.
- Runtime overflow gates should distinguish nested component breakage from mild root-layout rounding drift.
- Timeline gates should only fire when timeline structure is real, not when a radial or hub-and-spoke diagram reuses connector primitives.
- Semantic-color gates should treat map markers, badges, and semantic CSS variables as valid design evidence when they carry analysis meaning.

## Integration

- **Owner**: `ppt-html-generator`
- **Dependencies**:
  - `ppt-brand-style-system` (for brand-style consistency gates)
  - `ppt-slide-layout-library` (for layout contract and recovery context)
  - `ppt-chart-engine` (for chart contract context)
  - `ppt-map-storytelling` (for map-page routing and overlay context)
  - `ppt-component-library` (for semantic component context)

## Reference

See [references/reference.md](references/reference.md) for detailed gate definitions, metrics, and fallback sequences.

## Changelog

- **2026-02-21** (v1.1.0): Refactored to match Agent Skills spec
  - Renamed to SKILL.md per specification
  - Moved executable to scripts/run_visual_qa.py
  - Split gates into assets/gates.yml
  - Added references/ and examples/
  
- **2026-02-15** (v1.0.0): Initial release
