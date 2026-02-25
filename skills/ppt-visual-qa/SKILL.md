---
name: ppt-visual-qa
description: "Unified quality assurance gate system for PPT HTML slides — validates presentation quality with 80+ gates, supports draft/production modes, and provides automated fallback repair sequences."
metadata:
  version: 1.0.0
  author: ppt-visual-qa
---

# ppt-visual-qa

Unified quality assurance gate system for PPT HTML slides.

## Overview

This skill provides a comprehensive QA system for validating PPT HTML presentations. It merges delivery gates and quality checklist into a single check matrix with draft-mode degradation support.

## Key Features

| Feature | Description |
|---------|-------------|
| **80+ Gates** | Comprehensive check matrix covering structural, content, chart, layout, brand, and visual categories |
| **Draft Mode** | Fast validation (~16 gates) for draft content |
| **Production Mode** | Full validation (all gates) for release-ready content |
| **Auto-Fallback** | 8-step repair sequence for automated fixes |
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
    ├── fallback.yml      # Auto-fallback sequence
    └── quality_rules.yml # Quality constraints
```

## Execution Commands

Run the QA script from the project root:

```bash
# Production mode (full validation - required for release)
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
| `--strict` | No | False | Exit with code 1 if any gate fails |
| `--allow-unimplemented` | No | False | Don't fail on `not_implemented` gates |
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
| **Production** | All 80+ | 2 | Release validation |
| **Draft** | ~16 (structural only) | 1 | Quick validation |

## Gate Categories

- **Structural** (13 gates): HTML skeleton, chart initialization
- **Content** (4 gates): Text density, three-part insights
- **Chart** (17 gates): Chart.js/ECharts configuration
- **Layout** (25 gates): Multi-layout specific checks
- **Visual** (12 gates): Color, spacing, overflow
- **Brand** (3 gates): Brand consistency
- **Post-generation** (11 gates): Runtime validation

## Integration

- **Owner**: `ppt-html-generator`
- **Dependencies**: `ppt-brand-system` (for brand consistency gates)

## Reference

See [references/reference.md](references/reference.md) for detailed gate definitions, metrics, and fallback sequences.

## Changelog

- **2026-02-21** (v1.1.0): Refactored to match Agent Skills spec
  - Renamed to SKILL.md per specification
  - Moved executable to scripts/run_visual_qa.py
  - Split gates into assets/gates.yml
  - Added references/ and examples/
  
- **2026-02-15** (v1.0.0): Initial release
