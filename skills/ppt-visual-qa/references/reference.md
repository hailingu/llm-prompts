# ppt-visual-qa Reference

Detailed reference documentation for the QA gate system.

## Gate System

The gate system validates PPT HTML slides across multiple categories:

### Gate Categories

| Category | Count | Description |
|----------|-------|-------------|
| Structural | 13 | HTML skeleton, script tags, CDN links |
| Content | 4 | Text density, three-part insights |
| Chart | 17 | Chart.js/ECharts configuration |
| Layout | 25 | Multi-layout specific checks |
| Visual | 12 | Color, spacing, overflow |
| Brand | 3 | Brand consistency |
| Post-generation | 11 | Runtime validation |

### Gate Definition Format

```yaml
gate_id: G01
category: structural
description: Check for HTML skeleton
check_type: static
selector: html
expected: present
```

## Metrics

### Runtime Metrics (M01-M08)

- **M01**: scroll_gap - Main container scroll overflow
- **M02**: visual_safe_gap - Footer collision risk
- **M03**: overflow_nodes - Hidden overflow masking
- **M04**: stack_risk - Content stack overflow
- **M05**: visual_collapse - Minimum height thresholds
- **M06**: content_bottom_max - Max bottom position
- **M07**: footer_top - Footer top position
- **M08**: backend - Runtime backend type

## Fallback Sequence

When a gate fails, the system attempts fixes in order:

1. Spacing adjustment (gap/padding reduction)
2. Font size reduction
3. Content compression
4. Layout type change
5. Component replacement
6. Data filtering
7. Visualization transformation
8. Full regeneration

## Output Reports

- `layout-runtime-report.json` - Per-slide, per-profile metrics
- `gate-report.json` - Gate pass/fail status
- `summary.json` - Aggregate results
