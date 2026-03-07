# ppt-visual-qa Reference

Detailed reference documentation for the QA-assisted diagnostic system.

## Gate System

The gate system reviews PPT HTML slides across multiple categories.
It is intended to surface structural, runtime, and readability risks, not to override upstream PPT contracts.

### Gate Categories

- Structural: 13 checks covering HTML skeleton, script tags, and CDN links
- Content: 4 checks covering text density and structured insights
- Chart: 17 checks covering Chart.js and ECharts configuration
- Layout: 25 checks covering multi-layout specific checks
- Visual: 12 checks covering color, spacing, and overflow
- Brand-Style: 3 checks covering brand-style consistency
- Post-generation: 11 checks covering runtime validation

Interpretation rule:

- `block`: high-signal structural/runtime/readability defect that usually merits repair
- `warn`: review with design judgment; can still be deliverable
- `info`: advisory signal only

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

Runtime interpretation notes:

- A small overflow delta on the root `.grid`, `.flex`, or `.h-full` page wrapper can come from browser layout rounding and should be treated differently from nested component overflow.
- Runtime budget checks should be read with a small tolerance before being escalated into a hard redesign request.
- `overflow_nodes` is most useful when it points to nested cards, tables, charts, or masked containers, not when it only points to the top-level page grid.

### Heuristic Boundaries

These QA checks intentionally use design-aware heuristics rather than literal string matching only.

#### Brand-Style Gates

- Validate the active brand/style scope used on the current slide.
- Theme CSS may be loaded from the presentation root, `assets/slide-theme.css`, or `design/slide-theme.css`.
- Do not require every supported brand archetype to coexist in a single deck theme file.

#### Semantic Color Gates

- Treat semantic color evidence broadly: badge classes, semantic CSS variables, border-left accents, map markers, and map annotations can all satisfy semantic-color intent.
- Do not assume semantic meaning appears only through utility classes like `text-red-*` or `bg-green-*`.
- On map-led pages, semantic overlays and annotated markers are valid semantic-color landing points.

#### Timeline Gates

- Do not infer timeline layout from `.connection-line` alone.
- Require stronger timeline evidence such as `timeline-item`, `timeline-track`, `timeline-node` with anchor semantics, or explicit year/phase structure.
- Hub-and-spoke, radial, and systems diagrams may reuse line connectors without being timeline pages.

#### Chart Defensive Gates

- `G07` is intended to catch chart code that assumes labels and data arrays are already aligned.
- If a chart explicitly trims, slices, or otherwise normalizes labels and dataset lengths before render, treat that as real defensive handling even when the literal string `labels.length` is absent.
- If a page has no explicit alignment guard and no normalization step, keep `G07` as a high-signal chart defect.

These metrics should be read together with upstream contracts from:

- `ppt-slide-layout-library`
- `ppt-chart-engine`
- `ppt-map-storytelling`
- `ppt-component-library`

## Recovery Guidance

When a gate fails, use the QA result as a diagnosis signal, then repair in this order:

1. Apply the chosen layout asset's `overflow_recovery_order` if one exists
2. Apply chart-contract or map-overlay recovery while the layout remains valid
3. Reduce local component or copy density only after upstream recovery has been tried
4. Switch layouts only if `fallback_layouts` explicitly allow it
5. Escalate to manual redesign when contract-guided recovery is exhausted

## Output Reports

- `layout-runtime-report.json` - Per-slide, per-profile metrics
- `gate-report.json` - Gate pass/fail status
- `summary.json` - Aggregate results

Report interpretation:

- A failing report is a debugging input, not an automatic delivery veto.
- A passing report is useful evidence, not proof that the page is strategically optimal.
- When a single gate fails, check whether the failure comes from a real layout/runtime defect or from an outdated heuristic assumption before redesigning the page.
