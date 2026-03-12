---
name: ppt-chart-engine
description: Chart engine for PPT HTML slides - chart type catalog, selection algorithm, semantic mapping, data contracts, rendering constraints (Bubble/Line/Heatmap), container height contract, data guards, and code examples.
metadata: 
  - version: 1.0.0
  - author: ppt-chart-engine
---

# PPT Chart Engine

## Overview

This skill provides a chart decision, rendering constraint, and data contract system for PPT HTML slides, supporting both Chart.js and ECharts engines.

`SKILL.md` is the human-facing decision guide. `assets/charts.yml` is the detailed rulebook for chart types, mappings, constraints, and examples. If they ever drift, `assets/charts.yml` should be treated as the more specific reference and `SKILL.md` should be updated to match.

For budget-sensitive layouts, `assets/charts.yml#chart_candidate_generation_examples` now provides worked examples showing how to go from region budget to chart-component candidates to final selection.

## When to Use This Skill

- Select appropriate chart type for data
- Apply chart rendering constraints
- Handle data contracts and mappings
- Implement data guard rules
- Write chart code examples
- Generate chart-component candidates inside a pre-decided layout region

## Thinking Alignment

For chart-led slides, pair this skill with:

- `templates/ppt-slide-thinking-template.md` for the base Thinking structure
- `templates/ppt-chart-thinking-examples.md` for chart-first worked examples

Before implementation, the Thinking file should explicitly declare:

1. `chart_family`
2. `contract_fields`
3. `null_policy`
4. `contract_source`
5. `fallback_plan`

If the chart lives inside a standard layout, the Thinking file should also declare:

6. `layout_key`
7. `layout_contract_source`
8. `overflow_recovery_order`
9. `fallback_layouts`

Do not enter chart implementation until those fields are stable.

Use `assets/charts.yml#thinking_contracts` as the machine-readable source for these fields. The markdown examples are reference patterns; `thinking_contracts` is the tighter contract.

When chart and layout constraints interact, treat them in this order:

1. `ppt-slide-layout-library` decides whether the chart-heavy layout is valid for the reading task.
2. `layout_contract` decides the recovery order when the page budget becomes unstable.
3. `ppt-chart-engine` decides chart-family contract validity and chart-level fallback.

Chart fallback must not skip layout recovery. If the page can be stabilized by the chosen layout's `overflow_recovery_order`, do that before changing chart family or switching layouts.

## Decision Gate

Before selecting a chart, decide whether the page should use a chart at all.

1. If the insight is clearer as a timeline, process flow, matrix, KPI panel, or comparison cards, prefer layout over chart.
2. Use charts to compress relationships, not to decorate the page.
3. Avoid pie/donut charts when category count exceeds 5, labels are long, or comparisons are the real task.
4. Architecture, containment, and layered-system views should default to HTML+CSS structure, not chart libraries.
5. If the narrative depends on exact wording more than pattern recognition, prefer table/card expression over a chart.

## Geo Boundary

Geographic charts remain part of this skill only at the chart-encoding layer.

This skill owns:
- map / geo heatmap / geo scatter / geo lines as chart families
- geographic data contracts for value encoding
- legend, color scale, tooltip, and rendering constraints for geo charts

This skill does not own:
- regional crop decisions
- conflict/route/corridor storytelling grammar
- callout placement logic on map-heavy narrative pages
- map-first composition strategy

Use `ppt-map-storytelling` for those narrative concerns.
When a geo chart needs a basemap or registered geography source, follow the basemap source contract defined by `ppt-map-storytelling` instead of inventing a chart-local map source rule.
When that geo chart is hosted inside a standard layout such as `map_overlay` or `side_by_side`, also follow the chosen layout asset's `layout_contract` for recovery and fallback instead of inventing chart-local layout behavior.

## Chart Type Catalog

### Basic Chart Types

| Data Type | Chart Type | Use Cases |
|-----------|------------|-----------|
| Multi-dimensional comparison | Radar Chart | Technology maturity, capability assessment |
| Time trend | Line Chart | Performance metrics changes, market share |
| Categorical comparison | Bar Chart | Segment performance, product comparison |
| Proportional relationship | Pie/Donut Chart | Market share, cost structure |
| Geographic distribution | Map | Regional market analysis |

### Extended Chart Types

| Data Type | Chart Type | Engine | CDN |
|-----------|------------|--------|-----|
| Multi-variable relationship | Bubble Chart | Chart.js | Native |
| Dual-axis trend | Dual-axis Combo | Chart.js | Native |
| Time scheduling | Gantt Chart | ECharts | echarts@5 |
| Matrix data | Heatmap | ECharts | echarts@5 |
| Distribution analysis | Box Plot | Chart.js | @sgratzl/chartjs-chart-boxplot |
| Hierarchical structure | Treemap | Chart.js | chartjs-chart-treemap |
| Relationship network | Sankey Diagram | ECharts | echarts@5 |
| Geographic heatmap | Geo Heatmap | ECharts | echarts@5 |
| Architecture/Containment | Stack Diagram | HTML+CSS | None |

### Engine Constraints

- Chart types marked as "requires plugin" must include corresponding CDN in `<head>`
- ECharts types must use `<div>` container + `echarts.init()`
- HTML+CSS types use Flex/Grid layout to manually build DOM
- Geo chart inputs that depend on map sources should reuse `ppt-map-storytelling` basemap inputs (`SVG` / `GeoJSON` / `TopoJSON` / justified vector-tile source)

## Chart Selection Algorithm

### By Data Dimensions

| Dimensions | Recommended Chart |
|------------|-------------------|
| 1D | Pie Chart, Bar Chart |
| 2D | Line Chart, Scatter Plot |
| 3D | Bubble Chart, Heatmap |
| 4+D | Radar Chart, Combo Chart |

### By Insight Type

| Insight Type | Recommended Chart |
|--------------|-------------------|
| Comparison | Bar Chart, Radar Chart |
| Trend | Line Chart, Area Chart |
| Distribution | Box Plot, Histogram |
| Relationship | Scatter Plot, Bubble Chart |
| Composition | Pie Chart, Donut Chart |

### By Presentation Intent

| Intent | Default Expression |
|--------|--------------------|
| Executive takeaway | KPI cards, short comparison bars |
| Stage evolution | Timeline/process layout |
| Schedule/roadmap | Gantt chart or timeline depending on precision |
| Scenario x metric matrix | Heatmap or structured table |
| System architecture | HTML+CSS stack diagram |

## Semantic Mapping

### Matrix Data → Heatmap

When target style ≈ Notion + row-column matrix data (scenario × year/metric), default to heatmap (ECharts), avoid multiple line charts.

### Timeline/Roadmap Types

| Condition | Processing |
|-----------|------------|
| Annual event flow with ≥6 event points | Milestone timeline layout (no Chart Engine needed) |
| Project scheduling/task management | Gantt Chart (ECharts) + key action cards |
| Parseable phase intervals | Illustrative Gantt + footer note "illustrative" |
| Below timeline threshold | Process layout + phase cards |

## Data Contracts

### Contract Rules

1. Prefer tidy/tabular input rows over ad hoc chart-specific arrays.
2. Every chart dataset should make field roles explicit: dimension, measure, series, and optional semantic fields.
3. If a chart requires derived values, keep the raw source rows and derive in mapping code.
4. When the same page can be expressed by chart or table, reuse the same contract instead of inventing a second schema.
5. Missing values must be handled by an explicit null policy, not by silent omission.

### Common Contract Fields

**Recommended metadata fields:**
- `source_id`: dataset identifier or source filename
- `metric_name`: business metric name
- `unit`: `%`, `index`, `USD mn`, `count`, etc.
- `notes`: optional footnote or caveat

These fields are not required for every row, but the chart config layer should know them before rendering titles, axes, legends, and notes.

### Line/Trend Chart

**Required Fields:**
- `time` or `period`: x-axis time key
- `value`: numeric measure

**Conditionally Required Fields:**
- `series`: required when multiple lines share the same plot

**Optional Fields:**
- `unit`: display unit
- `target` or `benchmark`: comparison overlay
- `annotation`: important event marker

**Accepted Shapes:**
- tidy rows: one row per `time x series`
- single-series rows: one row per `time`

### Bar/Comparison Chart

**Required Fields:**
- `category`: compared item name
- `value`: numeric measure

**Conditionally Required Fields:**
- `series`: required for grouped/stacked bars

**Optional Fields:**
- `rank`: sort order
- `unit`: display unit
- `highlight_flag`: emphasize key bar

### Heatmap / Matrix Chart

**Required Fields:**
- `row_key`: matrix row label
- `col_key`: matrix column label
- `value`: numeric cell value

**Optional Fields:**
- `value_label`: formatted cell text
- `series_group`: optional higher-level grouping
- `unit`: display unit

### Bubble Chart

**Required Fields:**
- `x`: numeric x-axis measure
- `y`: numeric y-axis measure
- `size`: numeric bubble size driver
- `label`: object name

**Recommended Semantic Alias:**
- `x=growth_rate`
- `y=index_0_100`
- `size=confidence_level`

**Optional Fields:**
- `group`: legend/color grouping
- `short_label`: compact legend label
- `note`: object-specific annotation

### Milestone Timeline

**Required Fields:**
- `time` or `year`: Time point
- `event_type`: Event type
- `description`: Event description (16-40 characters)

**Optional Fields:**
- `impact_direction`: Impact direction (positive/negative/both)
- `proxy_metrics`: Metric bars

### Gantt Chart

**Required Fields:**
- `task`: Task name
- `start`: Start time
- `end` or `duration`: End time or duration
- `phase`: Phase attribution

**Optional Fields:**
- `progress`: Completion (0-100)
- `owner`: Owner/team

### Fallback Discipline

1. If a line or bar contract cannot provide readable labels within the space budget, degrade to table/cards before forcing the chart.
2. If a heatmap cannot preserve legibility of both axes, degrade to structured matrix/table.
3. If a bubble chart cannot provide stable label, size, and legend readability, degrade to ranked scatter/table narrative.

## Chart-Specific Constraints

### Bubble Chart + Narrative

1. **Fixed Semantic Mapping**: x=growth_rate, y=index_0_100, r=confidence_level
2. **Radius Normalization Formula**: `r = clamp(8, 24, 8 + ((c - c_min) / max(1, c_max - c_min)) * 16)`
3. **Minimum Discriminable Difference**: Unique radius values ≥3 after deduplication
4. **Right Column Structure**: Three-segment cards, each with at least 2 key points
5. **Legend Required**: Must provide visual legend
6. **Color Consistency**: Use same colors across pages

### Bubble Chart Axis Protection

1. **Y-axis Headroom**: `y_suggested_max = y_data_max + max(2, ceil(r_max / 8))`
2. **Explicit stepSize**: Disable automatic nice ticks
3. **Clipping Gate**: `y + y_radius_equiv < y.max`

### Line/Trend Charts

1. **No Mechanical**: Not allowed `beginAtZero:true, max:100` (unless absolute percentage)
2. **Data Envelope Padding**: 15% padding
3. **Visual Utilization**: Data bandwidth occupies 55%-85% of plot area height

## Rendering Constraints

### Container Height Contract

1. **Deterministic Height**: flex-1 parent must have explicit height
2. **Anti-collapse**: Add `min-h-0` to chart child elements
3. **Chart.js**: `maintainAspectRatio: false`
4. **ECharts**: Container must have explicit width and height
5. **Canvas Minimum Height**: 200px

### Visual Budget

1. **Heatmap Minimum Height**: ≥220px
2. **Bottom Component Reserved**: `grid.bottom >= 44`
3. **Chart-Card Height Linkage**: `chart_height + title_ui + padding <= card_container_h`

### Data Guard Rules (Mandatory)

1. **NaN/null Check**: Replace with 0 or remove
2. **Length Consistency**: `labels.length == datasets[].data.length`
3. **Empty Data Fallback**: Render "No data available" placeholder card
4. **Style Profile Switch Redraw**: Call `chart.resize()` with 50ms delay after `switchStyleProfile()`; `switchBrand()` is compatibility-only
5. **First Frame Protection**: Initialize after DOMContentLoaded

## Data Loading Protocol

### Principles

- **No Hardcoding**: Not allowed to write `[10, 20, 30]`
- **Source Data Embedding**: Convert CSV to JSON and store in variable
- **Mapping Logic**: Use map/filter to extract data

### Code Template

```javascript
// 1. Embed Source Data
const sourceData = [
  { year: 2021, value: 10, category: 'A' },
  { year: 2022, value: 20, category: 'A' }
];

// 2. Map to Chart Format
const labels = sourceData.map(d => d.year);
const dataValues = sourceData.map(d => d.value);

// 3. Init Chart
new Chart(ctx, {
  data: { labels, datasets: [{ data: dataValues }] }
});
```

## Page-Level Vertical Budget Linkage

1. **Total Budget**: `header_h + main_h + footer_h == 720px`
2. **Main Area Safety Line**: `main_scroll_h <= main_client_h - 8`
3. **Chart Linkage Budget**: `title_h + chart_h + legend_h + note_h + card_padding <= card_inner_h`
4. **Line Chart Default Height**: 180-240px
5. **Runtime Validation**: Validate at 1280x720, 1366x768, 1512x982 profiles

If the chart sits inside a standard layout asset, apply that asset's `overflow_recovery_order` before using chart-specific degradation such as fewer series, shorter labels, or a table fallback.

## Dependencies

- **ppt-brand-style-system**: Chart colors from the brand-style palette
- **ppt-slide-layout-library**: Layout container height contract, `layout_contract_source`, `overflow_recovery_order`, `fallback_layouts`

## Resource Files

For detailed chart constraints, rendering rules, and code examples, refer to `assets/charts.yml`.
