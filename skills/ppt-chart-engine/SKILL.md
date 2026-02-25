---
name: ppt-chart-engine
description: Chart engine for PPT HTML slides - chart type catalog, selection algorithm, semantic mapping, data contracts, rendering constraints (Bubble/Line/Heatmap), container height contract, data guards, and code examples.
metadata: 
  - version: 1.0.0
  - author: ppt-chart-engine
---

# PPT Chart Engine

## Overview

This skill provides a complete chart selection, rendering constraints, and data contract system, supporting both Chart.js and ECharts engines.

## When to Use This Skill

- Select appropriate chart type for data
- Apply chart rendering constraints
- Handle data contracts and mappings
- Implement data guard rules
- Write chart code examples

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
4. **Brand Switch Redraw**: Call `chart.resize()` with 50ms delay after `switchBrand()`
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

## Dependencies

- **ppt-brand-system**: Chart colors from brand palette
- **ppt-slide-layout-library**: Layout container height contract

## Resource Files

For detailed chart constraints, rendering rules, and code examples, refer to `assets/charts.yml`.
