# PPT Chart Thinking Examples

This file provides chart-led Thinking examples that align with `ppt-chart-engine` contracts.

Use them when a slide is chart-first and the chart contract must be explicit before implementation.

---

## Example A: Multi-Series Line Chart

```markdown
# Slide 5: Thinking

## 1. Core Task and Reasoning (Mission & Reasoning)

- **Goal**: Show how baseline, stress, and extreme scenarios diverge over time.
- **Information Structure**: Evidence-led trend comparison.
- **Data Strategy**: Chart-led because time, slope, and inflection are the argument.
- **Layout Trade-offs**:
  - *Option A*: KPI strip only. Rejected because it hides path divergence.
  - *Option B*: Side-by-side with line chart left and one evidence card right. Chosen because it preserves trend shape and interpretability.

## 2. Execution Specs (Execution Specs)

### 2.1 Layout Anchor (Layout Anchor)

- **Layout Key**: side_by_side
- **Layout Contract Source**: `skills/ppt-slide-layout-library/assets/layouts/side_by_side.yml#layout_contract`
- **Narrative Fit Match**: `scenario_tradeoff`
- **Required Fields Check**: `layout_key`, `comparison_axis`, `option_count`, `scoring_basis`, `recommendation_logic`, `evidence_type`, `fallback_plan`
- **Overflow Recovery Order**: `reduce_chart_width_pressure` -> `reduce_card_copy_density` -> `move_secondary_note_to_footer` -> `downgrade_to_data_chart`
- **Fallback Layouts**: `comparison`, `data_chart`
- **Primary Region Strategy**: chart occupies the dominant reading region
- **Secondary Region Strategy**: one evidence card interprets the inflection

### 2.2 Content Encoding (Content Encoding)

- **Primary Encoding**: chart
- **Chart Family**: line_trend
- **Contract Fields**: `time`, `value`, `series`
- **Null Policy**: keep_null
- **Source**: scenario price path dataset
- **Filter Logic**: baseline / stress / extreme only

### 2.3 Chart Contract Check (Chart Contract Check)

- **Contract Source**: `skills/ppt-chart-engine/assets/charts.yml#line_trend`
- **Fallback Plan**: if label density breaks readability, degrade to compact comparison table
- **Semantic Note**: use one critical line only; supporting lines remain neutral or secondary

### 2.4 Visual Props (Visual Props)

- **Style Profile**: Editorial Briefing
- **Density**: default
- **Highlights**: one red line, neutral grid, restrained annotation marker

### 2.5 Narrative (Narrative)

- **Headline**: Scenario divergence appears before physical supply collapse
- **Insight**: Timing and slope divergence matter more than the terminal value alone.

### 2.6 Layout Recovery and Fallback (Layout Recovery)

- **Recovery Trigger**: if tick labels or legend density shrink the main chart below readable size
- **Recovery Action**: reduce annotation density before touching series coverage
- **Fallback Trigger**: if readability still fails, switch to `data_chart`
```

---

## Example B: Heatmap Matrix

```markdown
# Slide 8: Thinking

## 1. Core Task and Reasoning (Mission & Reasoning)

- **Goal**: Compare scenario intensity across regions and time windows in one view.
- **Information Structure**: Matrix-led pattern recognition.
- **Data Strategy**: Heatmap because the insight is relative intensity, not precise sequential reading.
- **Layout Trade-offs**:
  - *Option A*: Three small bar charts. Rejected because cross-comparison becomes fragmented.
  - *Option B*: Single heatmap with short row/column labels and one takeaway card. Chosen because it makes hotspots instantly legible.

## 2. Execution Specs (Execution Specs)

### 2.1 Layout Anchor (Layout Anchor)

- **Layout Key**: dashboard_grid
- **Layout Contract Source**: `skills/ppt-slide-layout-library/assets/layouts/dashboard_grid.yml#layout_contract`
- **Narrative Fit Match**: `multi_metric_monitoring`
- **Required Fields Check**: `layout_key`, `metric_groups`, `chart_mix`, `summary_logic`, `recommendation_logic`, `fallback_plan`
- **Overflow Recovery Order**: `reduce_chart_count` -> `reduce_card_density` -> `collapse_secondary_metrics` -> `downgrade_to_hybrid`
- **Fallback Layouts**: `hybrid`, `data_chart`, `full_width`
- **Primary Region Strategy**: matrix dominates the page center
- **Secondary Region Strategy**: one card explains the strongest pattern

### 2.2 Content Encoding (Content Encoding)

- **Primary Encoding**: chart
- **Chart Family**: heatmap_matrix
- **Contract Fields**: `row_key`, `col_key`, `value`
- **Null Policy**: replace_zero
- **Source**: regional scenario matrix
- **Filter Logic**: keep only top-level regions and core time buckets

### 2.3 Chart Contract Check (Chart Contract Check)

- **Contract Source**: `skills/ppt-chart-engine/assets/charts.yml#heatmap_matrix`
- **Fallback Plan**: if axis labels exceed the space budget, degrade to structured matrix table
- **Semantic Note**: color scale must encode intensity only; narrative meaning stays in headline/card

### 2.4 Visual Props (Visual Props)

- **Style Profile**: KPMG
- **Density**: compact
- **Highlights**: short labels, quiet grid chrome, one highlighted hotspot callout

### 2.5 Narrative (Narrative)

- **Headline**: Exposure clusters around a small set of regional-time intersections
- **Insight**: The risk pattern is concentrated enough that a matrix reads faster than multiple trend charts.

### 2.6 Layout Recovery and Fallback (Layout Recovery)

- **Recovery Trigger**: if labels, legend, and explanation card compete for the same vertical budget
- **Recovery Action**: reduce secondary metrics and keep the heatmap dominant
- **Fallback Trigger**: if the matrix still becomes unreadable, switch to `hybrid`
```

---

## Example C: Bubble Relationship

```markdown
# Slide 12: Thinking

## 1. Core Task and Reasoning (Mission & Reasoning)

- **Goal**: Compare actors by growth, resilience, and confidence in one compact frame.
- **Information Structure**: Relationship-led quadrant view.
- **Data Strategy**: Bubble chart because x, y, and size each carry distinct analytical meaning.
- **Layout Trade-offs**:
  - *Option A*: Ranked table only. Rejected because it suppresses the multidimensional pattern.
  - *Option B*: Bubble chart with short labels and one interpretation card. Chosen because the positional relationships are the main story.

## 2. Execution Specs (Execution Specs)

### 2.1 Layout Anchor (Layout Anchor)

- **Layout Key**: side_by_side
- **Layout Contract Source**: `skills/ppt-slide-layout-library/assets/layouts/side_by_side.yml#layout_contract`
- **Narrative Fit Match**: `scenario_tradeoff`
- **Required Fields Check**: `layout_key`, `comparison_axis`, `option_count`, `scoring_basis`, `recommendation_logic`, `evidence_type`, `fallback_plan`
- **Overflow Recovery Order**: `reduce_chart_width_pressure` -> `reduce_card_copy_density` -> `move_secondary_note_to_footer` -> `downgrade_to_data_chart`
- **Fallback Layouts**: `comparison`, `data_chart`
- **Primary Region Strategy**: bubble chart occupies the left analysis region
- **Secondary Region Strategy**: right card interprets quadrant meaning and outliers

### 2.2 Content Encoding (Content Encoding)

- **Primary Encoding**: chart
- **Chart Family**: bubble_relationship
- **Contract Fields**: `x`, `y`, `size`, `label`
- **Recommended Alias**: `growth_rate`, `index_0_100`, `confidence_level`
- **Null Policy**: drop_row only if label integrity is preserved
- **Source**: actor assessment table
- **Filter Logic**: keep only top-priority actors with short labels

### 2.3 Chart Contract Check (Chart Contract Check)

- **Contract Source**: `skills/ppt-chart-engine/assets/charts.yml#bubble_relationship`
- **Fallback Plan**: if label collisions persist, degrade to scatter + ranking table
- **Semantic Note**: bubble radius is contract-driven, not manually styled by intuition

### 2.4 Visual Props (Visual Props)

- **Style Profile**: Deloitte
- **Density**: default
- **Highlights**: one highlighted actor, muted peers, compact axis labels

### 2.5 Narrative (Narrative)

- **Headline**: The leaders are differentiated by confidence, not just position
- **Insight**: Bubble size changes the reading: similar positions do not imply similar execution confidence.

### 2.6 Layout Recovery and Fallback (Layout Recovery)

- **Recovery Trigger**: if label collision or outlier annotation blocks quadrant reading
- **Recovery Action**: shorten labels and move secondary interpretation to the side card first
- **Fallback Trigger**: if collision persists after recovery, switch to `data_chart` or scatter-plus-table fallback
```
