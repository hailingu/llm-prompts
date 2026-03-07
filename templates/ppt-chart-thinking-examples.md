# PPT Chart Thinking Examples

This file provides chart-led Thinking examples that align with `ppt-chart-engine` contracts.

Use them when a slide is chart-first and the chart contract must be explicit before implementation.

---

## Example A: Multi-Series Line Chart

```markdown
# Slide 5: Thinking

## 1. 核心任务与推理 (Mission & Reasoning)

- **目标**: Show how baseline, stress, and extreme scenarios diverge over time.
- **信息结构**: Evidence-led trend comparison.
- **数据策略**: Chart-led because time, slope, and inflection are the argument.
- **布局权衡**:
  - *方案 A*: KPI strip only. Rejected because it hides path divergence.
  - *方案 B*: Side-by-side with line chart left and one evidence card right. Chosen because it preserves trend shape and interpretability.

## 2. 执行规格 (Execution Specs)

### 2.1 页面骨架 (Layout Anchor)

- **Layout Key**: side_by_side
- **Layout_Contract_Source**: `skills/ppt-slide-layout-library/assets/layouts/side_by_side.yml#layout_contract`
- **Narrative_Fit_Match**: `scenario_tradeoff`
- **Required_Thinking_Fields_Check**: `layout_key`, `comparison_axis`, `option_count`, `scoring_basis`, `recommendation_logic`, `evidence_type`, `fallback_plan`
- **Overflow_Recovery_Order**: `reduce_chart_width_pressure` -> `reduce_card_copy_density` -> `move_secondary_note_to_footer` -> `downgrade_to_data_chart`
- **Fallback_Layouts**: `comparison`, `data_chart`
- **Primary Region Strategy**: chart occupies the dominant reading region
- **Secondary Region Strategy**: one evidence card interprets the inflection

### 2.2 内容编码 (Content Encoding)

- **Primary Encoding**: chart
- **Chart_Family**: line_trend
- **Contract_Fields**: `time`, `value`, `series`
- **Null_Policy**: keep_null
- **Source**: scenario price path dataset
- **Filter Logic**: baseline / stress / extreme only

### 2.3 图表契约判断 (Chart Contract Check)

- **Contract_Source**: `skills/ppt-chart-engine/assets/charts.yml#line_trend`
- **Fallback_Plan**: if label density breaks readability, degrade to compact comparison table
- **Semantic_Note**: use one critical line only; supporting lines remain neutral or secondary

### 2.4 视觉细节 (Visual Props)

- **Style**: Editorial Briefing
- **Density**: default
- **Highlights**: one red line, neutral grid, restrained annotation marker

### 2.5 叙事文案 (Narrative)

- **Headline**: Scenario divergence appears before physical supply collapse
- **Insight**: Timing and slope divergence matter more than the terminal value alone.

### 2.6 布局恢复与降级 (Layout Recovery)

- **Recovery Trigger**: if tick labels or legend density shrink the main chart below readable size
- **Recovery Action**: reduce annotation density before touching series coverage
- **Fallback Trigger**: if readability still fails, switch to `data_chart`
```

---

## Example B: Heatmap Matrix

```markdown
# Slide 8: Thinking

## 1. 核心任务与推理 (Mission & Reasoning)

- **目标**: Compare scenario intensity across regions and time windows in one view.
- **信息结构**: Matrix-led pattern recognition.
- **数据策略**: Heatmap because the insight is relative intensity, not precise sequential reading.
- **布局权衡**:
  - *方案 A*: Three small bar charts. Rejected because cross-comparison becomes fragmented.
  - *方案 B*: Single heatmap with short row/column labels and one takeaway card. Chosen because it makes hotspots instantly legible.

## 2. 执行规格 (Execution Specs)

### 2.1 页面骨架 (Layout Anchor)

- **Layout Key**: dashboard_grid
- **Layout_Contract_Source**: `skills/ppt-slide-layout-library/assets/layouts/dashboard_grid.yml#layout_contract`
- **Narrative_Fit_Match**: `multi_metric_monitoring`
- **Required_Thinking_Fields_Check**: `layout_key`, `metric_groups`, `chart_mix`, `summary_logic`, `recommendation_logic`, `fallback_plan`
- **Overflow_Recovery_Order**: `reduce_chart_count` -> `reduce_card_density` -> `collapse_secondary_metrics` -> `downgrade_to_hybrid`
- **Fallback_Layouts**: `hybrid`, `data_chart`, `full_width`
- **Primary Region Strategy**: matrix dominates the page center
- **Secondary Region Strategy**: one card explains the strongest pattern

### 2.2 内容编码 (Content Encoding)

- **Primary Encoding**: chart
- **Chart_Family**: heatmap_matrix
- **Contract_Fields**: `row_key`, `col_key`, `value`
- **Null_Policy**: replace_zero
- **Source**: regional scenario matrix
- **Filter Logic**: keep only top-level regions and core time buckets

### 2.3 图表契约判断 (Chart Contract Check)

- **Contract_Source**: `skills/ppt-chart-engine/assets/charts.yml#heatmap_matrix`
- **Fallback_Plan**: if axis labels exceed the space budget, degrade to structured matrix table
- **Semantic_Note**: color scale must encode intensity only; narrative meaning stays in headline/card

### 2.4 视觉细节 (Visual Props)

- **Style**: KPMG
- **Density**: compact
- **Highlights**: short labels, quiet grid chrome, one highlighted hotspot callout

### 2.5 叙事文案 (Narrative)

- **Headline**: Exposure clusters around a small set of regional-time intersections
- **Insight**: The risk pattern is concentrated enough that a matrix reads faster than multiple trend charts.

### 2.6 布局恢复与降级 (Layout Recovery)

- **Recovery Trigger**: if labels, legend, and explanation card compete for the same vertical budget
- **Recovery Action**: reduce secondary metrics and keep the heatmap dominant
- **Fallback Trigger**: if the matrix still becomes unreadable, switch to `hybrid`
```

---

## Example C: Bubble Relationship

```markdown
# Slide 12: Thinking

## 1. 核心任务与推理 (Mission & Reasoning)

- **目标**: Compare actors by growth, resilience, and confidence in one compact frame.
- **信息结构**: Relationship-led quadrant view.
- **数据策略**: Bubble chart because x, y, and size each carry distinct analytical meaning.
- **布局权衡**:
  - *方案 A*: Ranked table only. Rejected because it suppresses the multidimensional pattern.
  - *方案 B*: Bubble chart with short labels and one interpretation card. Chosen because the positional relationships are the main story.

## 2. 执行规格 (Execution Specs)

### 2.1 页面骨架 (Layout Anchor)

- **Layout Key**: side_by_side
- **Layout_Contract_Source**: `skills/ppt-slide-layout-library/assets/layouts/side_by_side.yml#layout_contract`
- **Narrative_Fit_Match**: `scenario_tradeoff`
- **Required_Thinking_Fields_Check**: `layout_key`, `comparison_axis`, `option_count`, `scoring_basis`, `recommendation_logic`, `evidence_type`, `fallback_plan`
- **Overflow_Recovery_Order**: `reduce_chart_width_pressure` -> `reduce_card_copy_density` -> `move_secondary_note_to_footer` -> `downgrade_to_data_chart`
- **Fallback_Layouts**: `comparison`, `data_chart`
- **Primary Region Strategy**: bubble chart occupies the left analysis region
- **Secondary Region Strategy**: right card interprets quadrant meaning and outliers

### 2.2 内容编码 (Content Encoding)

- **Primary Encoding**: chart
- **Chart_Family**: bubble_relationship
- **Contract_Fields**: `x`, `y`, `size`, `label`
- **Recommended_Alias**: `growth_rate`, `index_0_100`, `confidence_level`
- **Null_Policy**: drop_row only if label integrity is preserved
- **Source**: actor assessment table
- **Filter Logic**: keep only top-priority actors with short labels

### 2.3 图表契约判断 (Chart Contract Check)

- **Contract_Source**: `skills/ppt-chart-engine/assets/charts.yml#bubble_relationship`
- **Fallback_Plan**: if label collisions persist, degrade to scatter + ranking table
- **Semantic_Note**: bubble radius is contract-driven, not manually styled by intuition

### 2.4 视觉细节 (Visual Props)

- **Style**: Deloitte
- **Density**: default
- **Highlights**: one highlighted actor, muted peers, compact axis labels

### 2.5 叙事文案 (Narrative)

- **Headline**: The leaders are differentiated by confidence, not just position
- **Insight**: Bubble size changes the reading: similar positions do not imply similar execution confidence.

### 2.6 布局恢复与降级 (Layout Recovery)

- **Recovery Trigger**: if label collision or outlier annotation blocks quadrant reading
- **Recovery Action**: shorten labels and move secondary interpretation to the side card first
- **Fallback Trigger**: if collision persists after recovery, switch to `data_chart` or scatter-plus-table fallback
```
