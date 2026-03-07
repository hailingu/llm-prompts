# PPT Thinking Examples

This file provides reference-grade Thinking examples for the three most common slide execution paths.

Use them as calibration examples, not as rigid templates. The operative templates remain:

- `templates/ppt-slide-thinking-template.md`
- `templates/ppt-map-page-thinking-template.md`

---

## Example A: Chart-Led Slide

```markdown
# Slide 6: Thinking

## 1. 核心任务与推理 (Mission & Reasoning)

- **目标**: Show that oil-price upside is driven by corridor-specific disruption rather than global supply collapse.
- **信息结构**: Evidence-led. The chart carries the main argument and the text only interprets it.
- **数据策略**: Chart-led. A line chart is the clearest way to show timing, inflection, and divergence across scenarios.
- **布局权衡**:
  - *方案 A*: KPI cards + prose summary. Rejected because it hides timing and over-compresses the inflection.
  - *方案 B*: Side-by-side layout with line chart left and interpretation card right. Chosen because the chart proves the claim while the card locks the insight.

## 2. 执行规格 (Execution Specs)

### 2.1 页面骨架 (Layout Anchor)

- **Layout Key**: side_by_side
- **Layout_Contract_Source**: `skills/ppt-slide-layout-library/assets/layouts/side_by_side.yml#layout_contract`
- **Narrative_Fit_Match**: `scenario_tradeoff`
- **Required_Thinking_Fields_Check**: `layout_key`, `comparison_axis`, `option_count`, `scoring_basis`, `recommendation_logic`, `evidence_type`, `fallback_plan`
- **Overflow_Recovery_Order**: `reduce_chart_width_pressure` -> `reduce_card_copy_density` -> `move_secondary_note_to_footer` -> `downgrade_to_data_chart`
- **Fallback_Layouts**: `comparison`, `data_chart`
- **Primary Region Strategy**: left region holds the time-series chart
- **Secondary Region Strategy**: right region holds one accent card and one metric

### 2.2 内容编码 (Content Encoding)

- **Primary Encoding**: chart
- **Source**: scenario dataset + price path notes
- **Filter Logic**: keep only baseline / stress / extreme scenarios; remove low-signal variants
- **Mapping**: headline states the conclusion; chart shows path divergence; card explains why the inflection matters

### 2.3 组件语义解析 (Component Semantic Resolution)

- **Component_Selection**: `Metric_Big`, `Card_Accent`
- **Semantic_Roles**:
  - `Metric_Big` -> `component_family: Metric_Big`, `emphasis_role: critical`, `value_role: primary_text`
  - `Card_Accent` -> `component_family: Card_Accent`, `emphasis_role: warning`, `surface_role: elevated`
- **Resolver_Source**: `skills/ppt-brand-style-system/assets/component_semantic_mappings.yml`
- **Fallback_Policy**: resolver first; only keep example payload as emergency fallback

### 2.4 视觉细节 (Visual Props)

- **Style**: Editorial Briefing
- **Density**: default
- **Highlights**: one red scenario line, one amber watchpoint card, restrained neutral chrome elsewhere

### 2.5 叙事文案 (Narrative)

- **Headline**: Price upside is corridor-driven, not system-wide
- **Insight**: The market reacts to localized chokepoint stress well before any global supply breakdown materializes.

### 2.6 布局恢复与降级 (Layout Recovery)

- **Recovery Trigger**: if right-side interpretation card compresses the chart below an analytically legible width
- **Recovery Action**: first reduce card copy density, then move secondary note into footer support
- **Fallback Trigger**: if the line chart still loses label/tick readability after recovery, switch to `data_chart`
```

---

## Example B: Component-Led Slide

```markdown
# Slide 9: Thinking

## 1. 核心任务与推理 (Mission & Reasoning)

- **目标**: Summarize the operating model as three compact takeaways with one sequence and one KPI row.
- **信息结构**: Component-led. The page is about structured reading, not data visualization.
- **数据策略**: Mixed notes + normalized facts. No chart is needed because the task is grouping and sequencing.
- **布局权衡**:
  - *方案 A*: Dashboard of six small cards. Rejected because it fragments the reading path.
  - *方案 B*: Hybrid layout with KPI row on top, accent card left, timeline right. Chosen because it gives a clear top-down reading order.

## 2. 执行规格 (Execution Specs)

### 2.1 页面骨架 (Layout Anchor)

- **Layout Key**: hybrid
- **Layout_Contract_Source**: `skills/ppt-slide-layout-library/assets/layouts/hybrid.yml#layout_contract`
- **Narrative_Fit_Match**: `mixed_evidence_summary`
- **Required_Thinking_Fields_Check**: `layout_key`, `primary_chart_family`, `chart_role`, `supporting_cards`, `synthesis_order`, `recommendation_logic`, `fallback_plan`
- **Overflow_Recovery_Order**: `reduce_card_count` -> `reduce_chart_annotation_density` -> `move_secondary_metrics_to_cards` -> `downgrade_to_data_chart`
- **Fallback_Layouts**: `data_chart`, `dashboard_grid`, `chart_synthesis`
- **Primary Region Strategy**: top strip gives compact KPIs, lower region gives interpretation + sequence
- **Secondary Region Strategy**: one side explains, one side sequences

### 2.2 内容编码 (Content Encoding)

- **Primary Encoding**: components
- **Source**: research summary + scenario notes
- **Filter Logic**: keep only operational facts that directly support the page thesis
- **Mapping**: KPI row for scale, accent card for watchpoint, timeline for sequence

### 2.3 组件语义解析 (Component Semantic Resolution)

- **Component_Selection**: `Metric_KpiRow`, `Card_Accent`, `List_Timeline`
- **Semantic_Roles**:
  - `Metric_KpiRow` -> `component_family: Metric_KpiRow`, `surface_role: subtle`, `structure_role: neutral_structure`, `value_role: primary_text`
  - `Card_Accent` -> `component_family: Card_Accent`, `emphasis_role: warning`, `surface_role: elevated`
  - `List_Timeline` -> `component_family: List_Timeline`, `timeline_role: neutral_structure`, `active_step_role: primary`, `inactive_step_role: neutral`
- **Resolver_Source**: `skills/ppt-brand-style-system/assets/component_semantic_mappings.yml`
- **Fallback_Policy**: resolver first; examples only define safe fallback classes

### 2.4 视觉细节 (Visual Props)

- **Style**: KPMG
- **Density**: compact
- **Highlights**: keep one amber emphasis point only; let structure stay neutral

### 2.5 叙事文案 (Narrative)

- **Headline**: The operating model stays concentrated and sequential
- **Insight**: A small set of nodes and a short escalation chain explain most of the downside transmission.

### 2.6 布局恢复与降级 (Layout Recovery)

- **Recovery Trigger**: if KPI row plus lower cards creates vertical compression in the sequence region
- **Recovery Action**: reduce card count before reducing sequence fidelity
- **Fallback Trigger**: if the lower region still fragments the reading path, switch to `chart_synthesis` or `data_chart`
```

---

## Example C: Map-Led Slide

```markdown
# Slide 11: Thinking

## 1. 核心任务与推理 (Mission & Reasoning)

- **目标**: Show which maritime corridor concentrates the visible disruption burden.
- **视觉隐喻**: Route corridor with one chokepoint and surrounding exposure halo.
- **数据策略**: Data-driven map. `ECharts Geo` is needed because the route geometry and hotspot positions are part of the argument.
- **布局权衡**:
  - *方案 A*: Standard side-by-side with abstract node network. Rejected because it weakens real geography.
  - *方案 B*: Map overlay layout with route, hotspot ring, and one supporting KPI card. Chosen because location is the meaning.

## 2. 执行规格 (Execution Specs)

### 2.1 地图叙事输入下限 (Map Input Minimum)

- **Narrative_Archetype**: route_corridor
- **Geographic_Scope**: regional
- **Primary_Question**: Which corridor concentrates the highest disruption risk?
- **Render_Engine**: echarts-geo
- **Basemap_Source**:
  - **Type**: geojson
  - **ID**: gulf_corridor_v1
  - **Purpose**: coastline and chokepoint anchors for route line and hotspot callouts

### 2.2 布局锚点 (Layout Anchor)

- **Layout Key**: map_overlay
- **Layout_Contract_Source**: `skills/ppt-slide-layout-library/assets/layouts/map_overlay.yml#layout_contract`
- **Narrative_Fit_Match**: `route_corridor`
- **Required_Thinking_Fields_Check**: `layout_key`, `narrative_archetype`, `render_engine`, `overlay_families`, `overlay_routing`, `routing_source`
- **Overflow_Recovery_Order**: `reduce_overlay_density` -> `reduce_support_card_count` -> `collapse_secondary_labels` -> `downgrade_to_side_by_side`
- **Fallback_Layouts**: `side_by_side`, `full_width`
- **Component**: map + KPI panel

### 2.3 数据绑定 (Data Binding)

- **Source**: shipping route notes + corridor risk dataset
- **Filter Logic**: retain only the primary export route and top two exposure nodes
- **Mapping**:
  - route line -> main disruption corridor
  - radar ring -> highest-risk chokepoint
  - KPI card -> concentration metric

### 2.4 视觉细节 (Visual Props)

- **Style**: Editorial Briefing
- **Component_Variant**: restrained basemap + high-contrast overlays
- **Highlights**: one route line, one red ring, one amber evidence card

### 2.5 组件语义解析 (Component Semantic Resolution)

- **Component_Selection**: `Map_FlowArrow`, `Map_RadarRing`, `Metric_Trend`
- **Semantic_Roles**:
  - `Map_FlowArrow` -> `component_family: Map_FlowArrow`, `flow_role: primary`
  - `Map_RadarRing` -> `component_family: Map_RadarRing`, `core_role: critical`, `icon_role: primary_text`, `ring_role: critical`
  - `Metric_Trend` -> `component_family: Metric_Trend`, `trend_role: warning`, `value_role: primary_text`
- **Resolver_Source**: `skills/ppt-brand-style-system/assets/component_semantic_mappings.yml`
- **Fallback_Policy**: resolver first; retain geo overlay fallback only for emergency rendering recovery

### 2.6 叙事文案 (Narrative)

- **Headline**: Disruption risk concentrates in one export corridor
- **Insight**: The transmission channel is geographically narrow, which is why localized disruption can still price like a global shock.

### 2.7 布局恢复与降级 (Layout Recovery)

- **Recovery Trigger**: if overlay labels and callouts begin to obscure the route geometry
- **Recovery Action**: reduce secondary labels before removing the primary route and hotspot overlay
- **Fallback Trigger**: if spatial readability still fails after overlay reduction, switch to `side_by_side`
```
