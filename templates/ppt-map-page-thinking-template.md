# Slide {N}: Thinking

## 1. 核心任务与推理 (Mission & Reasoning)

- **目标**: Explain the single spatial question this page must answer.
- **视觉隐喻**: Define the map story type in plain language, such as conflict theater, route corridor, or regional exposure.
- **数据策略**: State whether the page is data-driven (`ECharts Geo`) or editorial-first (`SVG` narrative map), and why.
- **布局权衡**:
  - *方案 A*: Describe the rejected non-map or alternate-map option.
  - *方案 B*: Describe the chosen map expression and why it is superior for this page.

---

## 2. 执行规格 (Execution Specs)

### 2.1 地图叙事输入下限 (Map Input Minimum)

- **Narrative_Archetype**: territory_snapshot | route_corridor | conflict_theater | footprint_network
- **Geographic_Scope**: local | regional | multi-region | global
- **Primary_Question**: One spatial question only.
- **Render_Engine**: echarts-geo | static-svg-overlay | maplibre-gl-js
- **Basemap_Source**:
  - **Type**: svg_asset | geojson | topojson | vector_tile_style
  - **ID**: asset or dataset identifier
  - **Purpose**: why this basemap source fits the page

### 2.2 布局锚点 (Layout Anchor)

- **Layout Key**: map_overlay | side_by_side | hybrid
- **Layout_Contract_Source**: `skills/ppt-slide-layout-library/assets/layouts/<layout_key>.yml#layout_contract`
- **Narrative_Fit_Match**:
  which `layout_contract.compatible_map_archetypes` or
  `layout_contract.narrative_fit` label this page satisfies
- **Required_Thinking_Fields_Check**: confirm the page has all fields required by the chosen layout contract
- **Overflow_Recovery_Order**: copy the ordered recovery path from the chosen layout contract
- **Fallback_Layouts**: copy the allowed fallback layouts from the chosen layout contract
- **Component**: map + cards | map + KPI panel | map + callout stack

### 2.3 数据绑定 (Data Binding)

- **Source**: file, note, dataset, or research source
- **Filter Logic**: what is included and excluded from the map
- **Mapping**:
  - key places / regions / routes
  - overlay families
  - whether geo series are required

### 2.4 Overlay Routing

- **Overlay_Families**: list the overlay families used on the page
- **Overlay_Routing**:
  for each overlay family, state whether it routes to chart layer, component family, or pure narrative overlay
- **Routing_Source**: `skills/ppt-map-storytelling/assets/patterns.yml#overlay_component_contracts`
- **Escalation_Check**: note whether any overlay must escalate to `ppt-chart-engine`

### 2.5 视觉细节 (Visual Props)

- **Style**: active style profile or intended page tone
- **Component_Variant**: the chosen visual treatment
- **Highlights**: arrows, hotspots, fills, floating cards, KPI emphasis

### 2.6 组件语义解析 (Component Semantic Resolution)

- **Component_Selection**: list the standard components used on the page, if any
- **Semantic_Roles**: for each standard component, list the semantic roles used by the payload
- **Resolver_Source**: `skills/ppt-brand-style-system/assets/component_semantic_mappings.yml` when semantic_payload is present
- **Fallback_Policy**: note whether the page uses resolver-first classes only, or resolver + safe fallback payload

### 2.7 叙事文案 (Narrative)

- **Headline**: one clear map headline
- **Insight**: one map-linked conclusion, not a generic summary

### 2.8 布局恢复与降级 (Layout Recovery)

- **Recovery Trigger**: what spatial readability failure would trigger overlay reduction
- **Recovery Action**: which first action from `Overflow_Recovery_Order` will be executed before layout switching
- **Fallback Trigger**: when to stop overlay recovery and switch to a layout in `Fallback_Layouts`

## Example Filled Pattern

- **Narrative_Archetype**: route_corridor
- **Geographic_Scope**: regional
- **Primary_Question**: Which corridor concentrates the highest disruption risk?
- **Render_Engine**: echarts-geo
- **Basemap_Source**:
  - **Type**: geojson
  - **ID**: gulf_corridor_v1
  - **Purpose**: coastline and chokepoint anchors for route lines and hotspot callouts
- **Overlay_Families**: `route_line`, `hotspot_callout`
- **Overlay_Routing**:
  - `route_line` -> chart layer via `ECharts Geo` path encoding + `Map_FlowArrow` fallback only for editorial simplification
  - `hotspot_callout` -> narrative overlay + `Card_Accent` / `Badge_Pill`
- **Routing_Source**: `skills/ppt-map-storytelling/assets/patterns.yml#overlay_component_contracts`
- **Escalation_Check**: `route_line` escalates to `ppt-chart-engine`; `hotspot_callout` stays in map-storytelling
- **Layout Key**: map_overlay
- **Layout_Contract_Source**: `skills/ppt-slide-layout-library/assets/layouts/map_overlay.yml#layout_contract`
- **Narrative_Fit_Match**: `route_corridor`
- **Required_Thinking_Fields_Check**:
  `layout_key`, `narrative_archetype`, `render_engine`,
  `overlay_families`, `overlay_routing`, `routing_source`
- **Overflow_Recovery_Order**:
  `reduce_overlay_density` -> `reduce_support_card_count` ->
  `collapse_secondary_labels` -> `downgrade_to_side_by_side`
- **Fallback_Layouts**: `side_by_side`, `full_width`
- **Component_Selection**: `Metric_Trend`, `Card_Accent`
- **Semantic_Roles**:
  - `Metric_Trend` -> `trend_role: warning`, `value_role: primary_text`
  - `Card_Accent` -> `emphasis_role: warning`, `surface_role: elevated`
- **Resolver_Source**: `skills/ppt-brand-style-system/assets/component_semantic_mappings.yml`
