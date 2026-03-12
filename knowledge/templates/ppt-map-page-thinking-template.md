# Slide {N}: Thinking

## 1. Core Task and Reasoning (Mission & Reasoning)

- **Goal**: Explain the single spatial question this page must answer.
- **Visual Metaphor**: Define the map story type in plain language, such as conflict theater, route corridor, or regional exposure.
- **Data Strategy**: State whether the page is data-driven (`ECharts Geo`) or editorial-first (`SVG` narrative map), and why.
- **Layout Trade-offs**:
  - *Option A*: Describe the rejected non-map or alternate-map option.
  - *Option B*: Describe the chosen map expression and why it is superior for this page.

---

## 2. Execution Specs (Execution Specs)

### 2.1 Map Narrative Minimum Inputs (Map Input Minimum)

- **Narrative Archetype**: territory_snapshot | route_corridor | conflict_theater | footprint_network
- **Geographic Scope**: local | regional | multi-region | global
- **Primary Question**: one spatial question only
- **Render Engine**: echarts-geo | static-svg-overlay | maplibre-gl-js
- **Basemap Source**:
  - **Type**: svg_asset | geojson | topojson | vector_tile_style
  - **ID**: asset or dataset identifier
  - **Purpose**: why this basemap source fits the page

### 2.2 Layout Anchor (Layout Anchor)

- **Layout Key**: map_overlay | side_by_side | hybrid
- **Layout Contract Source**: `skills/ppt-slide-layout-library/assets/layouts/<layout_key>.yml#layout_contract`
- **Narrative Fit Match**:
  which `layout_contract.compatible_map_archetypes` or
  `layout_contract.narrative_fit` label this page satisfies
- **Required Fields Check**: confirm the page has all fields required by the chosen layout contract
- **Overflow Recovery Order**: copy the ordered recovery path from the chosen layout contract
- **Fallback Layouts**: copy the allowed fallback layouts from the chosen layout contract
- **Primary Region Strategy**: map-dominant | map-balanced | map-supported
- **Secondary Region Strategy**: cards | KPI panel | callout stack

### 2.3 Data Binding (Data Binding)

- **Source**: file, note, dataset, or research source
- **Filter Logic**: what is included and excluded from the map
- **Spatial Mapping**:
  - key places / regions / routes
  - what each retained element proves for the spatial question

### 2.4 Overlay Routing

- **Overlay Families**: list the overlay families used on the page
- **Overlay Routing**:
  for each overlay family, state whether it routes to chart layer, component family, or pure narrative overlay
- **Routing Source**: `skills/ppt-map-storytelling/assets/patterns.yml#overlay_component_contracts`
- **Escalation Check**: note whether any overlay must escalate to `ppt-chart-engine`

### 2.5 Component Semantic Resolution (Component Semantic Resolution)

- **Component Selection**: list the standard components used on the page, if any
- **Semantic Roles**: for each standard component, list the semantic roles used by the payload
- **Resolver Source**: `skills/ppt-brand-style-system/assets/component_semantic_mappings.yml` when semantic_payload is present
- **Fallback Policy**: note whether the page uses resolver-first classes only, or resolver + safe fallback payload

### 2.6 Visual Props (Visual Props)

- **Style Profile**: active style profile or intended page tone
- **Component Variant**: the chosen visual treatment
- **Density**: compact | default | relaxed
- **Highlights**: arrows, hotspots, fills, floating cards, KPI emphasis

### 2.7 Narrative (Narrative)

- **Headline**: one clear map headline
- **Insight**: one map-linked conclusion, not a generic summary

### 2.8 Layout Recovery and Fallback (Layout Recovery)

- **Recovery Trigger**: what spatial readability failure would trigger overlay reduction
- **Recovery Action**: which first action from `Overflow Recovery Order` will be executed before layout switching
- **Fallback Trigger**: when to stop overlay recovery and switch to a layout in `Fallback Layouts`

## Example Filled Pattern

- **Narrative Archetype**: route_corridor
- **Geographic Scope**: regional
- **Primary Question**: Which corridor concentrates the highest disruption risk?
- **Render Engine**: echarts-geo
- **Basemap Source**:
  - **Type**: geojson
  - **ID**: gulf_corridor_v1
  - **Purpose**: coastline and chokepoint anchors for route lines and hotspot callouts
- **Overlay Families**: `route_line`, `hotspot_callout`
- **Overlay Routing**:
  - `route_line` -> chart layer via `ECharts Geo` path encoding + `Map_FlowArrow` fallback only for editorial simplification
  - `hotspot_callout` -> narrative overlay + `Card_Accent` / `Badge_Pill`
- **Routing Source**: `skills/ppt-map-storytelling/assets/patterns.yml#overlay_component_contracts`
- **Escalation Check**: `route_line` escalates to `ppt-chart-engine`; `hotspot_callout` stays in map-storytelling
- **Layout Key**: map_overlay
- **Layout Contract Source**: `skills/ppt-slide-layout-library/assets/layouts/map_overlay.yml#layout_contract`
- **Narrative Fit Match**: `route_corridor`
- **Required Fields Check**:
  `layout_key`, `narrative_archetype`, `render_engine`,
  `overlay_families`, `overlay_routing`, `routing_source`
- **Overflow Recovery Order**:
  `reduce_overlay_density` -> `reduce_support_card_count` ->
  `collapse_secondary_labels` -> `downgrade_to_side_by_side`
- **Fallback Layouts**: `side_by_side`, `full_width`
- **Component Selection**: `Metric_Trend`, `Card_Accent`
- **Semantic Roles**:
  - `Metric_Trend` -> `trend_role: warning`, `value_role: primary_text`
  - `Card_Accent` -> `emphasis_role: warning`, `surface_role: elevated`
- **Resolver Source**: `skills/ppt-brand-style-system/assets/component_semantic_mappings.yml`
