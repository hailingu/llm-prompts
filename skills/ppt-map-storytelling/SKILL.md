---
name: ppt-map-storytelling
description: Map storytelling system for PPT HTML slides - geographic narrative patterns, map-first page types, overlay rules, conflict/path annotations, and collaboration boundaries with chart/layout skills.
metadata:
  - version: 1.0.0
  - author: ppt-html-generator
---

# PPT Map Storytelling

## Overview

This skill governs how maps are used as narrative surfaces in PPT HTML slides.

It does not replace `ppt-chart-engine`. Instead, it covers the storytelling layer that begins after a team has already decided the page needs a geographic view.

Use this skill when the page must explain geography, conflict space, movement, routes, regional positioning, or map-centered narrative structure rather than only render a geo chart.

## Decision Mnemonic

Use this rule at the start of every map-capable page:

**Decide the page first, then the chart; decide the narrative first, then the skeleton; only invoke a map engine when geo series are actually required.**

Expanded execution order:

1. Determine whether this is truly a map page.
2. Choose the map narrative archetype, geographic scope, and overlay grammar.
3. Choose the map-capable layout skeleton.
4. Use a map rendering library only when the chosen story actually needs a geographic base or geo series.

This mnemonic exists to prevent drift between `ppt-map-storytelling`, `ppt-slide-layout-library`, and `ppt-chart-engine`.

## Rendering Library Policy

Do not draw maps from scratch.

### Default Library: `ECharts` Geo

Use `ECharts` as the default map rendering engine when the page needs data-driven geography:

- choropleth / shaded regions
- geo scatter / point markers
- route lines / directional paths
- heat or intensity overlays
- a single map container that integrates with KPI cards and presentation-safe HTML output

Why this is the default:

- already aligned with `ppt-chart-engine`
- already used elsewhere in this PPT system
- fits static or lightly animated presentation pages better than full GIS-style interactive stacks

### Secondary Base: Static `SVG` Map + HTML Overlay

Use a simplified static SVG map base when the page is primarily editorial rather than data-dense:

- conflict theater backdrop
- corridor or chokepoint explanation
- regional focus page with only a few labels, arrows, or callouts
- cases where map readability matters more than geographic data precision

In this mode, the map base should come from a prepared SVG asset or simplified vector source, then be annotated with HTML/SVG overlays. Do not hand-draw coastlines or borders inside the slide code.

### Exception Library: `MapLibre GL JS`

Use `MapLibre GL JS` only when the page genuinely requires vector-tile behavior that `ECharts` or static SVG cannot reasonably cover, such as:

- large-scale pan/zoom exploration
- multi-layer basemap styling as part of the story
- reusable tile-driven geographic scenes across many slides

This is the exception, not the default, because PPT pages are usually fixed-view narrative surfaces rather than interactive GIS products.

### Avoid by Default

- `Leaflet`: avoid as the default PPT map engine; it is better for app-like slippy maps than editorial slide storytelling.
- raw `D3` geography from zero: only use for highly custom overlays after the base map source already exists.
- manual border/coastline drawing in HTML/CSS/SVG: prohibited for normal workflow.

### Practical Selection Rule

1. If the page needs geo series or geographic data encoding, use `ECharts`.
2. If the page needs a simplified narrative backdrop with light overlays, use static `SVG` + HTML/SVG overlay.
3. If the page truly needs tile/vector-map behavior, escalate to `MapLibre GL JS`.

## Basemap Source Standard

Every map page must declare what the basemap source is before implementation.

### Allowed Basemap Inputs

- prepared `SVG` regional or world map asset for editorial pages
- `GeoJSON` for region polygons, geo scatter anchors, and simplified geographic features
- `TopoJSON` when polygon-heavy regional data needs lighter payloads
- vector tile / style source only when `MapLibre GL JS` is explicitly justified

### Source Selection Rule

1. Use prepared `SVG` when the page is primarily narrative and only needs a stable visual geography.
2. Use `GeoJSON` when the page needs ECharts geo registration, route anchors, region fills, or point data binding.
3. Use `TopoJSON` when the polygon set is large enough that lighter topology-aware assets materially improve slide weight.
4. Use vector-tile sources only for the small set of pages that truly need `MapLibre GL JS`.

### Prohibited Source Behavior

- no hand-drawn coastlines, borders, or country silhouettes inside slide HTML/CSS
- no ad hoc screenshot basemaps that cannot be restyled or annotated cleanly
- no opaque map assets whose projection, region names, or licensing status cannot be traced

### Source Readiness Checklist

Before using a basemap, confirm:

1. The geography is cropped to the actual story area.
2. Labels can be added without fighting the coastline or borders.
3. The source can be recolored to match the active style profile.
4. The source is light enough for presentation runtime and export.

## What Belongs Here

- Map-first page types
- Regional crop and scope decisions
- Overlay grammar for arrows, routes, hotspots, zones, and callouts
- Conflict / logistics / corridor / influence storytelling patterns
- Map + narrative card composition rules
- Label density and annotation hierarchy for map-heavy slides

## What Does Not Belong Here

- Generic chart selection logic
- Basic geo series configuration already handled by `ppt-chart-engine`
- Brand tokens and typography rules already handled by `ppt-brand-style-system`
- Generic page skeleton/layout constraints already handled by `ppt-slide-layout-library`

## Boundary Contract

### With `ppt-chart-engine`

`ppt-chart-engine` owns the geo-chart layer:

- choropleth / geo heatmap
- geo scatter
- geo lines / link series
- geographic data contracts
- low-level rendering and legend constraints

`ppt-map-storytelling` owns the narrative layer:

- why a map is needed on this page
- what geographic scope to show
- which overlays to combine
- how annotations and insight cards relate to the map
- whether the page is about territory, movement, exposure, control, or escalation

### With `ppt-slide-layout-library`

`ppt-slide-layout-library` owns reusable map-capable skeletons such as `map_overlay`.

`ppt-map-storytelling` decides how to fill those skeletons:

- which map crop and focal region to use
- where to place floating cards
- which arrow/callout grammar fits the story
- how much of the map should remain visible versus overlaid

Once a layout skeleton is chosen, the selected layout asset's `layout_contract` becomes binding for the Thinking and recovery phases.

At minimum, the Thinking file should mirror:

- `layout_contract_source`
- `narrative_fit_match`
- `overflow_recovery_order`
- `fallback_layouts`

Map-storytelling does not override these layout recovery rules. If map overlays, labels, or floating cards create density failure, reduce them in the order declared by the chosen layout's `overflow_recovery_order` before switching to any fallback layout.

## When to Use This Skill

- Direct geopolitical conflict expressed geographically
- Route, corridor, shipping lane, supply chain, or strike path pages
- Regional comparison where location matters more than ranking alone
- Global or regional footprint pages
- Pages where a plain geo chart would be correct but not expressive enough

## Decision Gate

Before using a map, verify all of the following:

1. Geography changes the meaning of the insight, not just its decoration.
2. The page needs spatial reasoning: adjacency, distance, route, region, exposure, or control.
3. A map will clarify the story more than a ranked list, matrix, or timeline.
4. The visible area can stay readable after overlays, labels, and narrative cards are added.

If any of these fail, prefer a non-map expression.

## Map Narrative Archetypes

### 1. Territory Snapshot

- Use when the insight is about regional state, exposure, or concentration.
- Default overlays: fills, hotspots, short callouts.

### 2. Route / Corridor Map

- Use when the insight is about movement, trade, logistics, supply, or transit risk.
- Default overlays: directional lines, checkpoints, bottlenecks, route labels.

### 3. Conflict Theater Map

- Use when the insight is about escalation, attack direction, force projection, or contested zones.
- Default overlays: strike arcs, pressure zones, conflict labels, risk markers.

### 4. Footprint / Network Map

- Use when the insight is about distributed presence across countries or regions.
- Default overlays: node markers, cluster labels, reach indicators.

## Overlay Grammar

### Allowed Overlay Types

- `fill_zone`: territorial emphasis or exposure shading
- `point_marker`: city, port, base, or node marker
- `route_line`: movement path or connection
- `direction_arrow`: directional emphasis or strike path
- `hotspot_callout`: high-attention labeled point
- `floating_card`: detached narrative/KPI card referencing map regions

### Overlay Rules

1. Use at most 2 primary overlay families on one map page before introducing floating cards.
2. Arrows imply movement or direction; do not use them for static region labeling.
3. Fill zones express area/state; do not overload them with exact numeric reading tasks.
4. Hotspot callouts should point to places the user cannot infer from geography alone.
5. Floating cards should summarize, not duplicate, on-map labels.

## Overlay to Component Routing

Use `assets/patterns.yml#overlay_component_contracts` to decide whether an overlay family should stay in map-storytelling, resolve to a component family, or escalate to `ppt-chart-engine`.

Working rule:

1. If the overlay is data-bound and needs geo series, tooltip, legend, or continuous value encoding, route it to `ppt-chart-engine`.
2. If the overlay is narrative scaffolding, directional emphasis, or detached summary, keep it in map-storytelling and pair it with the mapped component family.
3. Do not use a component family as a disguise for a chart obligation; do not use a geo chart when a narrative overlay would be clearer.

## Scope and Crop Rules

1. Default to the smallest geography that preserves the story.
2. Global maps are for reach, alignment, or multi-theater relationships, not local detail.
3. Regional maps are preferred for conflict, routes, chokepoints, and operational implications.
4. If the story depends on precise local adjacency, crop aggressively and reduce overlay count.

## Annotation Hierarchy

1. Primary labels: 1-3 items that define the story.
2. Secondary labels: supporting places or routes.
3. Context labels: only if the audience cannot orient themselves without them.

Do not label every visible place on the map.

## Composition Rules

1. The map should remain visually legible behind overlays and cards.
2. Narrative cards should anchor to distinct regions, not float without spatial relation.
3. If overlays dominate the page and the map becomes background noise, switch to a non-map layout.
4. Map pages should usually answer one spatial question only.

## Preferred Workflow

1. Choose the narrative archetype.
2. Choose geographic scope and crop.
3. Select overlay families.
4. Check `assets/patterns.yml#overlay_component_contracts` to route each overlay family toward chart layer, component family, or pure narrative overlay.
5. Hand skeleton selection to `ppt-slide-layout-library` and lock the chosen layout asset's `layout_contract`.
6. Mirror `layout_contract_source`, `overflow_recovery_order`, and `fallback_layouts` into the Thinking file.
7. Decide whether the page uses pure map, map + floating cards, or map + KPI panel within that layout contract.
8. Hand geo-series details to `ppt-chart-engine` if a geo chart layer is needed.

## Minimal Input Examples

Canonical minimal examples now live in `assets/examples.yml`.

`assets/examples.yml` now covers all four narrative archetypes with reusable minimum-input skeletons:

- `territory_snapshot`
- `route_corridor`
- `conflict_theater`
- `footprint_network`

Use:

- `minimal_examples.territory_snapshot_page` for regional exposure/state snapshots driven by editorial overlays
- `minimal_examples.echarts_geo_page` for geo-series pages driven by `ECharts Geo`
- `minimal_examples.svg_narrative_map_page` for editorial map pages driven by prepared `SVG` + HTML/SVG overlay
- `minimal_examples.footprint_network_page` for distributed-node and reach-pattern pages

Every map page input should declare at least:

- `narrative_archetype`
- `geographic_scope`
- `primary_question`
- `render_engine`
- `basemap_source`

If the page also selects a standard layout, it should additionally declare:

- `layout_key`
- `layout_contract_source`
- `overflow_recovery_order`
- `fallback_layouts`

## Dependencies

- `ppt-chart-engine`: geo series, geo chart contracts, rendering constraints
- `ppt-slide-layout-library`: map-capable page skeletons such as `map_overlay`
- `ppt-brand-style-system`: color, typography, semantic emphasis

## Resource Files

- `assets/patterns.yml`: pattern index for narrative archetypes, overlay rules, crop guidance, basemap source contract, and overlay-to-component routing
- `assets/examples.yml`: input skeleton index covering all four narrative archetypes, with render-engine, basemap, and overlay-routing examples
- `knowledge/templates/ppt-map-page-thinking-template.md`: reusable map-page thinking sample aligned with the PPT specialist workflow
