# Slide {N}: Thinking

## 1. Core Task and Reasoning (Mission & Reasoning)

- **Goal**: State the single reading task this slide must accomplish.
- **Information Structure**: Explain whether the slide is evidence-led, comparison-led, process-led, or conclusion-led.
- **Data Strategy**: State whether the slide is chart-led, component-led, layout-led, or mixed, and why.
- **Layout Trade-offs**:
  - *Option A*: Describe the rejected layout or encoding path.
  - *Option B*: Describe the chosen layout and why it is superior for this slide.

---

## 2. Execution Specs (Execution Specs)

### 2.1 Layout Anchor (Layout Anchor)

- **Layout Key**: side_by_side | dashboard_grid | hybrid | comparison | process | conclusion | executive_summary
- **Layout Contract Source**: `skills/ppt-slide-layout-library/assets/layouts/<layout_key>.yml#layout_contract`
- **Narrative Fit Match**: which `layout_contract.narrative_fit` label this slide satisfies
- **Required Fields Check**: confirm the page has all fields required by the chosen layout contract
- **Overflow Recovery Order**: copy the ordered recovery path from the chosen layout contract
- **Fallback Layouts**: copy the allowed fallback layouts from the chosen layout contract
- **Primary Region Strategy**: what the main reading region does
- **Secondary Region Strategy**: what the supporting region does

### 2.2 Content Encoding (Content Encoding)

- **Primary Encoding**: chart | components | text block | table | mixed
- **Source**: file, note, dataset, or research source
- **Filter Logic**: what is included and excluded
- **Mapping**: which facts go to headline / chart / cards / footer notes

### 2.3 Chart Contract Check (Chart Contract Check)

- **Chart Family**: required when `Primary Encoding = chart`
- **Contract Fields**: required when `Primary Encoding = chart`
- **Null Policy**: required when `Primary Encoding = chart`
- **Contract Source**: `skills/ppt-chart-engine/assets/charts.yml#thinking_contracts.<chart_family>` when chart-led
- **Fallback Plan**: required when `Primary Encoding = chart`

### 2.4 Component Semantic Resolution (Component Semantic Resolution)

- **Component Selection**: list the standard components used on the page, if any
- **Semantic Roles**: for each standard component, list the semantic roles used by the payload
- **Resolver Source**: `skills/ppt-brand-style-system/assets/component_semantic_mappings.yml` when semantic_payload is present
- **Fallback Policy**: resolver first, then safe fallback payload only if needed

### 2.5 Visual Props (Visual Props)

- **Style Profile**: active style profile or intended page tone
- **Density**: compact | default | relaxed
- **Highlights**: emphasis zones, separators, chips, arrows, glow, or KPI anchors

### 2.6 Narrative (Narrative)

- **Headline**: one clear headline
- **Insight**: one sentence explaining why the page matters

### 2.7 Layout Recovery and Fallback (Layout Recovery)

- **Recovery Trigger**: what content-density or structure condition would trigger recovery
- **Recovery Action**: which first action from `Overflow Recovery Order` will be tried before layout switching
- **Fallback Trigger**: when to stop recovery and switch to a layout in `Fallback Layouts`

## Example Filled Pattern

- **Layout Key**: side_by_side
- **Layout Contract Source**: `skills/ppt-slide-layout-library/assets/layouts/side_by_side.yml#layout_contract`
- **Narrative Fit Match**: `scenario_tradeoff`
- **Required Fields Check**:
  `layout_key`, `comparison_axis`, `option_count`, `scoring_basis`,
  `recommendation_logic`, `evidence_type`, `fallback_plan`
- **Overflow Recovery Order**:
  `reduce_chart_width_pressure` -> `reduce_card_copy_density` ->
  `move_secondary_note_to_footer` -> `downgrade_to_data_chart`
- **Fallback Layouts**: `comparison`, `data_chart`
- **Primary Encoding**: components
- **Source**: research summary + scenario notes
- **Component Selection**: `Metric_Big`, `Card_Accent`, `List_Icon`
- **Semantic Roles**:
  - `Metric_Big` -> `emphasis_role: critical`, `value_role: primary_text`
  - `Card_Accent` -> `emphasis_role: warning`, `surface_role: elevated`
  - `List_Icon` -> `emphasis_role: positive`, `text_role: primary_text`
- **Resolver Source**: `skills/ppt-brand-style-system/assets/component_semantic_mappings.yml`
