# Slide {N}: Thinking

## 1. 核心任务与推理 (Mission & Reasoning)

- **目标**: State the single reading task this slide must accomplish.
- **信息结构**: Explain whether the slide is evidence-led, comparison-led, process-led, or conclusion-led.
- **数据策略**: State whether the slide is chart-led, component-led, layout-led, or mixed, and why.
- **布局权衡**:
  - *方案 A*: Describe the rejected layout or encoding path.
  - *方案 B*: Describe the chosen layout and why it is superior for this slide.

---

## 2. 执行规格 (Execution Specs)

### 2.1 页面骨架 (Layout Anchor)

- **Layout Key**: side_by_side | dashboard_grid | hybrid | comparison | process | conclusion | executive_summary
- **Layout_Contract_Source**: `skills/ppt-slide-layout-library/assets/layouts/<layout_key>.yml#layout_contract`
- **Narrative_Fit_Match**: which `layout_contract.narrative_fit` label this slide satisfies
- **Required_Thinking_Fields_Check**: confirm the page has all fields required by the chosen layout contract
- **Overflow_Recovery_Order**: copy the ordered recovery path from the chosen layout contract
- **Fallback_Layouts**: copy the allowed fallback layouts from the chosen layout contract
- **Primary Region Strategy**: what the main reading region does
- **Secondary Region Strategy**: what the supporting region does

### 2.2 内容编码 (Content Encoding)

- **Primary Encoding**: chart | components | text block | table | mixed
- **Source**: file, note, dataset, or research source
- **Filter Logic**: what is included and excluded
- **Mapping**: which facts go to headline / chart / cards / footer notes

### 2.3 图表契约判断 (Chart Contract Check)

- **Chart_Family**: required when `Primary Encoding = chart`
- **Contract_Fields**: required when `Primary Encoding = chart`
- **Null_Policy**: required when `Primary Encoding = chart`
- **Contract_Source**: `skills/ppt-chart-engine/assets/charts.yml#thinking_contracts.<chart_family>` when chart-led
- **Fallback_Plan**: required when `Primary Encoding = chart`

### 2.4 组件语义解析 (Component Semantic Resolution)

- **Component_Selection**: list the standard components used on the page, if any
- **Semantic_Roles**: for each standard component, list the semantic roles used by the payload
- **Resolver_Source**: `skills/ppt-brand-style-system/assets/component_semantic_mappings.yml` when semantic_payload is present
- **Fallback_Policy**: resolver first, then safe fallback payload only if needed

### 2.5 视觉细节 (Visual Props)

- **Style**: active style profile or intended page tone
- **Density**: compact | default | relaxed
- **Highlights**: emphasis zones, separators, chips, arrows, glow, or KPI anchors

### 2.6 叙事文案 (Narrative)

- **Headline**: one clear headline
- **Insight**: one sentence explaining why the page matters

### 2.7 布局恢复与降级 (Layout Recovery)

- **Recovery Trigger**: what content-density or structure condition would trigger recovery
- **Recovery Action**: which first action from `Overflow_Recovery_Order` will be tried before layout switching
- **Fallback Trigger**: when to stop recovery and switch to a layout in `Fallback_Layouts`

## Example Filled Pattern

- **Layout Key**: side_by_side
- **Layout_Contract_Source**: `skills/ppt-slide-layout-library/assets/layouts/side_by_side.yml#layout_contract`
- **Narrative_Fit_Match**: `scenario_tradeoff`
- **Required_Thinking_Fields_Check**:
  `layout_key`, `comparison_axis`, `option_count`, `scoring_basis`,
  `recommendation_logic`, `evidence_type`, `fallback_plan`
- **Overflow_Recovery_Order**:
  `reduce_chart_width_pressure` -> `reduce_card_copy_density` ->
  `move_secondary_note_to_footer` -> `downgrade_to_data_chart`
- **Fallback_Layouts**: `comparison`, `data_chart`
- **Primary Encoding**: components
- **Source**: research summary + scenario notes
- **Component_Selection**: `Metric_Big`, `Card_Accent`, `List_Icon`
- **Semantic_Roles**:
  - `Metric_Big` -> `emphasis_role: critical`, `value_role: primary_text`
  - `Card_Accent` -> `emphasis_role: warning`, `surface_role: elevated`
  - `List_Icon` -> `emphasis_role: positive`, `text_role: primary_text`
- **Resolver_Source**: `skills/ppt-brand-style-system/assets/component_semantic_mappings.yml`
