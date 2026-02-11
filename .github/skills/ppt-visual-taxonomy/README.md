````markdown
# ppt-visual-taxonomy Skill

## Purpose

Provides a unified visual type classification system shared by `ppt-content-planner` and `ppt-visual-designer`. This is the single source of truth for visual type names, annotation standards, and placeholder data templates. Any new visual type MUST be added here first to ensure consistency across all PPT agents.

## Core Features

- **3-level taxonomy**: Core types (10), Extended types (8), Domain-specific types (6)
- **Annotation format**: Standardized visual type tags for `slides.md`
- **Cognitive intent mapping**: Clear guidelines on when to use each type
- **Placeholder data templates**: Data structure examples for each visual type

## Usage

Agents reference this skill when:
1. **Content Planner**: Selecting visual types during slide outlining
2. **Visual Designer**: Validating visual type names and generating placeholder data
3. **Adding new types**: Follow the 3-level hierarchy and update this file first

---

## Visual Type Taxonomy

### Level 1 — Core Types (Most Slide Scenarios)

| Type | Use Case | Data Requirement |
|---|---|---|
| `bar_chart` | Categorical comparison | ≥2 categories × ≥1 series |
| `line_chart` | Trend over time/sequence | ≥3 data points × ≥1 series |
| `pie_chart` | Part-of-whole (≤6 slices) | ≥2 categories, values sum to 100% |
| `flow_diagram` | Process, pipeline, workflow | ≥3 steps with relationships |
| `architecture_diagram` | System structure, components | ≥2 layers/components |
| `table` | Structured multi-attribute comparison | ≥2 rows × ≥2 columns |
| `timeline` | Sequential events/milestones | ≥3 chronological items |
| `matrix` | Two-dimension classification | ≥2×2 grid with items |
| `comparison_table` | Side-by-side evaluation | ≥2 options × ≥3 criteria |
| `kpi_dashboard` | Feature metric overview | ≥2 KPIs with targets |

### Level 2 — Extended Types (Specialized Content)

| Type | Use Case | Data Requirement |
|---|---|---|
| `radar_chart` | Multi-dimension capability comparison | ≥3 dimensions with normalized values |
| `waterfall_chart` | Cumulative change / build-up | ≥3 steps, each with delta |
| `gantt_chart` | Project schedule, phases | ≥3 tasks with start/duration/status |
| `heatmap` | Density / intensity across 2 axes | ≥3×3 grid with values |
| `sankey_diagram` | Flow volume between stages | ≥2 stages, ≥3 flow paths |
| `scatter_plot` | Correlation of two variables | ≥5 data points (x, y [, size]) |
| `treemap` | Hierarchical part-of-whole | ≥3 nodes with sizes |
| `bubble_chart` | Three-dimension comparison | ≥3 items (x, y, size) |

### Level 3 — Domain-Specific Types

| Type | Domain | Use Case |
|---|---|---|
| `engineering_schematic` | Hardware / Manufacturing | Circuit topology, mechanical assembly |
| `simulation_plot` | R&D / Test Lab | FEA/CFD scalar fields, parameter sweeps |
| `bom_hierarchy` | Manufacturing / Procurement | Bill of Materials breakdown by cost/weight |
| `compliance_matrix` | Standards / Safety | Requirement-to-test cross-reference |
| `decision_tree` | Strategy / Risk | Sequential branching decisions |
| `network_topology` | Software / Infra | Service mesh, cloud architecture |

---

## Visual Annotation Format

### 2.1 Complete Annotation Example

```markdown
**VISUAL**:
Type: radar_chart
Title: "算法性能对比 — 五维评估"
Axes: [精度, 速度, 可解释性, 可扩展性, 成本效率]
Series:
  - name: "方案A: 基于规则"
    values: [40, 90, 95, 60, 80]
  - name: "方案B: ML模型"
    values: [85, 70, 30, 80, 50]
  - name: "方案C: 混合方案"
    values: [75, 80, 65, 85, 65]
Highlight: "方案C 综合得分最高 (74/100)"
cognitive_intent: "compare"
annotation_notes: "数据源: benchmarks/eval_20250115.csv, N=1000 样本"
```

### 2.2 Scope Guidelines

| Annotation Field | Required | Notes |
|---|---|---|
| `Type` | ✅ Always | Must be from §1 taxonomy |
| `Title` | ✅ Always | Descriptive, includes unit or context |
| `Data fields` | ✅ When data exists | Axes, Series, Categories, Items, etc. |
| `Highlight` | ✅ When key insight exists | One-sentence takeaway |
| `cognitive_intent` | ✅ On critical/high priority slides | See §3 |
| `annotation_notes` | Optional | Data source, methodology, caveats |

### 2.3 Content Scope Rules

Content-planner annotations MUST be **semantic** (what to show), NOT **visual** (how it looks):

#### Special Case: Gantt Chart Data Structure

For `gantt_chart` type, provide structured data instead of only mermaid code:

```json
{
  "type": "gantt_chart",
  "title": "Project Implementation Roadmap",
  "placeholder_data": {
    "gantt_data": {
      "timeline": {
        "start": "2026-02",
      ... (file continues)
### Level 2 — Extended Types (Specialized Content)

| Type | Use Case | Data Requirement |
|---|---|---|
| `radar_chart` | Multi-dimension capability comparison | ≥3 dimensions with normalized values |
| `waterfall_chart` | Cumulative change / build-up | ≥3 steps, each with delta |
| `gantt_chart` | Project schedule, phases | ≥3 tasks with start/end |
| `heatmap` | Density / intensity across 2 axes | ≥3×3 grid with values |
| `sankey_diagram` | Flow volume between stages | ≥2 stages, ≥3 flow paths |
| `scatter_plot` | Correlation of two variables | ≥5 data points (x, y [, size]) |
| `treemap` | Hierarchical part-of-whole | ≥3 nodes with sizes |
| `bubble_chart` | Three-dimension comparison | ≥3 items (x, y, size) |

### Level 3 — Domain-Specific Types

| Type | Domain | Use Case |
|---|---|---|
| `engineering_schematic` | Hardware / Manufacturing | Circuit topology, mechanical assembly |
| `simulation_plot` | R&D / Test Lab | FEA/CFD scalar fields, parameter sweeps |
| `bom_hierarchy` | Manufacturing / Procurement | Bill of Materials breakdown by cost/weight |
| `compliance_matrix` | Standards / Safety | Requirement-to-test cross-reference |
| `decision_tree` | Strategy / Risk | Sequential branching decisions |
| `network_topology` | Software / Infra | Service mesh, cloud architecture |

---

## Visual Annotation Format

### 2.1 Complete Annotation Example

```markdown
**VISUAL**:
Type: radar_chart
Title: "算法性能对比 — 五维评估"
Axes: [精度, 速度, 可解释性, 可扩展性, 成本效率]
Series:
  - name: "方案A: 基于规则"
    values: [40, 90, 95, 60, 80]
  - name: "方案B: ML模型"
    values: [85, 70, 30, 80, 50]
  - name: "方案C: 混合方案"
    values: [75, 80, 65, 85, 65]
Highlight: "方案C 综合得分最高 (74/100)"
cognitive_intent: "compare"
annotation_notes: "数据源: benchmarks/eval_20250115.csv, N=1000 样本"
```

### 2.2 Scope Guidelines

| Annotation Field | Required | Notes |
|---|---|---|
| `Type` | ✅ Always | Must be from §1 taxonomy |
| `Title` | ✅ Always | Descriptive, includes unit or context |
| `Data fields` | ✅ When data exists | Axes, Series, Categories, Items, etc. |
| `Highlight` | ✅ When key insight exists | One-sentence takeaway |
| `cognitive_intent` | ✅ On critical/high priority slides | See §3 |
| `annotation_notes` | Optional | Data source, methodology, caveats |

### 2.3 Content Scope Rules

Content-planner annotations MUST be **semantic** (what to show), NOT **visual** (how it looks):

| ✅ Correct (content-planner) | ❌ Wrong (visual-designer's job) |
|---|---|
| `Type: bar_chart` | `Width: 600px, Color: #2196F3` |
| `Highlight: "方案A效率最高"` | `Highlight animation: fadeIn 0.5s` |
| `cognitive_intent: "compare"` | `Layout: left-aligned, 2-column` |

---

## Cognitive Intent Reference

| Intent | Purpose | Typical Visual Types | Design Direction |
|---|---|---|---|
| `compare` | Show differences between options | bar_chart, radar_chart, comparison_table | Parallel layout, contrasting colors |
| `trend` | Show change over time | line_chart, waterfall_chart | Sequential palette, trend line emphasis |
| `composition` | Show parts of a whole | pie_chart, treemap, sankey_diagram | Harmonious palette, proportional sizing |
| `distribution` | Show data spread | heatmap, scatter_plot, bubble_chart | Gradient palette, density mapping |
| `relationship` | Show connections | flow_diagram, architecture_diagram, network_topology | Graph layout, directional indicators |
| `inform` | Present key metrics | kpi_dashboard, table | Bold typography, dashboard layout |
| `persuade` | Drive a decision point | matrix, decision_tree | Highlight preferred option, red/green |

---

## Visual Type Selection Guide

```
Need to show…
├── Comparison between items?
│   ├── ≤5 options, ≥3 dimensions → radar_chart
│   ├── Categorical with values → bar_chart
│   └── Side-by-side evaluation → comparison_table
├── Change over time?
│   ├── Continuous trend → line_chart
│   ├── Cumulative impact → waterfall_chart
│   └── Project phases → gantt_chart
├── Structure/Flow?
│   ├── Process steps → flow_diagram
│   ├── System layers → architecture_diagram
│   ├── Data/resource flow volumes → sankey_diagram
│   └── Decision branching → decision_tree
├── Proportions?
│   ├── ≤6 categories → pie_chart
│   ├── Hierarchical → treemap
│   └── Flow between stages → sankey_diagram
├── Data density?
│   ├── Multi-attribute records → table
│   ├── 2D intensity → heatmap
│   └── 3D comparison → bubble_chart
└── Metrics overview?
    └── ≥2 KPIs → kpi_dashboard
```

---

## Placeholder Data Templates

When source document doesn't provide exact numbers, generate **realistic placeholder data** following these templates:

### 5.1 Chart Placeholder

```yaml
Type: bar_chart
Title: "[指标名称] 对比分析"
Categories: ["方案A", "方案B", "方案C"]
Values: [85, 72, 91]
Unit: "%"
Source: "基于 [来源文档] §X.Y 数据估算"
```

### 5.2 Diagram Placeholder

```yaml
Type: flow_diagram
Title: "[流程名称] 核心流程"
Steps:
  - id: 1, label: "输入/采集"
  - id: 2, label: "处理/转换"
  - id: 3, label: "验证/测试"
  - id: 4, label: "输出/部署"
Connections: [[1,2], [2,3], [3,4]]
```

### 5.3 Timeline Placeholder

```yaml
Type: timeline
Title: "[项目名称] 里程碑规划"
Items:
  - date: "Q1 2025", event: "立项 & 需求确认", status: "completed"
  - date: "Q2 2025", event: "原型开发", status: "in_progress"
  - date: "Q3 2025", event: "测试验证", status: "planned"
  - date: "Q4 2025", event: "量产部署", status: "planned"
```

### 5.4 KPI Dashboard Placeholder

```yaml
Type: kpi_dashboard
Title: "核心指标概览"
KPIs:
  - name: "效率", value: "95%", target: "≥90%", status: "green"
  - name: "成本", value: "$1.2M", target: "≤$1.5M", status: "green"
  - name: "进度", value: "72%", target: "80%", status: "yellow"
```

---

## Content Ceiling Rules

| Component Type | Max Items per Slide | Split Strategy |
|---|---|---|
| `bullets` | 7 | Split by theme, add transition |
| `comparison_items` | 5 | Group by category |
| `risks` | 5 | Prioritize by severity |
| `kpis` | 4 | Group by domain |
| `timeline_items` | 6 | Split by phase |
| `action_items` | 5 | Group by owner/priority |
| `table_data rows` | 8 | Split by category, show top-N |

**Rule**: Pre-split at content planning time. Do NOT rely on specialist auto-split; it produces worse narrative flow.
