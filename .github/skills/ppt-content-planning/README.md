# ppt-content-planning Skill

## Purpose

Provides structured schemas, templates, algorithms, and quality standards for slide content planning. Designed for `ppt-content-planner` agent to reference when creating `slides.md` files with audience-tailored content.

## Core Features

- **Front-matter schema** for audience profiling and content adaptation
- **Content planning algorithms** for decision extraction and evidence organization
- **Output template examples** for slide structure and speaker notes
- **Quality checklist** (10-item validation)
- **Design philosophy guide** (McKinsey Pyramid, Presentation Zen, etc.)

## Usage

Agent references this skill when:
1. Creating new `slides.md` files with appropriate front-matter
2. Extracting key decisions from source documents
3. Organizing content with cognitive-load-aware structure
4. Generating speaker notes with technical traceability
5. Validating output before handoff to `ppt-visual-designer`

---

## Output Specifications

### 1.1 slides.md Front-Matter Template

```yaml
---
title: "演示标题"
author: "团队"
date: "YYYY-MM-DD"
language: "zh-CN"

# Audience Profile
audience:
  type: "technical_reviewers"          # executive / technical_reviewers / general_audience / academic
  knowledge_level: "expert"            # novice / intermediate / expert
  decision_authority: "high"           # low / medium / high
  time_constraint: "30min"
  expectations:
    - "Detailed technical data with traceability"
    - "Clear decision rationale with alternatives considered"

# Content Adaptation
content_strategy:
  technical_depth: "high"              # low / medium / high
  visual_complexity: "medium"
  language_style: "formal_technical"   # conversational / formal_business / formal_technical
  data_density: "high"
  bullet_limit: 5

# Title Slide Metadata (REQUIRED for specialist rendering)
title_slide:
  subtitle: "一句话副标题"
  author: "作者/团队名"
  date: "YYYY-MM-DD"
  version: "v0.1"

# Design Philosophy Recommendation
recommended_philosophy: "McKinsey Pyramid"
philosophy_rationale: "..."
alternative_philosophies:
  - name: "Presentation Zen"
    reason_rejected: "..."

# Story Structure (SCQA) — Hierarchical
story_structure:
  framework: "SCQA"
  macro:
    situation: [1]
    complication: [2, 3]
    question: "..."
    answer: [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    next_steps: [15, 16]
  sections:                            # For ≥15 slide decks
    - label: "核心方案"
      slides: [4, 5, 6, 7]
      section_role: "answer"
      internal_logic: "MVP范围 → 架构 → 算法 → 性能"
  transitions:
    - from: "核心方案"
      to: "验证与风险"
      expected: "从技术方案自然过渡到验证与风险评估"

# Timing & Pacing Analysis
timing:
  total_time: "30min"
  total_slides: 16
  avg_time_per_slide: "1.9 min"
  section_allocation:
    - section: "开场+结论"
      slides: [1, 2, 3]
      allocated_time: "5 min"
      pacing_status: "OK"
  warnings: []

# KPI Traceability
kpi_traceability:
  tracked_kpis: []
  # Example:
  # - name: "交互延迟"
  #   target: "p95 < 100ms"
  #   referenced_in: [4, 8, 14]
  #   consistency: "PASS"

# QA Metadata
content_qa:
  overall_score: 92
  key_decisions_present: true
  key_decisions_location: "Slide 2"
  speaker_notes_coverage: 94
  visual_coverage: 50
  scqa_complete: true
  kpi_consistency: true
  timing_feasible: true
---
```

### 1.2 Slide Schema (Per-Slide Template)

```markdown
## Slide N: [topic]
**Title**: [Assertion-style title ≤ 10 words]

**Content**:
- [bullet 1 ≤ 15 words]
- [bullet 2 ≤ 15 words]
- [bullet 3 ≤ 15 words]

**SPEAKER_NOTES**:
(see §2 Speaker Notes Standards)

**VISUAL**:
(see skills/ppt-visual-taxonomy §2 Visual Annotation Format)

**COMPONENTS** (REQUIRED):
(see §1.4 Component Types)

**METADATA**:
{"slide_type": "...", "slide_role": "...", "requires_diagram": true, "priority": "critical"}
```

### 1.3 slides_semantic.json Sections Array

```json
{
  "schema": "standards/slides-render-schema.json@v1",
  "sections": [
    {"id": "A", "title": "市场与战略", "start_slide": 4, "accent": "primary"},
    {"id": "B", "title": "技术基础", "start_slide": 8, "accent": "secondary"}
  ],
  "slides": [...]
}
```

### 1.4 Component Types

| Component Type     | Use When                                     |
|--------------------|----------------------------------------------|
| `kpis`             | Quantitative metrics, KPIs, performance data |
| `decisions`        | Key decisions with status/rationale          |
| `comparison_items` | Side-by-side technology/option comparisons   |
| `table_data`       | Structured tabular data (headers + rows)     |
| `timeline_items`   | Milestones, phases, roadmap events           |
| `risks`            | Risk items with severity/mitigation          |
| `action_items`     | Action items, next steps, checklist items    |
| `callouts`         | Highlighted insights, key takeaways          |

### 1.4.1 Component Field Requirements

**⚠️ CRITICAL**: All components MUST strictly follow `standards/slides-render-schema.json@v1` field definitions.

**decisions** component schema (from `decision_item` in schema):
```json
{
  "decisions": [
    {
      "id": "D1",                    // ✅ REQUIRED: decision identifier
      "title": "批准示范场景与首轮预算", // ✅ REQUIRED: decision title (NOT "label")
      "description": "快速验证市场与工程假设", // ✅ OPTIONAL: brief elaboration (NOT "rationale")
      "owner": "CTO",                 // ✅ OPTIONAL: decision owner/approver
      "budget": "$0.5-1.5M",          // ✅ OPTIONAL: budget range
      "priority": "P0",               // ✅ OPTIONAL: P0/P1/P2
      "timeline": "立即",              // ✅ OPTIONAL: expected timeline (NOT "time_to_decision")
      "status": "pending"             // ✅ OPTIONAL: pending/approved/rejected (NOT "proposed")
    }
  ]
}
```

**Common Field Mapping Errors** (❌ Forbidden):
- ❌ `"label"` → use `"title"` instead
- ❌ `"rationale"` → use `"description"` instead
- ❌ `"time_to_decision"` → use `"timeline"` instead
- ❌ `"status": "proposed"` → use `"pending"` instead

**kpis** component schema:
```json
{
  "kpis": [
    {
      "label": "效率",              // ✅ REQUIRED: metric name
      "value": "≥98%",              // ✅ REQUIRED: current value (string or number)
      "unit": "%",                  // ✅ OPTIONAL: unit
      "delta": "↑5%",               // ✅ OPTIONAL: trend indicator
      "status": "normal"            // ✅ OPTIONAL: normal/warning/critical (default: normal)
    }
  ]
}
```

**comparison_items** component schema:
```json
{
  "comparison_items": [
    {
      "label": "EV 快充",           // ✅ REQUIRED: item name
      "attributes": {                // ✅ REQUIRED: key-value pairs (≥3 attributes)
        "市场规模": "$120–180M",
        "增长驱动": "政策补贴+基础设施扩建",
        "技术成熟度": "中-高",
        "不确定性": "充电标准分裂"
      },
      "highlight": false             // ✅ OPTIONAL: emphasize this item (default: false)
    }
  ]
}
```

**comparison_items quality rules:**
- **Completeness**: If the source document lists N comparable items, ALL N must appear as comparison_items. Do NOT truncate.
- **Richness**: Each item MUST have ≥ 3 attributes in the `attributes` object. Attributes should be descriptive text or labeled numbers, NOT bare numerics.
- **Schema**: All data fields MUST go inside `attributes: {}`. Do NOT put fields like `min`/`max`/`score` as top-level keys alongside `label`.
- **slide_type selection**: If items only have 1–2 numeric attributes (e.g., just min/max), use `slide_type: "data-heavy"` with `table_data` or a chart instead of comparison cards.
```

**timeline_items** component schema:
```json
{
  "timeline_items": [
    {
      "phase": "Phase 1: 立项",      // ✅ REQUIRED: phase name
      "period": "0–3 个月",           // ✅ REQUIRED: time range
      "description": "完成示范选址与预算批准", // ✅ OPTIONAL: key deliverables
      "milestone": "预算批准",         // ✅ OPTIONAL: gate/acceptance criteria
      "status": "planned"             // ✅ OPTIONAL: planned/active/completed (default: planned)
    }
  ]
}
```

**See `standards/slides-render-schema.json@v1` for complete definitions of all component types.**

### 1.5 slide_type / Component Alignment Rule

| Primary component in `components` | Required `slide_type` | Forbidden `slide_type` |
|---|---|---|
| `comparison_items` | `comparison` | `bullet-list` |
| `table_data` (≥3 rows) | `data-heavy` | `bullet-list` |
| `timeline_items` | `timeline` or `gantt` | `bullet-list` |
| `decisions` | `two-column` or `decision` | `bullet-list` |
| `risks` | `matrix` | `bullet-list` |
| `action_items` | `call_to_action` | `bullet-list` |
| `kpis` only | `data-heavy` | — |

### 1.6 Content Ceiling Rule

| Component Type | Max Items | Split Threshold |
|---|---|---|
| `bullets` | 7 | ≥8 → split |
| `comparison_items` | 5 | ≥6 → split |
| `risks` | 5 | ≥6 → split |
| `kpis` | 4 | ≥5 → split |
| `timeline_items` | 6 | ≥7 → split |
| `action_items` | 5 | ≥6 → split |
| `table_data rows` | 8 | ≥9 → split |

Pre-split at generation time produces better narrative flow than relying on specialist auto-split.

### 1.7 Section Divider Slides (for ≥15 slide decks)

```markdown
## Slide N: [Section Name]
**Title**: [Section Name — e.g., "技术基础"]
**Content**:
- [Concatenated titles: "• 算法架构 • 性能指标 • 部署方案"]
  **FORBIDDEN**: Generic placeholder text like "本节覆盖 X 相关内容".
**METADATA**:
{"slide_type": "section_divider", "slide_role": "transition", "section_id": "B", "section_index": 2, "total_sections": 6}
```

### 1.8 Callout Accent Rule

Every `callout` MUST include an `accent` field. Cycle per section: Section A → `primary`, Section B → `secondary`, Section C → `tertiary`, repeat.

### 1.9 Data Integrity Rules

#### 1.9.1 No Data Fabrication
- **ALL numerical values** in `components` and `visual.placeholder_data` MUST originate from the source document.
- ❌ **FORBIDDEN**: Inventing scores like `"impact": 95, "feasibility": 80` when the source document contains no such numbers.
- ✅ **ALLOWED**: Using qualitative labels ("高"/"中"/"低") when source supports but lacks exact numbers.
- ✅ **ALLOWED**: Extracting explicit numbers from source (e.g., "$0.5-1.5M" from budget section).
- **Self-check**: For every number in the output, can you cite the exact source paragraph? If not, remove it.

#### 1.9.2 Content[] Deduplication with Components
- When `components` fully represent the slide's structured data, `content[]` MUST be empty or contain ONLY supplementary text NOT already in components.
- ❌ **FORBIDDEN example** (Slide 2 anti-pattern):
  ```json
  "content": ["批准示范场景与首轮预算", "批准材料验证与中试计划", "授权参与标准化"],
  "components": { "decisions": [{"title": "批准示范场景与首轮预算", ...}, ...] }
  ```
  ❌ **Problem**: Same text in `content[]` AND `components.decisions[].title` → renders twice.
- ✅ **CORRECT**: Set `content: []` when components cover all slide content.

#### 1.9.3 Components vs Visual — Single Source of Truth
- Each data item MUST appear in exactly ONE place: either `components` OR `visual.placeholder_data`.
- ❌ **FORBIDDEN**: Decision data in both `components.decisions[]` AND `visual.placeholder_data.chart_config.series[]`.
- ❌ **FORBIDDEN**: Comparison items in `components.comparison_items[]` AND the same data repeated in `visual.placeholder_data.chart_config`.
- **Decision rule**:
  - Data has a matching component type → put in `components`, set `visual.type: "none"`
  - Data is a pure chart/diagram with no matching component type → put in `visual.placeholder_data`
  - Data has BOTH a qualitative dimension (descriptions, pros/cons) AND a complementary numeric chart → put qualitative in `comparison_items` and numeric chart in `visual.placeholder_data`, ensuring items are NOT merely repeated between the two. The renderer supports a hybrid cards+chart layout for this case.

#### 1.9.5 Comparison Items Completeness & Richness
- When the source document lists N comparable items, `comparison_items` MUST include ALL N — do NOT truncate.
- Each comparison_item MUST have ≥ 3 meaningful attributes inside `attributes: {}`.
- ❌ **FORBIDDEN**: Sparse items with only 1-2 bare numeric fields (e.g., `{"label": "EV", "min": 120, "max": 180}`). This produces nearly empty cards with large whitespace.
- ✅ **REQUIRED**: Rich attributes drawn from the source document (e.g., `{"label": "EV 快充", "attributes": {"市场规模": "$120–180M", "增长驱动": "政策+基础设施", "成熟度": "中-高", "风险": "标准分裂"}}`).
- If source data is purely numeric with < 3 attributes per item, prefer `slide_type: "data-heavy"` with `table_data` or chart instead of comparison cards.

#### 1.9.4 Title Slide Restrictions
- `slide_type: "title"` slides MUST have `components: {}` (empty object).
- KPIs from speaker notes MUST NOT be promoted to title slide components. Place them on a dedicated KPI slide instead.

---

## Speaker Notes Standards

### 2.1 Required Fields (200-300 words per slide)

| # | Field | Length | Content |
|---|---|---|---|
| 1 | **Summary** | 1-2 sentences | What this slide says (main message in plain language) |
| 2 | **Rationale** | 2-3 sentences | Why this matters; decision reasoning; connection to SCQA |
| 3 | **Evidence** | 2-3 sentences | Data source, methodology, alternatives considered, caveats |
| 4 | **Audience Action** | 1-2 sentences | What to remember/decide/ask |
| 5 | **Risks/Uncertainties** | 1 sentence (optional) | Assumptions, unknowns, edge cases |

### 2.2 Optional Extension Fields

Auto-recommend based on slide content patterns:

| Extension Field | Triggers | Example |
|---|---|---|
| **KPIs** | ≥, ≤, target, 目标, KPI, p95, MTBF | 效率≥98%, MTBF≥100k h |
| **Budget/Investment** | USD, ¥, 预算, budget, 拨款, capex, opex | USD 0.5–1.5M / 站点 |
| **Acceptance Criteria** | 验收, acceptance, 放行, go/no-go | 样机效率≥95% 后拨付下阶段 |
| **Milestone Checkpoints** | 里程碑, milestone, 阶段, phase, Q1 | 0-3月:立项, 3-9月:样机 |
| **Cross-References** | 参见 Slide, see Slide, 见附录 | 参见 Slide 19 KPI 仪表盘 |
| **Formulae/Methodology** | 方程, equation, RMSE, Steinmetz | Steinmetz 方程, RMSE≤5% |

### 2.3 Example

```markdown
**SPEAKER_NOTES**:
**Summary**: MVP prioritizes core interactive features over advanced AI and real-time collaboration.
**Rationale**: Must-have features enable basic editing with <50ms latency, covering 80% of user use cases. Optional features add 10x complexity and can be phased in after MVP validation.
**Evidence**: User research (N=50) shows 80% of editing tasks use only layers, brush, and basic filters. AI 抠图/修复 has 10x complexity vs basic filters. WebGPU shows <20% gain.
**Audience Action**: Approve MVP scope as defined. Confirm 7-week delivery timeline.
**Risks/Uncertainties**: Scope creep risk. Performance assumptions need validation with 10MB+ images.
```

---

## Key Decisions Extraction

### 3.1 Universal Identification Patterns

**Decision verbs:**
- Chinese: 决策 / 选择 / 采用 / 推荐 / 优先 / 决定 / 放弃 / 排除 / 建议
- English: decision / chose / instead of / vs / trade-off / prioritize / recommend / adopt / reject

**Scope & phasing markers:**
- 必须 vs 可选 / MVP / Phase 1 / 短期 / 中期 / 长期
- 示范 (pilot) vs 量产 (mass production) vs 规模化 (scale-up)

**Comparison & trade-off markers:**
- vs / 对比 / 权衡 / 取舍 / 优点/缺点 / pros/cons / 成本/收益

**Risk & mitigation markers:**
- 为了避免 / 缓解 / 降级 / 监控 / 风险 / 不确定性 / 假设 / 验证

### 3.2 Domain Extension Packs

Use `skills/domain-keyword-detection/` for automatic domain detection:

```bash
python3 skills/domain-keyword-detection/bin/domain_detector.py detect \
  --input docs/source.md --threshold 0.3 --output json
```

Supported domains: `software`, `hardware`, `manufacturing`, `standards`, `business`, `biotech`.

Run domain detection **before** decision extraction. Report `activated_packs` in `content_qa_report.json`.

### 3.3 Decision Completeness Validation

Every key decision MUST have:
1. **Decision statement** (one sentence, actionable)
2. **Rationale** (why this choice? 1-2 bullets with evidence)
3. **Alternatives considered** (1+ bullet)
4. **Risks** (1+ bullet)
5. **Success criteria** (optional)

### 3.4 Key Decisions Slide Template

```markdown
# 关键决策 (Key Decisions)

**Decision 1: 采用 McKinsey Pyramid 演示哲学**
- **Rationale**: 技术评审听众期望数据和可追溯性
- **Alternatives Considered**: Presentation Zen (rejected: 数据密度不足)
- **Risks**: 数据过载可能降低可读性
- **Success Criteria**: 评审通过率 ≥ 80%
```

---

## Design Philosophy Selection Guide

| Philosophy | Audience | Content | Structure | Bullet Limit | Visual Style |
|---|---|---|---|---|---|
| **McKinsey Pyramid** | Executives, tech reviewers, decision-makers | Data-heavy, analytical | Conclusion-first, hierarchical, MECE | ≤5 | Charts, tables, architecture |
| **Presentation Zen** | General public, inspirational | Emotional, narrative-driven | Story arc, Hero's Journey, one idea/slide | ≤2 (prefer 0) | High-impact photography |
| **Guy Kawasaki (10/20/30)** | Investors, time-constrained | High-level pitch, persuasion | 10 slides, 20 min, 30pt min font | ≤3 | Simple charts, product screenshots |
| **Assertion-Evidence** | Academic, scientific, research | Research findings, evidence-based | Assertion title + evidence visual | ≤7 | Data plots, experiment results |

---

## Quality Assurance

### 5.1 Content QA Checklist

**Audience & Philosophy (15%)**:
- [ ] Audience profile complete (type, knowledge_level, decision_authority, expectations)
- [ ] Philosophy recommended with rationale + ≥1 rejected alternative

**Story Structure (20%)**:
- [ ] SCQA structure defined and complete
- [ ] Hierarchical SCQA for ≥15 slides (macro + section-level + transitions)
- [ ] Pyramid Principle applied: conclusion-first

**Key Decisions (15%)**:
- [ ] Key Decisions slide in slides 2-3, ≥2 decisions
- [ ] Each decision has statement + rationale + alternatives + risks
- [ ] Domain extension packs activated and reported

**Content Quality (15%)**:
- [ ] Assertion-style titles (≤10 words)
- [ ] Bullets within limit (exec ≤3, tech ≤5, academic ≤7)
- [ ] Speaker notes ≥90% coverage with structured template
- [ ] Extension fields auto-recommended where applicable

**Content Density & Components (10%)**:
- [ ] Component coverage ≥90% of content slides
- [ ] slide_type / component alignment — 0 violations
- [ ] bullet-list ratio ≤30%
- [ ] Content ceiling rule respected (no over-stuffed slides)

**Visual Annotations (10%)**:
- [ ] Visual types from taxonomy (not ad-hoc strings)
- [ ] Cognitive intent on critical/high priority visuals
- [ ] Placeholder data for charts/diagrams

**KPI Traceability (5%)**: Cross-slide KPI consistency
**Timing & Pacing (5%)**: ≤1.5 min/slide average
**Deliverable Completeness (5%)**: slides.md + slides_semantic.json + content_qa_report.json all present

### 5.2 content_qa_report.json Schema

```json
{
  "overall_score": 92,
  "timestamp": "2026-01-28T10:30:00Z",
  "checks": {
    "audience_profile": {"status": "PASS", "details": "..."},
    "philosophy_recommendation": {"status": "PASS", "recommended": "McKinsey Pyramid"},
    "key_decisions_present": {"status": "PASS", "location": "Slide 2", "count": 2},
    "scqa_structure": {"status": "PASS"},
    "bullet_counts": {"status": "PASS", "limit": 5, "violations": []},
    "speaker_notes_coverage": {"status": "PASS", "coverage_percent": 94},
    "visual_annotations": {"status": "PASS", "total_visuals": 8, "critical_visuals": 3},
    "assertion_style_titles": {"status": "PASS"}
  },
  "domain_packs_activated": ["hardware", "standards"],
  "kpi_traceability": {"tracked_kpis": [], "consistency": "PASS"},
  "timing_analysis": {"avg_time_per_slide": "1.9min", "warnings": []},
  "warnings": [],
  "fix_suggestions": []
}
```

**Scoring weights**: Audience 15%, Structure 20%, Decisions 15%, Content 15%, Density 10%, Visuals 10%, KPI 5%, Timing 5%, Completeness 5%.

**Pass threshold**: overall_score ≥ 70.

**Critical fail (block handoff)**: Missing audience profile, missing philosophy, missing Key Decisions, broken SCQA, speaker notes <80%, missing slides_semantic.json, KPI inconsistency.

---

## Example Prompts

### Technical Review
```
"Analyze `docs/online-ps-algorithm-v1.md` for developer audience (expert, high authority, 30min).
Emphasize architecture decisions and risk mitigation. Produce slides.md + slides_semantic.json + content_qa_report.json."
```

### Executive Pitch
```
"Create ≤12 slide pitch for C-level (business-focused, 20min).
Summarize roadmap, investment ask, ROI. Guy Kawasaki style, ≤3 bullets."
```

### Industrial / Hardware Review
```
"Analyze MFT report for engineering management (expert, 30min, 25-30 slides).
Activate Power Electronics + Manufacturing + Standards packs.
Hierarchical SCQA across 6 sections. Use radar, waterfall, kpi_dashboard, engineering_schematic visual types."
```
