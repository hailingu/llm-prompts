# PPT Agent Collaboration Protocol

**Purpose**: Define the collaboration workflow, iteration limits, and quality gates for the PPT creation team to ensure efficient and high-quality presentation generation.

**Version**: 1.0  
**Last Updated**: 2026-01-28

---

## Overview

The PPT creation follows a **specialist-driven collaboration model** where each agent has a clear, focused responsibility, aligned to real-world creative industry workflows rather than software development patterns.

**Design Philosophy**:
- Visual designers handle both layouts and charts (like Apple Keynote team)
- Content strategists do self-QA on content quality (like McKinsey consultants)
- Creative directors coordinate and make final decisions (like IDEO/Pentagram)

---

## Team Structure (3-Agent Architecture)

```mermaid
graph TB
    User[👤 用户请求] --> CD[🎯 ppt-creative-director<br/>流程协调 + 质量门控]
    
    CD --> CP[📝 ppt-content-planner<br/>内容策略 + 内容质量]
    CP --> VD[🎨 ppt-visual-designer<br/>视觉设计 + 图表 + 视觉质量]
    
    VD --> Gen[⚙️ ppt_generator.py<br/>PPTX 生成]
    Gen --> CD
    
    CD --> Decision{📊 质量门控<br/>Score≥70?<br/>Critical=0?}
    Decision -->|Yes| Deliver[✅ 自动交付<br/>PPTX + 报告]
    Decision -->|No| AutoFix{🔍 可自动修复?}
    
    AutoFix -->|Yes| Fix[🔧 自动修复<br/>iter ≤2]
    AutoFix -->|No| Human[👤 人工审查<br/>生成预览]
    
    Fix --> Gen
    
    style CD fill:#FFD700,stroke:#FF8C00,stroke-width:3px
    style CP fill:#87CEEB,stroke:#4682B4,stroke-width:2px
    style VD fill:#98FB98,stroke:#32CD32,stroke-width:2px
    style Decision fill:#FFF9C4,stroke:#FBC02D,stroke-width:2px
    style Deliver fill:#C8E6C9,stroke:#4CAF50,stroke-width:3px
    style Human fill:#FFCDD2,stroke:#F44336,stroke-width:2px
    style Gen fill:#E0E0E0,stroke:#757575,stroke-width:1px
```

**Key Design Decisions**:
- ✅ **3 core roles** (vs 5 in original): content-planner, visual-designer, creative-director
- ✅ **Chart integration**: chart-specialist → visual-designer (real designers handle both layout and charts)
- ✅ **Distributed QA**: qa-reviewer → content-planner (content QA) + visual-designer (visual QA) + creative-director (final gate)
- ✅ **Creative coordination**: tech-lead → creative-director (aligned to creative director role, not tech lead)

**Industry Alignment**:
| Agent             | Real-World Role                | Representative Companies/People                      |
| ----------------- | ------------------------------ | ---------------------------------------------------- |
| -------           | ----------------               | --------------------------------                     |
| content-planner   | Content Strategist             | McKinsey Consultants, Barbara Minto                  |
| visual-designer   | Visual Designer + Chart Expert | Apple Keynote, Edward Tufte, Cole Nussbaumer Knaflic |
| creative-director | Creative Director              | IDEO, Pentagram, Duarte Design                       |

---

## Agent Responsibilities

### ppt-content-planner

**Role**: Content Strategist (aligned to McKinsey, Barbara Minto)

**Deliverables**: `.slides.md` with structured content and outline

**Input**: User request, design documents

**Output to**: `ppt-visual-designer`

**Core Responsibilities**:
- **Content Planning**: Slide count, story flow, bullet points
- **Structure Design**: Pyramid Principle, SCQA framework
- **Content Quality Self-Check**: Logic consistency, Key Decisions completeness, bullet compliance

**Key Decisions**:
- Slide count and story flow
- Which slides need visualizations
- Key Decisions placement (first 3-5 slides)
- Bullet points and text density (content level)

**Quality Ownership**: Content quality (40/100 points)

**Referenced Standards**:
- `agents/ppt-content-planner.agent.md`
- Barbara Minto - *The Pyramid Principle*
- McKinsey - MECE Framework

---

### ppt-visual-designer

**Role**: Visual Designer + Chart Specialist (aligned to Apple Keynote, Edward Tufte, Cole Nussbaumer Knaflic)

**Deliverables**: 
- `.slides.md` (with theme + layouts + chart configs)
- PNG chart files

**Input**: `ppt-content-planner` output

**Output to**: `ppt_generator.py`

**Core Responsibilities**:
- **Visual Design**: Theme, colors, typography, layouts
- **Chart Design**: Architecture, flowchart, comparison, timeline, matrix
- **Visual Quality Self-Check**: Contrast, aesthetic consistency, chart readability

**Key Decisions**:
- Design philosophy selection (Assertion-Evidence, Tufte, etc.)
- Color scheme (primary, secondary, accent)
- Typography (fonts, sizes)
- Chart type selection and visual encoding
- Layout templates (title-only, bullet-list, two-column, full-image)

**Quality Ownership**: Visual quality (40/100 points)

**Integrated Capabilities**:
- Original visual-designer capabilities
- Original chart-specialist chart design
- Original qa-reviewer visual quality checks

**Referenced Standards**:
- `agents/ppt-visual-designer.agent.md`
- `skills/ppt-visual.skill.md`
- `skills/ppt-chart.skill.md`
- Edward Tufte - *The Visual Display of Quantitative Information*
- Cole Nussbaumer Knaflic - *Storytelling with Data*
- Cleveland Perception Hierarchy

---

### ppt-creative-director

**Role**: Creative Director (aligned to IDEO, Pentagram, Duarte Design)

**Core Responsibilities**:
- **Process Orchestration**: Coordinate content-planner ↔ visual-designer
- **Requirements Understanding**: Identify presentation_type, audience, tone
- **Quality Gating**: Comprehensive scoring (content 40 + visual 40 + overall 20)
- **Decision Making**: Auto-deliver vs Auto-fix vs Human-review
- **Final Review**: Approval before delivery

**Input**: All agent outputs + generated PPTX

**Deliverables**: 
- Final PPTX file
- quality_report.json

**Decision Framework**:
- ✅ **Auto-deliver**: Score≥70 AND Critical=0
- 🔧 **Auto-fix**: Score<70 AND Critical=0 AND fixable AND iter<2
- 👤 **Human-review**: Critical>0 OR Score<50 OR iter>2

**Quality Ownership**: Overall quality (20/100 points) + final gate

**Authority**: Final decision maker (but doesn't micromanage)

**Referenced Standards**:
- `agents/ppt-creative-director.agent.md`
- Nancy Duarte - *Slide:ology*
- Guy Kawasaki - 10/20/30 Rule

---

## Workflow Steps

### 1. Content Planning

```yaml
agent: ppt-content-planner
input: [user_request, source_md, presentation_type]
output: slides.md (content outline)
self_check: content_quality
success_criteria:
  - Key Decisions identified
  - Slide count within limits
  - Visual needs marked
  - Logical structure (Pyramid Principle)
  - Bullets within limits
```

### 2. Visual Design

```yaml
agent: ppt-visual-designer
input: slides.md (from planner)
output: slides.md (with theme + charts) + *.png
depends_on: [1_planning]
self_check: visual_quality
success_criteria:
  - Theme applied (colors, fonts)
  - Layout templates defined
  - Design philosophy selected
  - All required diagrams configured
  - Visual style consistent
  - Color contrast ≥4.5:1
  - High-resolution output (200 DPI)
```

### 3. PPTX Generation

```yaml
tool: ppt_generator.py
input: slides.md
output: *.pptx
depends_on: [2_design]
```

### 4. Final Review

```yaml
agent: ppt-creative-director
input: [slides.md, *.pptx, all_agent_outputs]
output: quality_report.json
depends_on: [3_generation]
evaluation:
  - content_score: 40 points
  - visual_score: 40 points
  - overall_score: 20 points
success_criteria:
  - Final Score ≥ 70
  - Critical issues = 0
  - Key Decisions present in first 5 slides
```

### 5. Delivery Decision

```yaml
agent: ppt-creative-director
condition: final_score ≥ 70 AND critical_issues == 0
actions:
  auto_deliver: 
    trigger: passed AND key_decisions_present
    output: [*.pptx, quality_report.json]
  auto_fix:
    trigger: not_passed AND fixable AND iter < 2
    action: regenerate_from_step_3
  human_review:
    trigger: critical > 0 OR score < 50 OR iter > 2
    output: [preview.pptx, review_request.md]
```

---

## Iteration Limits

### Rule 1: Maximum Iterations = 2

Any feedback loop between two agents is limited to **2 iterations**.

| Interaction                                | Max Iterations  | Escalation                    |
| ------------------------------------------ | --------------- | ----------------------------- |
| -------------                              | --------------- | -----------                   |
| content-planner ↔ visual-designer          | 2               | creative-director arbitration |
| creative-director → auto-fix → re-evaluate | 2               | human-review required         |

### Rule 2: Iteration Counting

```text
Iteration 1: content-planner → visual-designer (initial submission)
Iteration 2: visual-designer → content-planner (feedback / change request)
Iteration 3: ❌ Exceeded - escalate to creative-director
```

### Rule 3: Iteration Tracking Template

Every feedback message MUST include the iteration count:

```markdown
## Feedback (Iteration 1/2)

**From**: @ppt-visual-designer
**To**: @ppt-content-planner
**Remaining Iterations**: 1

**Issue**: Slide 4 has 8 bullets, exceeds limit of 5 for technical-review

**Request**: Split Slide 4 into two slides or convert to visual diagram

**Reason**: presentation_type = technical-review requires max_bullets = 5
           Visual diagram more effective for complex comparisons

---
⚠️ Note: If not resolved, next iteration escalates to @ppt-creative-director
```

---

## Quality Gates

### Content Quality (Self-Check by ppt-content-planner)

- ✅ Key Decisions in first 3-5 slides
- ✅ Bullet points ≤ max_bullets (per presentation_type)
- ✅ Text density ≤ max_chars (per slide)
- ✅ Speaker notes coverage ≥ 80%
- ✅ Logical structure (SCQA, Pyramid)

### Visual Quality (Self-Check by ppt-visual-designer)

- ✅ Color contrast ratio ≥ 4.5:1 (WCAG AA)
- ✅ Visual coverage ≥ 30% (charts/images)
- ✅ Aesthetic consistency (same theme)
- ✅ Chart quality (Cleveland Hierarchy)
- ✅ Layout balance (white space)

### Overall Quality (Final Gate by ppt-creative-director)

**Blocking Conditions** (prevent delivery):
- ❌ Final Score < 70
- ❌ Critical issues > 0
- ❌ Key Decisions missing

**Warning Conditions** (deliver with notes):
- ⚠️ Major issues > 2
- ⚠️ Visual coverage < 30%

---

## Evaluation Formula

```python
# Content Quality (40分)
content_score = 40 * (
    0.3 * key_decisions_score +      # 关键决策 12分
    0.3 * bullets_compliance +        # bullets规范 12分
    0.2 * speaker_notes_coverage +    # speaker notes 8分
    0.2 * text_density_compliance     # 文本密度 8分
)

# Visual Quality (40分)
visual_score = 40 * (
    0.3 * color_contrast +            # 对比度 12分
    0.3 * visual_coverage +           # 可视化覆盖 12分
    0.2 * aesthetic_consistency +     # 美学一致性 8分
    0.2 * chart_quality               # 图表质量 8分
)

# Overall Quality (20分)
overall_score = 20 * (
    0.5 * slide_count_compliance +    # 页数 10分
    0.5 * design_philosophy_match     # 哲学符合 10分
)

final_score = content_score + visual_score + overall_score
passed = (final_score >= 70) and (critical_issues == 0)
```

---

## Success Metrics

Target KPIs for PPT generation workflow:

- ✅ End-to-end automation rate ≥80%
- ✅ Average generation time <60 seconds
- ✅ Quality score ≥70 in 90% of cases
- ✅ Human intervention rate <20%
- ✅ Agent iteration overruns <5%
- ✅ Content-planner self-check pass rate ≥95%
- ✅ Visual-designer self-check pass rate ≥88%

---

## Anti-patterns

### ❌ Anti-pattern 1: Iteration Overflow

```text
content-planner → visual-designer → content-planner → visual-designer → ...
```

**Problem**: No iteration limit leads to never-ending cycles

**Correct approach**: Escalate to creative-director after 2 iterations

### ❌ Anti-pattern 2: Skipping Self-Check

```text
content-planner → visual-designer (without content quality check)
```

**Problem**: Low-quality content flows downstream, wasting visual-designer time

**Correct approach**: Always run self-check before submitting to next agent

### ❌ Anti-pattern 3: Micromanagement by Creative Director

```text
creative-director: "Change bullet 3 on slide 5 to use different wording"
```

**Problem**: Violates "orchestrate, don't micromanage" principle

**Correct approach**: Provide feedback through proper agent (e.g., ask content-planner to revise)

### ❌ Anti-pattern 4: Ignoring Quality Gates

```text
Final score = 65, Critical = 1
Action: Auto-deliver ❌
```

**Problem**: Delivering low-quality PPT damages reputation

**Correct approach**: Trigger human-review when gates fail

---

## Degraded Output Strategies

When user input is incomplete, employ graceful degradation:

### Strategy 1: Assumptions with Placeholders

```markdown
## Degraded Output Declaration

**Reason**: User did not specify target audience

**Assumptions Made**:
- Audience: Technical team (default for design docs)
- Presentation type: technical-review
- Tone: Professional

**Placeholders**:
- Slide 1: [Company Logo] - replace with actual logo
- Slide 15: [Contact Info] - replace with actual contact

⚠️ **User**: Please confirm assumptions or provide corrections
```

### Strategy 2: Minimal Viable PPT

```markdown
## MVP Delivery

Due to incomplete input, delivering minimal viable version:

### Phase 1: Completed ✅
- Core content structure (10 slides)
- Key Decisions slide
- Basic theme applied

### Phase 2: Pending User Input ⏳
- Custom branding (logo, colors)
- Detailed speaker notes
- Custom charts (need data)

**Required from User**:
- [ ] Company branding guidelines
- [ ] Detailed data for charts
- [ ] Speaker notes content
```

---

## Escalation to Human Review

### Automatic Triggers

1. **Critical Issues**: Any critical issue detected
2. **Low Quality**: Final score < 50
3. **Iteration Overflow**: Iterations > 2
4. **Generation Failures**: Chart generation fails >50%
5. **User Request**: User explicitly requests review

### Review Request Format

```markdown
@human-reviewer – PPT review requested

## Reason for Escalation
- [X] Critical issues detected
- [ ] Quality score below threshold
- [ ] Iteration limit exceeded
- [ ] Generation failures

## Summary
- Project: online-ps-algorithm-v1.md → PPT
- Slides Generated: 15
- Quality Score: 48/100 ⚠️
- Critical Issues: 2

## Issues Detected

### Critical (2)
1. Slide 3: Key Decision missing (required in first 5 slides)
2. Slide 8: Color contrast ratio 2.1:1 (requires ≥4.5:1)

### Major (4)
1. Slide 4: 8 bullets (limit: 5)
2. Slide 6: Text density 450 chars (limit: 300)
3. ...

## Deliverables
- preview.pptx (attached)
- quality_report.json (attached)
- slides.md (source)

## Recommendation
Fix critical issues manually or adjust design requirements
```

---

## Version History

| Version  | Date       | Changes                                                                                                                                                                                                                                                                                |
| -------- | ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| -------- | ------     | ---------                                                                                                                                                                                                                                                                              |
| 1.0      | 2026-01-28 | Initial release (3-agent architecture)<br/>- Established content-planner, visual-designer, creative-director roles<br/>- Defined quality gates and evaluation formula<br/>- Set iteration limits (2) and escalation rules<br/>- Separated from general agent-collaboration-protocol.md |
