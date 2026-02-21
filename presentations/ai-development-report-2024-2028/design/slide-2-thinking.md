# Slide 2: Thinking

## 1. 核心任务与推理 (Mission & Reasoning)
- **目标**: Present the executive summary of the AI Development Report, highlighting key breakthroughs and future predictions.
- **视觉隐喻**: High-level overview, structured and easy to read. Using a full-width layout with floating cards for key takeaways.
- **数据策略**: Extract key points from `ai-report-executive-summary.md`.
- **布局权衡**:
  - *方案 A*: Standard bullet points. -> Too boring and lacks visual hierarchy.
  - *方案 B*: Full-width layout with floating cards for key findings. -> **选定方案 B** because it provides a modern, consulting-style executive summary.

---

## 2. 执行规格 (Execution Specs)

### 2.1 布局锚点 (Layout Anchor)
- **Layout Key**: full_width
- **Component**: floating-cards, text-blocks

### 2.2 数据绑定 (Data Binding)
- **Source**: ai-report-executive-summary.md
- **Filter Logic**: Extract top 3 breakthroughs and top 3 predictions.
- **Mapping**: None

### 2.3 视觉细节 (Visual Props)
- **Style**: professional, structured
- **Highlights**: Use `.card-float` for key takeaways. Use semantic colors (emerald for achievements, sky for predictions).

### 2.4 叙事文案 (Narrative)
- **Headline**: Executive Summary
- **Insight**: AI technology has experienced breakthrough progress in LLMs, multimodal AI, and agent technologies, with significant cost reductions and performance improvements.
