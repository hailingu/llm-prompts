# Slide 5: Thinking

## 1. 核心任务与推理 (Mission & Reasoning)
- **目标**: 展示 AI Agent 技术和推理能力在 2026-2028 年的演进过程，从基础的数学推理到跨领域的 AGI 进展。
- **视觉隐喻**: “演进与突破”——使用流程图 (Process) 布局，展示从 2026 到 2028 年的阶段性里程碑。
- **数据策略**: 从 `ai-future-predictions-2026-2028.csv` 中提取 Agent 和 Reasoning 相关的预测：
  - 2026: Reasoning & Problem Solving (Mathematical reasoning at human expert level)
  - 2026: AI Agents (Autonomous task completion in complex environments)
  - 2027: AGI Progress (Cross-domain reasoning capabilities)
  - 2028: AGI Milestones (Human-level reasoning in narrow domains)
- **布局权衡**:
  - *方案 A*: `milestone_timeline` 布局。-> 适合展示时间线，但可能无法容纳足够多的结构化详情。
  - *方案 B*: `process` 布局。-> **选定方案 B**，能够清晰展示 4 个阶段的演进，并且下方有足够的空间容纳 Insight/Driver/Action 结构化信息，符合 G46 门控要求。

## 2. 执行规格 (Execution Specs)

### 2.1 布局锚点 (Layout Anchor)
- **Layout Key**: `process`
- **Component**: `process-flow` (4 steps) + `process-details` (4 cards)

### 2.2 数据绑定 (Data Binding)
- **Source**: `ai-future-predictions-2026-2028.csv`
- **Mapping**:
  - Step 1: 2026 - Expert Math Reasoning
  - Step 2: 2026 - Autonomous Agents
  - Step 3: 2027 - Cross-domain Reasoning
  - Step 4: 2028 - Narrow AGI Milestones

### 2.3 视觉细节 (Visual Props)
- **Style**: `professional` (使用 KPMG 品牌色，配合 FontAwesome 图标)
- **Highlights**: 强调 2028 年的 AGI 里程碑。

### 2.4 叙事文案 (Narrative)
- **Headline**: Agent Technology & Reasoning Capabilities
- **Insight**: AI agents are achieving autonomous task completion in complex environments, with mathematical reasoning approaching human expert levels.
