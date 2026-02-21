# Slide 3: Thinking

## 1. 核心任务与推理 (Mission & Reasoning)
- **目标**: Show the exponential growth of Large Language Models from 2024 to 2026, focusing on parameter count and reasoning scores.
- **视觉隐喻**: Upward trend, data-driven, analytical.
- **数据策略**: Use `ai-model-performance-2024-2025.csv`. Filter for GPT-4, GPT-4o, GPT-5, GPT-5.1, GPT-5.2, GPT-5.3. Plot Parameter Count (Trillions) and Reasoning Score.
- **布局权衡**:
  - *方案 A*: Full-width chart. -> Lacks space for insights.
  - *方案 B*: Data chart layout (chart on left, insights on right). -> **选定方案 B** because it balances data visualization with structured takeaways.

---

## 2. 执行规格 (Execution Specs)

### 2.1 布局锚点 (Layout Anchor)
- **Layout Key**: data_chart
- **Component**: line-chart, insight-cards

### 2.2 数据绑定 (Data Binding)
- **Source**: ai-model-performance-2024-2025.csv
- **Filter Logic**: Select GPT-4, GPT-4o, GPT-5, GPT-5.1, GPT-5.2, GPT-5.3.
- **Mapping**:
  - X-axis: Model Name
  - Y-axis 1 (Bar): Parameter Count (Trillions)
  - Y-axis 2 (Line): Reasoning Score

### 2.3 视觉细节 (Visual Props)
- **Style**: analytical, clean
- **Highlights**: Highlight GPT-5.3 as the latest milestone. Use Chart.js with dual Y-axes.

### 2.4 叙事文案 (Narrative)
- **Headline**: The Evolution of Large Language Models
- **Insight**: Model parameters and reasoning scores have grown exponentially, with GPT-5.3 reaching 4.2T parameters and a 96.7 benchmark score.
