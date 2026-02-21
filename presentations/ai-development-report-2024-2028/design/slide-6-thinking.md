# Slide 6: Thinking

## 1. 核心任务与推理 (Mission & Reasoning)
- **目标**: 展示企业级 AI 应用的广泛落地及其带来的巨大投资回报率 (ROI) 和成本降低。
- **视觉隐喻**: “全景与成效”——使用仪表盘 (Dashboard) 布局，顶部展示核心 KPI，底部展示行业分布或具体案例的图表分析。
- **数据策略**: 从 `enterprise-ai-case-studies.csv` 中提取关键指标：
  - 最高 ROI: 1200% (Tesla) / 2500% (Tesla 2026) -> 取 2024-2025 的实际案例，如 Tesla 1200%, Amazon 890%, Microsoft 720%。
  - 平均成本降低: 提取 `cost_reduction_percentage` 的平均值或亮点数据。
  - 准确率提升: 提取 `accuracy_improvement` 的亮点数据。
- **布局权衡**:
  - *方案 A*: `data_chart` 布局。-> 只能展示单一维度的图表，无法体现 ROI、成本、准确率等多维度的商业价值。
  - *方案 B*: `dashboard_grid` 布局。-> **选定方案 B**，顶部放置 3-4 个核心 KPI 卡片，底部左侧放置一个柱状图展示不同企业的 ROI，右侧放置结构化洞察。

## 2. 执行规格 (Execution Specs)

### 2.1 布局锚点 (Layout Anchor)
- **Layout Key**: `dashboard_grid`
- **Component**: Top KPI row (4 cards) + Bottom row (Main Chart + Insight Cards)

### 2.2 数据绑定 (Data Binding)
- **Source**: `enterprise-ai-case-studies.csv`
- **Mapping**:
  - KPI 1: Max ROI (1200% - Tesla)
  - KPI 2: Avg Cost Reduction (~70%)
  - KPI 3: Avg Accuracy Improvement (~95%)
  - KPI 4: Time Savings (Up to 99% - Vodafone)
  - Main Chart: Bar chart showing ROI across top 5 companies (Tesla, Google, Amazon, Microsoft, Netflix).

### 2.3 视觉细节 (Visual Props)
- **Style**: `corporate` (使用 KPMG 品牌色，强调数据和商业价值)
- **Highlights**: 突出 ROI 和 Cost Reduction。

### 2.4 叙事文案 (Narrative)
- **Headline**: Enterprise AI Applications & ROI
- **Insight**: Enterprise AI adoption yields massive ROI (up to 1200%), significantly reducing costs and improving accuracy across industries.
