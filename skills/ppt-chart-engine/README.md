# ppt-chart-engine

PPT HTML 幻灯片图表引擎 — 图表选择、渲染约束、数据契约的统一数据源。

## 解决的问题

| ID | 问题 | 解决方式 |
|----|------|----------|
| R1 | Bubble/Line/Heatmap 约束分散在多个 section | 按图表类型统一收归 `charts.yml → chart_constraints` |
| R3 | 容器高度、数据守卫等渲染规则与图表类型脱节 | 合并到 `rendering` 节，按子类分组 |
| R4 | 甘特图/时间线数据契约重复描述 | 统一到 `data_contracts` |
| R7 | 图表示例代码散落在 agent 尾部 | 收归 `examples` 节 |

## 文件结构

```
ppt-chart-engine/
├── manifest.yml    # 技能元信息
├── charts.yml      # 全部图表内容（7 节）
└── README.md       # 本文件
```

## charts.yml 结构

| 节 | 含义 | 条目数 |
|----|------|--------|
| `chart_types` | 基础+扩展图表类型（含 engine/CDN 映射） | 5 + 8 |
| `selection_algorithm` | 按维度/数据类型/洞察类型选图 | 3 组 |
| `semantic_mapping` | 语义强制映射（矩阵→热力图 / 路线→甘特 / …） | 3 条 |
| `data_contracts` | milestone-timeline + gantt 数据契约 | 2 |
| `chart_constraints` | Bubble+Narrative (10) / Bubble 坐标轴 (4) / Line/Trend (5) | 19 |
| `rendering` | 容器高度 (5) / 图元预算 (4) / Heatmap 填充率 (3) / 卡片内预算 (4) / 数据守卫 (5) | 21 |
| `examples` | Bubble + Heatmap 完整代码片段 | 2 |

## 使用方式

```yaml
# 在 agent 中通过 skill 引用：
# 选择图表 → charts.yml → chart_types + selection_algorithm
# 检查约束 → charts.yml → chart_constraints.{bubble_narrative|bubble_axis|line_trend}
# 渲染规则 → charts.yml → rendering.{container_height_contract|data_guards|...}
# 代码模板 → charts.yml → examples.{bubble_chart|heatmap_echarts}
```

## 与其他 skill 的关系

- **ppt-brand-system**：图表颜色从 `brands.yml → brands.{brand_id}.primary/secondary` 获取
- **ppt-visual-qa**：图表相关的门禁 gate（如 `bubble_clip_risk`、`chart_component_collision`）在 `gates.yml` 中定义
- **ppt-slide-layout-library**：布局模板中的 `chart-wrap` 容器遵循本 skill 的容器高度契约
