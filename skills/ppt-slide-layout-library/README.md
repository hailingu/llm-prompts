# ppt-slide-layout-library

PPT HTML 幻灯片布局类型库 — 8 种布局模板 + 版式约束 + 选择指南 + 页面级预算的统一数据源。

## 解决的问题

| ID | 问题 | 解决方式 |
|----|------|----------|
| R1 | 版式约束分散在多个 section（cover/side-by-side/process/…各自独立） | 按布局类型统一收归 `layouts.yml → layouts.{type}.constraints` |
| R2 | 版面平衡/垂直预算/溢出策略与布局定义脱节 | 合并到 `page_constraints` 节 |
| R7 | HTML 模板代码散落在 agent 文件中 | 收归 `layouts.{type}.template` |

## 文件结构

```
ppt-slide-layout-library/
├── manifest.yml    # 技能元信息
├── layouts.yml     # 全部布局内容（5 节）
└── README.md       # 本文件
```

## layouts.yml 结构

| 节 | 含义 | 条目数 |
|----|------|--------|
| `general` | 四段结构通用约束 | 1 |
| `layouts` | 8 种布局 + HTML 模板 + 各自的 constraints | 8+4 附属 |
| `selection_guide` | 内容类型 → 推荐布局映射表 | 7 |
| `dedup_rules` | 连续页禁止同构 + 例外 | 4 |
| `notion_skeleton` | 成片模式高保真页面骨架 HTML | 1 |
| `page_constraints` | 版面平衡(3) / 垂直预算(4) / 溢出处理(4) | 11 |

## 布局类型速查

| # | 布局类型 | YAML Key | 适用场景 |
|---|---------|----------|----------|
| 1 | 封面 | `cover` | 首页、品牌展示 |
| 2 | 数据图表 | `data_chart` | 深度分析单个数据点 |
| 3 | 并排比较 | `side_by_side` | A/B 测试、竞品对比 |
| 4 | 全宽重点 | `full_width` | 战略愿景、趋势展示 |
| 5 | 混合 | `hybrid` | 多层次数据展示 |
| 6 | 流程 | `process` | 工作流、步骤说明 |
| 7 | 仪表板 | `dashboard` | 实时监控、KPI 跟踪 |
| 8 | 里程碑时间线 | `milestone_timeline` | 年度事件演进、阶段转折 |

## 附属约束

| Key | 适用范围 | 条目数 |
|-----|---------|--------|
| `radar_kpi` | hybrid 且主图为雷达图 | 4 |
| `icon_consistency` | 跨布局通用 | 3 |
| `gantt` | Gantt 路线图页 | 5 |

## 使用方式

```yaml
# 在 agent 中通过 skill 引用：
# 选择布局 → layouts.yml → selection_guide
# HTML 模板 → layouts.yml → layouts.{type}.template
# 版式约束 → layouts.yml → layouts.{type}.constraints
# 去重检查 → layouts.yml → dedup_rules
# 页面预算 → layouts.yml → page_constraints
# 成片骨架 → layouts.yml → notion_skeleton
```

## 与其他 skill 的关系

- **ppt-brand-system**：布局模板中的 `brand-*` 类名和 CSS 变量从 `brands.yml` 获取
- **ppt-chart-engine**：布局模板中的 `chart-wrap` 容器遵循 `charts.yml → rendering.container_height_contract`
- **ppt-visual-qa**：布局相关的门禁 gate（如 `cover_style_violation`、`side_by_side_height_mismatch`）在 `gates.yml` 中定义
