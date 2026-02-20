# ppt-slide-layout-library

> PPT HTML 幻灯片布局类型库 — 12 种布局模板 + 版式约束 + 选择指南 + 页面级预算的统一数据源

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Version](https://img.shields.io/badge/Version-2.0-blue.svg)](#)
[![Status](https://img.shields.io/badge/Status-Active-green.svg)](#)

## 解决的问题

| ID | 问题 | 解决方式 |
|----|------|----------|
| R1 | 版式约束分散在多个 section（cover/side-by-side/process/…各自独立） | 按布局类型统一收归 `layouts.yml → layouts.{type}.constraints` |
| R2 | 版面平衡/垂直预算/溢出策略与布局定义脱节 | 合并到 `page_constraints` 节 |
| R7 | HTML 模板代码散落在 agent 文件中 | 收归 `layouts.{type}.template` |

## 快速开始

```yaml
# 1. 选择布局
# 参考 layouts.yml → selection_guide

# 2. 获取 HTML 模板
# 参考 layouts.yml → layouts.{type}.template

# 3. 应用版式约束
# 参考 layouts.yml → layouts.{type}.constraints

# 4. 检查页面预算
# 参考 layouts.yml → layout_budget_profiles
```

## 文件结构

```
ppt-slide-layout-library/
├── manifest.yml    # 技能元信息
├── layouts.yml    # 全部布局内容（7 节）
└── README.md      # 本文件
```

## layouts.yml 结构

| 节 | 含义 | 条目数 |
|----|------|--------|
| `general` | 四段结构通用约束 | 1 |
| `layouts` | 12 种布局 + HTML 模板 + 各自的 constraints | 12+4 附属 |
| `selection_guide` | 内容类型 → 推荐布局映射表 | 11 |
| `dedup_rules` | 连续页禁止同构 + 例外 | 4 |
| `notion_skeleton` | 成片模式高保真页面骨架 HTML | 1 |
| `page_constraints` | 版面平衡(3) / 垂直预算(4) / 溢出处理(4) | 11 |
| `layout_budget_profiles` | 可执行布局预算档案（高度/间距/数量上限） | 8 |

## 布局类型速查

| # | 布局类型 | YAML Key | 适用场景 |
|---|---------|----------|----------|
| 1 | 封面 | `cover` | 首页、品牌展示 |
| 2 | 数据图表 | `data_chart` | 深度分析单个数据点 |
| 3 | 仪表盘网格 | `dashboard_grid` | 多维 KPI+趋势图组合，受控即兴排版 |
| 4 | 并排比较 | `side_by_side` | A/B 测试、竞品对比 |
| 5 | 全宽重点 | `full_width` | 战略愿景、趋势展示 |
| 6 | 混合 | `hybrid` | 多层次数据展示 |
| 7 | 流程 | `process` | 工作流、步骤说明 |
| 8 | 仪表板 | `dashboard` | 实时监控、KPI 跟踪（经典版） |
| 9 | 里程碑时间线 | `milestone_timeline` | 年度事件演进、阶段转折 |
| 10 | 支柱型 | `pillar` | Executive Summary，核心支柱，战略概览 |
| 11 | 流程型步骤 | `process_steps` | 时间线，演变过程，Step 1-2-3 逻辑 |
| 12 | 对比型 | `comparison` | 竞品分析，多维度对比 columns |

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
# 预算档案 → layouts.yml → layout_budget_profiles
# 去重检查 → layouts.yml → dedup_rules
# 页面预算 → layouts.yml → page_constraints
# 成片骨架 → layouts.yml → notion_skeleton
```

## 与其他 skill 的关系

### 依赖关系

- **ppt-brand-system**：布局模板中的 `brand-*` 类名和 CSS 变量从 `brands.yml` 获取
- **ppt-chart-engine**：布局模板中的 `chart-wrap` 容器遵循 `charts.yml → rendering.container_height_contract`

### 被依赖关系

- **ppt-visual-qa**：布局相关的门禁 gate（如 `cover_style_violation`、`side_by_side_height_mismatch`）在 `gates.yml` 中定义，验证布局约束是否满足

### 数据流

```
ppt-brand-system (品牌色/字体)
        ↓
ppt-slide-layout-library (布局模板 + 约束)
        ↓
ppt-chart-engine (图表容器 + 渲染规则)
        ↓
ppt-visual-qa (门禁验证)
```

## 约束示例

### cover 布局约束

```yaml
layouts:
  cover:
    constraints:
      - "首页默认必须使用 cover 布局；除非用户显式要求'目录型/摘要型首页'，不得使用 data-chart/hybrid/dashboard"
      - "正文抑制：封面不得出现大段分析正文（≤2条短句，总计≤80中文字符）"
      - "层级结构：必须包含 eyebrow + 主标题 + 副标题/英文标题 + 元信息 四层信息"
```

### side_by_side 布局约束

```yaml
layouts:
  side_by_side:
    constraints:
      equal_height:
        - "默认同高：两侧主图容器高度必须一致"
        - "主次图例外：仅当明确标注'主图/辅图'语义时允许不等高；高度差≤15%"
      whitespace_balance:
        - "上下留白差阈值：每张卡片内'图上留白'与'图下留白'差值≤24px"
        - "图区占比下限：图表容器高度占卡片可用高度 70%–82%"
```

## 页面预算示例

```yaml
layout_budget_profiles:
  data_chart:
    max_vertical_budget_outer_px: 582
    max_content_budget_inner_px: 502
    default_chart_height_px: 220
    min_chart_height_px: 180
    max_right_cards: 3
    max_list_items_per_card: 5
  
  side_by_side:
    max_vertical_budget_outer_px: 582
    max_content_budget_inner_px: 502
    per_col_chart_height_px: 210
    min_per_col_chart_height_px: 180
    max_bottom_kpi_rows: 3
```

## 许可证

MIT License - 详见项目根目录 [LICENSE](../../LICENSE)
