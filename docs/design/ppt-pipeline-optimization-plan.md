# PPT Pipeline 长效优化方案

> **策略代号**：增量增强，双轨并存  
> **版本**：v1.0  
> **日期**：2026-02-11  
> **状态**：Draft

---

## 目录

- [1. 背景与问题](#1-背景与问题)
- [2. 竞品差距分析](#2-竞品差距分析)
- [3. 根因诊断](#3-根因诊断)
- [4. 战略方向](#4-战略方向)
- [5. 架构设计](#5-架构设计)
- [6. 实施路线图](#6-实施路线图)
- [7. KPI 体系](#7-kpi-体系)
- [8. 风险与缓解](#8-风险与缓解)
- [附录 A：Schema v2 字段定义](#附录-a-schema-v2-字段定义)
- [附录 B：区域渲染器清单](#附录-b-区域渲染器清单)

---

## 1. 背景与问题

### 1.1 当前管线架构

PPT 生成管线由 **4 个 Agent + 4 个 Skill + 1 个 Schema + 1 个渲染器** 组成：

```
┌──────────────────────────────────────────────────────────────────┐
│  Creative Director (CD)                                          │
│  ┌──────────────────┐    ┌──────────────────┐                    │
│  │ Content Planner  │───▶│ Visual Designer  │                    │
│  │     (CP)         │    │     (VD)         │                    │
│  └────────┬─────────┘    └────────┬─────────┘                    │
│           │                       │                              │
│           ▼                       ▼                              │
│    slides.md              design_spec.json                       │
│    slides_semantic.json                                          │
│           │                       │                              │
│           └───────────┬───────────┘                              │
│                       ▼                                          │
│              ┌────────────────┐                                  │
│              │ PPT Specialist │                                  │
│              │     (PS)       │                                  │
│              └────────┬───────┘                                  │
│                       ▼                                          │
│              generate_pptx.py                                    │
│              (~3100 lines)                                       │
│                       ▼                                          │
│                   .pptx 文件                                     │
└──────────────────────────────────────────────────────────────────┘
```

**关键文件清单**：

| 文件 | 用途 | 行数 |
|------|------|------|
| `agents/ppt-creative-director.agent.md` | 编排协调 | 364 |
| `agents/ppt-content-planner.agent.md` | 内容规划，产出 slides.md + slides_semantic.json | 380 |
| `agents/ppt-visual-designer.agent.md` | 视觉设计，产出 design_spec.json | 656 |
| `agents/ppt-specialist.agent.md` | 渲染执行 | 561 |
| `skills/ppt-content-planning/README.md` | 内容规划技能规范 | 534 |
| `skills/ppt-design-system/README.md` | MD3 设计系统 | 648 |
| `skills/ppt-visual-taxonomy/README.md` | 可视化分类法（24 种类型） | 283 |
| `skills/ppt-generator/README.md` | 渲染架构文档 | 1464 |
| `knowledge/standards/common/slides-render-schema.json` | Schema v1 接口契约 | 185 |
| `skills/ppt-generator/bin/generate_pptx.py` | 单体渲染器 | ~3100 |

### 1.2 已完成的迭代

经过 v1 → v8 共 8 轮迭代，已实现：

- Material Design 3 配色体系（31 色彩 token）
- 12 栏网格布局（13.333" × 7.5"）
- 9 种风格预设（BCG、McKinsey、Bain 等）
- KPI 比例条（proportion bars）
- 数据稀疏页洞察条（insight strip）
- 空图表兜底（placeholder + 提示文字）
- 小列表垂直居中
- 对比页水平柱状图变体

### 1.3 核心问题

**同一份 `slides.md` 输入，竞品工具（如 Google NotebookLM）产出的 PPT 在视觉质量和信息密度上远超我们的输出**。

具体表现：

| 维度 | 竞品表现 | 我们的现状 |
|------|----------|------------|
| 页数 | 12 页（高密度） | 23 页（信息分散） |
| 标题 | 断言式标题（assertion title） | 标签式标题（label title） |
| 图表 | 原生矢量图 + 表格复合 | matplotlib 位图嵌入 |
| 布局 | 多组件复合排版 | 单元素/页 |
| 形状 | 原生 shapes 绘制架构图 | 文字占位符描述 |
| 图标 | 丰富的 icon 体系 | 无 |

---

## 2. 竞品差距分析

### 2.1 五大差距矩阵

| # | 差距维度 | 严重度 | 影响面 | 当前 | 目标 |
|---|----------|--------|--------|------|------|
| G1 | 断言式标题 vs 标签式标题 | 🔴 | 全部页面 | "储能技术路线对比" | "EV 快充在成本密度上领先 40%" |
| G2 | 多组件复合 vs 单元素/页 | 🔴 | 内容页 | 图/表/文字分页 | 图+表+要点同页 |
| G3 | 原生图表 vs matplotlib 位图 | 🔴 | 数据页 | PNG 嵌入 | python-pptx 原生图表 |
| G4 | shapes 渲染 vs 文字占位 | 🟡 | 架构页 | "请参考流程图" | AutoShape 箭头/框 |
| G5 | 信息密度（压缩比） | 🔴 | 全局 | 0.92（几乎 1:1） | ≤0.4 |

### 2.2 信息密度对比

```
竞品信息密度模型：
┌─────────────────────────────────────────────────┐
│  标题：断言（自带 insight）                       │
│ ┌──────────────────┬──────────────────────────┐  │
│ │  左区域：图表/表格 │  右区域：关键发现/要点    │  │
│ │  （原生矢量）     │  （结构化 bullets）       │  │
│ └──────────────────┴──────────────────────────┘  │
│  底部：数据来源 / callout                         │
└─────────────────────────────────────────────────┘

我们的信息密度模型：
┌─────────────────────────────────────────────────┐
│  标题：标签（无 insight）                         │
│                                                   │
│           大面积 matplotlib PNG                    │
│           或 3-5 条 bullets                       │
│                                                   │
│                                                   │
└─────────────────────────────────────────────────┘
```

---

## 3. 根因诊断

### 3.1 根因一：管线中缺少"视觉思考者"

**现象**：竞品的图表/图标更丰富、更切题。

**根因**：

- CP（Content Planner）负责内容但不思考视觉；VD（Visual Designer）负责样式但不理解内容。"什么内容用什么方式呈现"这个核心决策**无人负责**。
- CP 产出的 `visual` 字段仅是 `{"type": "bar_chart", "placeholder_data": {...}}`——一种机械映射，而非基于认知负荷的可视化选择。
- 24 种可视化类型定义在 `ppt-visual-taxonomy` 中，但无 Agent 具备"理解内容语义 → 选择最佳可视化 → 设计复合展示"的端到端能力。

**本质**：内容理解和视觉决策被人为割裂，管线中没有人在做**展示设计**（Exhibit Design）。

### 3.2 根因二：Schema 的类型枚举阻止了复合布局

**现象**：竞品一页放 图+表+要点，我们一页只放一种元素。

**根因**：

- Schema v1 的 `slide_type` 是一个枚举值（13 种），每种类型绑定一个固定布局函数。
- 渲染器通过 `RENDERERS[slide_type]` 分发——这意味着一页只能属于一种类型，无法组合。
- `components` 字段虽然支持多种组件共存，但实际渲染逻辑忽略了非当前 `slide_type` 关联的组件。

**对比**：竞品的渲染模型是**区域组合**（Region Composition）——页面被分为若干区域，每个区域独立渲染不同组件。我们的模型是**类型派发**（Type Dispatch）——`slide_type` 决定一切。

```
类型派发（当前）：               区域组合（目标）：
slide_type = "data-heavy"       layout_intent:
       │                          regions:
       ▼                          ┌──────────────────────┐
RENDERERS["data-heavy"]()        │ top: kpi_row          │
       │                          │ main: chart + table   │
       ▼                          │ side: callout_stack   │
  固定布局                         └──────────────────────┘
```

### 3.3 根因三：CP 是"抄写员"而非"编辑"

**现象**：竞品用更少的输入信息产出更丰富的内容。

**根因**：

- CP 的 MO-6 自检规则（"数据必须有来源，不捏造"）被**过度解读**为"不做任何内容合成"。
- CP 接收到丰富的源文档（`slides.md`），但仅做 1:1 段落搬运——每个 heading 变一页 slide，每个 bullet 原封不动。这导致压缩比接近 1.0。
- 竞品的做法是**内容编辑**：合并相似段落、提炼断言、补充推论、交叉引用数据——最终 25 节内容浓缩为 12 页高密度展示。

**类比**：

| 角色 | CP（当前） | 竞品行为 |
|------|-----------|----------|
| 定位 | 抄写员 / Transcriber | 编辑 / Editor |
| 输入→输出 | 25 节 → 23 页 | 25 节 → 12 页 |
| 核心动作 | 搬运、格式化 | 合并、提炼、推论 |
| 压缩比 | ~0.92 | ~0.4 |

### 3.4 一句话总结

> **差距的根源不是"我们的工具少"，而是"我们的管线里没有人在思考"。**
>
> 竞品赢在**内容编辑能力**（合并提炼）和**展示设计能力**（区域复合），而不仅是渲染技术。

---

## 4. 战略方向

### 4.1 核心策略：增量增强，双轨并存

```
"不替换，只增强；不改 CP，加 EA；v1 自动降级，v2 渐进增能。"
```

#### 关键约束

1. **CP 必须保持不变**——`slides.md` 是可跨平台资产（可喂给竞品工具验证效果），改掉 CP 等于失去这个能力。
2. **渲染器必须向后兼容**——v1 格式的 `slides_semantic.json` 始终可正常渲染。
3. **渐进增强**——Schema v2 新字段为可选扩展，不破坏现有流程。

#### 双轨模型

```
          slides.md
              │
              ▼
    ┌──────────────────┐
    │  Content Planner │ ← 保持不变
    │      (CP)        │
    └────────┬─────────┘
             │
             ▼
    slides_semantic.json (v1)
             │
     ┌───────┴───────┐
     │               │
     ▼               ▼
  Track A          Track B
  (v1 直通)       (v2 增强)
     │               │
     │      ┌────────┴─────────┐
     │      │ Exhibit Architect│ ← 新增 Agent
     │      │      (EA)        │
     │      └────────┬─────────┘
     │               │
     │               ▼
     │      slides_semantic.json (v2)
     │      ┌──────────────────────┐
     │      │ + assertion 断言标题  │
     │      │ + insight  洞察提炼   │
     │      │ + layout_intent 区域  │
     │      │ + 页面合并/压缩       │
     │      └──────────────────────┘
     │               │
     └───────┬───────┘
             │
             ▼
    ┌────────────────┐
    │   Renderer     │ ← 自动检测 v1/v2
    │ generate_pptx  │
    └────────┬───────┘
             │
             ▼
         .pptx 文件
```

### 4.2 EA（Exhibit Architect）Agent 定位

| 维度 | 说明 |
|------|------|
| 输入 | v1 `slides_semantic.json` + `slides.md`（原始素材） |
| 输出 | v2 `slides_semantic.json`（增强版） |
| 核心能力 | ① 断言提炼 ② 页面合并 ③ 可视化升级 ④ 区域布局设计 |
| 与 CP 关系 | CP 产出 v1 → EA 增强为 v2（可选环节，跳过则用 v1 直通） |
| 触发方式 | CD 编排时决定是否启用 EA（默认启用） |

**EA 做什么 vs 不做什么**：

| EA 做 | EA 不做 |
|-------|---------|
| 从 bullets 中提炼 assertion title | 修改 slides.md |
| 合并相似 slide（25→15 页） | 创造不存在的数据 |
| 将 slide_type + components 转为 regions 布局 | 替代 CP 的内容规划 |
| 标注 insight（每页的核心发现） | 替代 VD 的配色/排版 |
| 升级 visual 类型（bar_chart → grouped_bar + table） | 渲染 PPTX |

---

## 5. 架构设计

### 5.1 Schema v2 设计（向后兼容）

Schema v2 在 v1 基础上**新增可选字段**，不修改任何 v1 字段：

```jsonc
// slide 层级新增字段
{
  "slide_id": 1,
  "title": "储能技术路线对比",              // ← v1 保留
  
  // ===== v2 新增字段（均可选） =====
  "assertion": "EV 快充在成本密度上领先 40%", // 断言式标题
  "insight": "锂电池技术在 2025 年成本下降...", // 页面核心洞察
  "layout_intent": {
    "template": "two-region-split",         // 布局模板名
    "regions": [
      {
        "id": "main",
        "position": "left-60",
        "renderer": "comparison_table",
        "data_source": "components.comparison_items"
      },
      {
        "id": "side",
        "position": "right-40",
        "renderer": "callout_stack",
        "data_source": "components.callouts"
      }
    ]
  },
  
  "slide_type": "comparison",              // ← v1 保留（v2 降级时使用）
  "content": [...],                         // ← v1 保留
  "components": {...}                       // ← v1 保留
}
```

**版本检测逻辑**：

```python
def detect_schema_version(slide: dict) -> int:
    """渲染器自动检测 slide 的 schema 版本"""
    if "layout_intent" in slide and "regions" in slide.get("layout_intent", {}):
        return 2
    return 1

def render_slide(slide: dict):
    version = detect_schema_version(slide)
    if version == 2:
        render_slide_v2(slide)    # 区域组合渲染
    else:
        render_slide_v1(slide)    # 原有 RENDERERS[slide_type] 分发
```

### 5.2 渲染器架构升级

#### 5.2.1 原生图表替代 matplotlib

**最高 ROI 单项改进**。将 matplotlib 位图替换为 `python-pptx` 原生图表 API：

```python
# 当前（matplotlib → PNG → 嵌入）：
def generate_chart_image(visual, width, height):
    fig, ax = plt.subplots()
    ax.bar(labels, values)
    buf = BytesIO()
    fig.savefig(buf, format='png')
    return buf

# 目标（python-pptx 原生图表）：
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE

def render_native_chart(slide, visual, left, top, width, height):
    chart_data = CategoryChartData()
    chart_data.categories = visual["placeholder_data"]["labels"]
    chart_data.add_series("Series 1", visual["placeholder_data"]["values"])
    chart = slide.shapes.add_chart(
        XL_CHART_TYPE.COLUMN_CLUSTERED,
        Inches(left), Inches(top), Inches(width), Inches(height),
        chart_data
    ).chart
    # 应用 MD3 配色
    apply_chart_theme(chart, section_accent)
```

**支持的原生图表类型**：

| python-pptx 枚举 | 对应页面场景 |
|-------------------|-------------|
| `COLUMN_CLUSTERED` | 技术对比、性能数据 |
| `BAR_CLUSTERED` | 水平对比 |
| `LINE` | 趋势、时间序列 |
| `PIE` | 占比、份额 |
| `DOUGHNUT` | 占比（变体） |
| `RADAR` | 多维评估 |
| `XY_SCATTER` | 散点分布 |

#### 5.2.2 区域组合渲染引擎

替代 `RENDERERS[slide_type]` 类型派发，实现基于 `layout_intent.regions[]` 的组合渲染：

```python
# 区域渲染器注册表
REGION_RENDERERS = {
    "chart":            render_region_chart,       # 原生图表
    "comparison_table": render_region_comparison,   # 对比表格
    "kpi_row":          render_region_kpi,          # KPI 卡片行
    "callout_stack":    render_region_callout,      # Callout 叠加
    "progression":      render_region_progression,  # 渐进/时间线
    "architecture":     render_region_architecture, # 架构图（shapes）
    "flow":             render_region_flow,         # 流程图（shapes）
    "bullet_list":      render_region_bullets,      # 结构化要点
}

def render_slide_v2(slide: dict):
    """v2 区域组合渲染"""
    # 1. 渲染断言标题
    if slide.get("assertion"):
        render_assertion_title(slide)
    else:
        render_label_title(slide)
    
    # 2. 渲染洞察条
    if slide.get("insight"):
        render_insight_bar(slide)
    
    # 3. 按区域逐个渲染
    for region in slide["layout_intent"]["regions"]:
        renderer = REGION_RENDERERS[region["renderer"]]
        bounds = compute_region_bounds(region["position"])
        data = resolve_data_source(slide, region["data_source"])
        renderer(pptx_slide, data, bounds)
```

**布局模板预设**：

| 模板名 | 区域配置 | 典型用途 |
|--------|----------|----------|
| `full-width` | main: 100% | 大图表、架构图 |
| `two-region-split` | left: 60%, right: 40% | 图表+要点 |
| `two-region-equal` | left: 50%, right: 50% | 双列对比 |
| `three-region-top` | top: 30%, left: 35%, right: 35% | KPI 行 + 双图 |
| `grid-2x2` | 4 等分 | 多维对比矩阵 |
| `t-shape` | top: 25%, bottom-left: 50%, bottom-right: 50% | 标题+双区域 |

#### 5.2.3 Shapes 渲染引擎

将 Mermaid 语法/文字描述转换为 `python-pptx` 原生 AutoShape：

```python
def render_region_architecture(pptx_slide, data, bounds):
    """渲染架构图：框 + 箭头 + 文字标注"""
    for node in data["nodes"]:
        shape = pptx_slide.shapes.add_shape(
            MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE,
            Inches(node["x"]), Inches(node["y"]),
            Inches(node["w"]), Inches(node["h"])
        )
        shape.text = node["label"]
        apply_shape_style(shape, node.get("style", "primary"))
    
    for edge in data["edges"]:
        connector = pptx_slide.shapes.add_connector(
            MSO_CONNECTOR_TYPE.STRAIGHT,
            Inches(edge["x1"]), Inches(edge["y1"]),
            Inches(edge["x2"]), Inches(edge["y2"])
        )
        add_arrowhead(connector)
```

### 5.3 Agent 调整方案

#### 5.3.1 Content Planner (CP) — 不修改

CP 保持原样，继续产出：

- `slides.md`：跨平台可读的演示大纲（可喂给任何工具）
- `slides_semantic.json` (v1)：结构化中间表示

**理由**：

- `slides.md` 是宝贵的跨平台资产
- CP 已有稳定的 17 条自检规则
- 内容增强由新增的 EA Agent 承担

#### 5.3.2 Exhibit Architect (EA) — 新增

```yaml
# EA Agent 核心职责
name: exhibit-architect
type: 新增 Agent（可选增强层）
input: slides_semantic.json (v1) + slides.md
output: slides_semantic.json (v2)

capabilities:
  - assertion_extraction:
      description: 从 bullets 中提炼 assertion title
      method: "找到每页的'所以呢？'(So What?) → 一句话断言"
      example: 
        before: "储能技术路线对比"
        after: "EV 快充在成本密度上领先 40%，但循环寿命仍制约规模化"
  
  - page_merging:
      description: 合并信息密度不足的相邻页
      rules:
        - "同 section 内的纯 bullet 页 → 合并"
        - "KPI 页 + 说明页 → 合并为 KPI+callout 复合页"
        - "对比页 + 数据页 → 合并为 chart+table 复合页"
      target: "23 页 → 12-15 页"
  
  - visual_upgrade:
      description: 升级可视化类型选择
      rules:
        - "多系列 bar_chart → grouped_bar + inline_table"
        - "纯 bullets ≥ 5 条 → icon_grid 或 process_flow"
        - "timeline + milestone → gantt_simplified"
  
  - layout_design:
      description: 为每页设计 layout_intent (区域布局)
      method: "根据组件数量和类型选择最佳布局模板"
```

#### 5.3.3 Creative Director (CD) — 微调编排

CD 新增 EA 调度逻辑：

```
原有流程：CD → CP → VD → PS
新增流程：CD → CP → EA(可选) → VD → PS
```

CD 决定是否启用 EA 的判断条件：

- 默认启用 EA
- 用户显式要求"快速/简单版"时跳过 EA
- slides.md 不足 10 页时跳过 EA（信息量不足以压缩）

#### 5.3.4 PPT Specialist (PS) — 适配 v2

PS 需要感知 Schema 版本：

```
v1 路径：直接调用 generate_pptx.py（现有逻辑）
v2 路径：调用 generate_pptx.py（自动检测并使用区域渲染）
```

### 5.4 Skill 调整方案

| Skill | 调整内容 |
|-------|----------|
| `ppt-content-planning` | 不修改 |
| `ppt-design-system` | 新增区域布局模板定义 |
| `ppt-visual-taxonomy` | 新增复合可视化类型（grouped_bar+table 等） |
| `ppt-generator` | 新增原生图表 API 文档、区域渲染器接口、shapes 绘制 API |

---

## 6. 实施路线图

### 6.1 全局视图

```
Week  1   2   3   4   5   6   7   8   9  10  11  12  13  14
     ├───────┤                                              P0: 原生图表
             ├───────┤                                      P1: 断言 + 洞察
                     ├───────────────┤                      P2: EA Agent
                                     ├───────────────┤     P3: 区域引擎
                                                     ├─────P4: Shapes 渲染
                                                       ├───P5: 反馈闭环
```

### 6.2 P0：原生图表替代（第 1-2 周）

**目标**：将 matplotlib → PNG → embed 替换为 `python-pptx` 原生图表。

**改动范围**：

| 文件 | 改动 |
|------|------|
| `generate_pptx.py` | 新增 `render_native_chart()`；改造 `render_visual()` 优先走原生路径 |
| `ppt-generator/README.md` | 更新渲染文档，标注原生 vs 位图路径 |

**实施步骤**：

1. 在 `generate_pptx.py` 中新增 `render_native_chart()` 函数，支持 7 种图表类型
2. 新增 `apply_chart_theme()` 将 MD3 配色应用于原生图表
3. 修改 `render_visual()` 的分发逻辑：优先尝试原生，不支持的类型 fallback matplotlib
4. 生成对比 PPTX，验证原生图表视觉效果
5. 更新 `ppt-generator/README.md` 文档

**验收标准**：

- 所有数据页使用原生图表（非 PNG）
- 图表可在 PowerPoint 中双击编辑
- 配色与 MD3 主题一致

**风险**：`python-pptx` 的图表 API 不支持全部 matplotlib 图表类型（如热力图、桑基图）→ 保留 matplotlib 作为 fallback。

### 6.3 P1：断言标题 + 洞察提炼（第 3-4 周）

**目标**：引入 `assertion` 和 `insight` 字段及其渲染。

**改动范围**：

| 文件 | 改动 |
|------|------|
| `slides-render-schema.json` | 新增 `assertion`、`insight` 可选字段 |
| `generate_pptx.py` | 新增 `render_assertion_title()`、`render_insight_bar()` |
| `ppt-content-planning/README.md` | 新增断言提取指南（供 EA 参考） |

**实施步骤**：

1. 在 Schema 中添加 `assertion` (string, optional) 和 `insight` (string, optional) 字段
2. 实现 `render_assertion_title()`：16pt 粗体断言 + 10pt 浅色原标题
3. 实现 `render_insight_bar()`：底部深色条，10pt 白字洞察
4. 修改渲染入口：检测 `assertion` 字段 → 使用断言标题渲染

**版本兼容**：

- 无 `assertion` 字段 → 正常使用 `title` 渲染（v1 行为）
- 有 `assertion` 字段 → 使用断言标题渲染（v2 行为）

### 6.4 P2：Exhibit Architect Agent（第 5-7 周）

**目标**：创建 EA Agent，实现 v1→v2 的内容增强。

**交付物**：

| 文件 | 说明 |
|------|------|
| `agents/ppt-exhibit-architect.agent.md` | EA Agent 定义 |
| `skills/ppt-exhibit-design/README.md` | 展示设计技能规范 |

**EA 处理流程**：

```
输入：slides_semantic.json (v1, 23 slides)
  │
  ├─ Step 1: Assertion Extraction
  │    对每页 bullets 提问 "So What?"
  │    → 提炼 assertion 字段
  │
  ├─ Step 2: Page Merging
  │    识别可合并的相邻页
  │    → 23 → 12-15 页
  │
  ├─ Step 3: Insight Annotation
  │    为每页标注核心发现
  │    → 填充 insight 字段
  │
  ├─ Step 4: Visual Upgrade
  │    升级可视化类型选择
  │    → 更新 visual.type + 增加复合组合
  │
  └─ Step 5: Layout Design
       为每页分配区域布局
       → 填充 layout_intent
  
输出：slides_semantic.json (v2, 12-15 slides)
```

**EA 自检规则（草案）**：

| 编号 | 规则 | 强度 |
|------|------|------|
| EA-0 | 不修改 slides.md（只读参考） | 强制 |
| EA-1 | 不创造不存在于原文的数据/事实 | 强制 |
| EA-2 | assertion 必须可从 bullets 中推导 | 强制 |
| EA-3 | 合并后页数 ≤ 原页数 × 0.65 | 建议 |
| EA-4 | 每页至少 2 个区域 | 建议 |
| EA-5 | assertion ≠ 原 title 的简单复述 | 强制 |

### 6.5 P3：区域组合渲染引擎（第 8-10 周）

**目标**：实现基于 `layout_intent.regions[]` 的复合渲染。

**改动范围**：

| 文件 | 改动 |
|------|------|
| `slides-render-schema.json` | 新增 `layout_intent`、`regions` 定义 |
| `generate_pptx.py` | 新增 `render_slide_v2()`、8 个区域渲染器、布局模板系统 |
| `ppt-design-system/README.md` | 新增 6 种布局模板规范 |

**关键实现**：

1. **布局模板解析器**：将 `"left-60"` 等位置标记转换为像素坐标
2. **8 个区域渲染器**：见 [附录 B](#附录-b-区域渲染器清单)
3. **数据源解析**：`"components.comparison_items"` → 动态取值
4. **回退策略**：无 `layout_intent` → 走 v1 `RENDERERS[slide_type]`

### 6.6 P4：Shapes 渲染（第 11-12 周）

**目标**：将架构图、流程图从文字描述转为原生 AutoShape。

**改动范围**：

| 文件 | 改动 |
|------|------|
| `generate_pptx.py` | 新增 `render_region_architecture()`、`render_region_flow()` |
| `ppt-visual-taxonomy/README.md` | 新增 shapes 类可视化类型定义 |
| Schema | `components` 新增 `architecture_data`、`flow_data` 定义 |

**支持的图形元素**：

- 圆角矩形（节点/模块）
- 直线/折线连接器（带箭头）
- 2×2 矩阵（四象限图）
- 漏斗图（MSO_AUTO_SHAPE_TYPE.FUNNEL 等）

### 6.7 P5：反馈闭环（第 13-14 周）

**目标**：建立 PPT 质量度量和持续优化机制。

**实现方案**：

```python
# 度量记录格式 (metrics.jsonl)
{
    "timestamp": "2026-03-15T10:30:00Z",
    "deck_id": "storage-frontier-20260211",
    "schema_version": 2,
    "total_slides": 14,
    "metrics": {
        "assertion_title_rate": 0.85,    # 断言标题覆盖率
        "native_visual_rate": 0.80,      # 原生可视化覆盖率
        "compression_ratio": 0.35,       # 页面压缩比
        "placeholder_rate": 0.05,        # 占位符残留率
        "multi_region_rate": 0.70,       # 多区域页占比
        "avg_components_per_slide": 2.3  # 平均组件数/页
    }
}
```

**审计规则**：

| 指标 | 黄线 | 红线 | 触发动作 |
|------|------|------|----------|
| assertion_title_rate | < 70% | < 50% | 提示 EA 增强提炼 |
| native_visual_rate | < 60% | < 40% | 检查图表类型覆盖 |
| compression_ratio | > 0.5 | > 0.7 | 提示 EA 加强合并 |
| placeholder_rate | > 10% | > 20% | 提示增加渲染器覆盖 |

---

## 7. KPI 体系

### 7.1 核心指标

| 指标 | 定义 | 当前值 | P0 后 | P2 后 | P5 后（目标） |
|------|------|--------|-------|-------|--------------|
| **assertion_title_rate** | 使用断言标题的页面占比 | 0% | 0% | 70% | ≥85% |
| **native_visual_rate** | 使用原生图表/shapes 的可视化占比 | 0% | 60% | 70% | ≥80% |
| **compression_ratio** | 输出页数/输入段落数 | 0.92 | 0.92 | 0.45 | ≤0.40 |
| **placeholder_rate** | 仍使用文字占位符的可视化占比 | 40% | 20% | 10% | ≤5% |
| **multi_region_rate** | 使用多区域复合布局的页面占比 | 0% | 0% | 50% | ≥65% |
| **avg_components_per_slide** | 每页平均组件数 | 1.2 | 1.2 | 2.0 | ≥2.3 |

### 7.2 质量里程碑

| 里程碑 | 达成条件 | 预期时间 |
|--------|----------|----------|
| **M1: 图表平权** | native_visual_rate ≥ 60% | P0 完成 |
| **M2: 标题升级** | assertion_title_rate ≥ 70% | P2 完成 |
| **M3: 密度追平** | compression_ratio ≤ 0.45 | P2 完成 |
| **M4: 布局追平** | multi_region_rate ≥ 50% | P3 完成 |
| **M5: 全面达标** | 所有 KPI 达标 | P5 完成 |

---

## 8. 风险与缓解

| 风险 | 概率 | 影响 | 缓解策略 |
|------|------|------|----------|
| `python-pptx` 图表 API 不支持某些图表类型 | 中 | 中 | 保留 matplotlib fallback；逐步覆盖 |
| EA 断言提取质量不稳定 | 中 | 高 | EA-5 自检规则 + CD 审核环节 |
| EA 过度合并导致信息丢失 | 低 | 高 | EA-1 强制不丢数据 + 用户可选跳过 EA |
| 区域布局在不同分辨率下错位 | 中 | 中 | 基于 12 栏网格的相对定位 |
| v1/v2 渲染路径维护成本 | 低 | 低 | v1 代码冻结，只修 bug；新功能只加 v2 |
| LLM 上下文长度限制影响 EA 处理长文档 | 中 | 中 | EA 按 section 分批处理 |

---

## 附录 A：Schema v2 字段定义

### slide 级新增字段

```jsonc
{
  // v2 新增字段，全部 optional
  "assertion": {
    "type": "string",
    "description": "断言式标题 —— 一句话概括本页核心发现/结论。渲染为 16pt 粗体主标题，原 title 降为 10pt 副标题。"
  },
  "insight": {
    "type": "string",
    "description": "页面洞察 —— 补充说明断言的依据或启示。渲染为底部深色条带白字。"
  },
  "layout_intent": {
    "type": "object",
    "description": "区域布局意图 —— 指导渲染器如何组合区域。",
    "properties": {
      "template": {
        "type": "string",
        "enum": ["full-width", "two-region-split", "two-region-equal", 
                 "three-region-top", "grid-2x2", "t-shape"],
        "description": "布局模板名"
      },
      "regions": {
        "type": "array",
        "items": {
          "type": "object",
          "required": ["id", "position", "renderer", "data_source"],
          "properties": {
            "id": { "type": "string", "description": "区域标识" },
            "position": { 
              "type": "string", 
              "description": "位置描述，如 'left-60', 'right-40', 'top-30', 'full'"
            },
            "renderer": { 
              "type": "string",
              "enum": ["chart", "comparison_table", "kpi_row", "callout_stack",
                       "progression", "architecture", "flow", "bullet_list"],
              "description": "区域渲染器类型"
            },
            "data_source": {
              "type": "string",
              "description": "数据来源路径，如 'components.kpis', 'visual'"
            }
          }
        }
      }
    }
  }
}
```

### components 新增类型

```jsonc
{
  // 新增到 definitions.components.properties
  "architecture_data": {
    "type": "object",
    "description": "架构图数据 —— 节点 + 连线",
    "properties": {
      "nodes": {
        "type": "array",
        "items": {
          "type": "object",
          "required": ["id", "label"],
          "properties": {
            "id": { "type": "string" },
            "label": { "type": "string" },
            "x": { "type": "number" },
            "y": { "type": "number" },
            "w": { "type": "number" },
            "h": { "type": "number" },
            "style": { "type": "string", "enum": ["primary", "secondary", "tertiary", "outline"] }
          }
        }
      },
      "edges": {
        "type": "array",
        "items": {
          "type": "object",
          "required": ["from", "to"],
          "properties": {
            "from": { "type": "string" },
            "to": { "type": "string" },
            "label": { "type": "string" },
            "style": { "type": "string", "enum": ["solid", "dashed"] }
          }
        }
      }
    }
  },
  "flow_data": {
    "type": "object",
    "description": "流程图数据 —— 步骤 + 箭头",
    "properties": {
      "steps": {
        "type": "array",
        "items": {
          "type": "object",
          "required": ["id", "label"],
          "properties": {
            "id": { "type": "string" },
            "label": { "type": "string" },
            "type": { "type": "string", "enum": ["start", "process", "decision", "end"] }
          }
        }
      },
      "transitions": {
        "type": "array",
        "items": {
          "type": "object",
          "required": ["from", "to"],
          "properties": {
            "from": { "type": "string" },
            "to": { "type": "string" },
            "condition": { "type": "string" }
          }
        }
      }
    }
  }
}
```

---

## 附录 B：区域渲染器清单

| 渲染器 | 输入数据 | 渲染效果 | 引入阶段 |
|--------|----------|----------|----------|
| `chart` | `visual.placeholder_data` | python-pptx 原生图表 | P0 |
| `comparison_table` | `components.comparison_items` | MD3 风格对比表格 | P3 |
| `kpi_row` | `components.kpis` | 横排 KPI 卡片（含比例条） | P3 |
| `callout_stack` | `components.callouts` | 纵向叠加引用条 | P3 |
| `progression` | `components.timeline_items` | 时间线/里程碑点 | P3 |
| `architecture` | `components.architecture_data` | AutoShape 框+箭头架构图 | P4 |
| `flow` | `components.flow_data` | AutoShape 流程图 | P4 |
| `bullet_list` | `content[]` | 结构化要点列表 | P3 |

---

*本文档是 PPT Pipeline 长效优化的战略规划，后续将根据实施进展持续更新。*
