# PPT Generator Skill

> 从 `slides_semantic.json` + `design_spec.json` 生成 PPTX 的完整技能规范

---

> ## ⛔ IMPLEMENTATION NOTE — 请先阅读
>
> **本 skill 的渲染能力已完整实现在 `skills/ppt-generator/bin/generate_pptx.py`（~1477 行，自包含）。**
>
> **唯一正确的执行命令**：
> ```bash
> python3 skills/ppt-generator/bin/generate_pptx.py \
>   --semantic <slides_semantic.json> \
>   --design <design_spec.json> \
>   --output <output.pptx>
> ```
>
> **以下命令已废弃，绝对不要使用**：
> - ❌ `python -m skills.ppt_generator.generate ...` （模块不存在，会报 ModuleNotFoundError）
> - ❌ `from skills.ppt_generator import generate_pptx` （phantom import，不存在）
> - ❌ `python3 scripts/generate_pptx_ci.py ...` （168 行，功能不完整）
>
> 本文档的代码片段仅用于**解释设计原理和逻辑结构**，不可直接执行。
> 如需修改渲染逻辑，直接编辑 `skills/ppt-generator/bin/generate_pptx.py`。

---

## 1. 概述

本技能将结构化的语义 JSON（`slides_semantic.json`）与设计规范（`design_spec.json`）转化为可交付的 PPTX 文件。所有内容、布局、样式均由输入文件驱动，不做主观设计决策。

### 输入文件

| 文件 | 来源 | 用途 |
|------|------|------|
| `slides_semantic.json` | ppt-content-planner | 全部幻灯片内容、结构、视觉类型、占位数据、演讲者笔记 |
| `design_spec.json` | ppt-visual-designer | 色彩、字体、间距、网格、组件库、无障碍规范 |

### 输出

| 产出物 | 格式 | 说明 |
|--------|------|------|
| `<project>.pptx` | PPTX | 16:9 宽屏 (1920×1080 / 13.33"×7.5") |
| `qa_report.json` | JSON | 6 阶段 QA 结果 |
| `previews/` | PNG | 每页预览图 |

### 技术栈

```
python-pptx >= 0.6.23
fonttools    (字体子集，可选)
pngquant     (图片压缩，可选)
```

---

## 2. 输入解析

### 2.1 slides_semantic.json 结构

```jsonc
{
  "deck_title": "string",
  "author": "string",
  "date": "YYYY-MM-DD",
  "language": "zh-CN | en-US",
  "slides": [
    {
      "slide_id": 1,
      "title": "string",
      "slide_type": "title | bullet-list | two-column | comparison | data-heavy | matrix | flowchart | timeline | gantt | technical | process | call_to_action | decision",
      "slide_role": "situation | complication | question | answer | evidence | action | next_steps",
      "content": ["bullet 1", "bullet 2"],
      "speaker_notes": {
        "summary": "string",
        "rationale": "string",
        "evidence": "string",
        "audience_action": "string",
        "risks": "string"
      },
      "visual": {
        "type": "none | comparison | matrix | sequence | flowchart | timeline | gantt | kpi_dashboard | engineering_schematic | ...",
        "title": "string",
        "priority": "critical | high | medium | low",
        "data_source": "string",
        "content_requirements": ["string"],
        "placeholder_data": {
          "chart_config": { "labels": [], "series": [] },
          "mermaid_code": "string"
        }
      },
      "metadata": { "priority": "critical | high | medium | low", "requires_diagram": true }
    }
  ]
}
```

**解析规则**：
- `slides` 数组按 `slide_id` 顺序渲染
- `visual` 为 `null` 或 `{"type": "none"}` 时为纯文本页
- `placeholder_data` 中的 `chart_config` 用于渲染表格/图表
- `placeholder_data` 中的 `mermaid_code` 用于渲染流程图/时序图/甘特图
- `speaker_notes` 各字段拼接为演讲者备注（保留原文，不重写）

### 2.2 design_spec.json 结构

```jsonc
{
  "color_system": {
    "primary": "#hex",
    "on_primary": "#hex",
    "primary_container": "#hex",
    "secondary": "#hex",
    "surface": "#hex",
    "on_surface": "#hex",
    "error": "#hex",
    "outline": "#hex",
    // ... Material Design 3 tokens
  },
  "typography": {
    "font_families": { "en": "Roboto, ...", "zh": "Noto Sans SC, ..." },
    "type_scale": {
      "headline_medium": { "size_pt": 28, "weight": 600 },
      "title_large":     { "size_pt": 20, "weight": 600 },
      "body_large":      { "size_pt": 18, "weight": 400 },
      "body_medium":     { "size_pt": 14, "weight": 400 },
      "label_large":     { "size_pt": 12, "weight": 600 }
    }
  },
  "spacing_system": { "base_unit": 4, "scale": [4,8,12,16,24,32,48] },
  "shape": { "corner_radius": { "small": 4, "medium": 8, "large": 16 } },
  "elevation": { "level_0": "none", "level_1": "...", "level_2": "..." },
  "grid_system": {
    "columns": 12, "gutter": 24,
    "margin_horizontal": 80,
    "slide_width_px": 1920, "slide_height_px": 1080
  },
  "component_library": {
    "card": { "padding": 24, "corner_radius": 8, "elevation": "level_1" },
    "callout": { "border_left": "4px solid primary", "background": "primary_container" },
    "data_table": { "header_weight": 600, "row_height": 48 },
    "chart_palette": ["#hex", ...],
    // ...
  },
  "accessibility_specs": {
    "contrast_requirements": { "normal_text": ">=4.5", "large_text": ">=3.0" },
    "colorblind": "..."
  }
}
```

**Token 加载**：

```python
import json
from pptx.dml.color import RGBColor

def load_design_spec(path: str) -> dict:
    """加载 design_spec.json 并构建 token 查找表"""
    with open(path) as f:
        spec = json.load(f)
    return spec

def hex_to_rgb(hex_str: str) -> RGBColor:
    """'#2563EB' → RGBColor(0x25, 0x63, 0xEB)"""
    h = hex_str.lstrip('#')
    return RGBColor(int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))

def get_color(spec: dict, token_name: str) -> RGBColor:
    """从 color_system 获取颜色 token"""
    return hex_to_rgb(spec['color_system'][token_name])

def get_font_size(spec: dict, scale_name: str) -> int:
    """从 type_scale 获取字号 (pt)"""
    return spec['typography']['type_scale'][scale_name]['size_pt']

def get_font_weight(spec: dict, scale_name: str) -> bool:
    """判断是否加粗 (weight >= 600)"""
    return spec['typography']['type_scale'][scale_name]['weight'] >= 600
```

---

## 3. 设计 Token 系统

### 3.1 颜色映射

从 `design_spec.json` 的 `color_system` 构建颜色查找表，所有元素的颜色**必须**来自 token：

| 元素 | Token | 用途 |
|------|-------|------|
| 标题栏背景 | `primary` | 深色标题栏 |
| 标题栏文字 | `on_primary` | 白色标题文字 |
| 页面背景 | `surface` | 浅色背景 |
| 正文文字 | `on_surface` | 主要内容颜色 |
| 辅助文字 | `outline` | 副标题、注释 |
| 强调色 | `primary` | 分割线、色条 |
| 卡片背景 | `primary_container` | 轻色卡片 |
| 卡片文字 | `on_primary_container` | 卡片内文字 |
| 成功/进展 | `secondary` | 正面指标 |
| 警告 | `tertiary` | 需关注项 |
| 错误/风险 | `error` | 高风险项 |

### 3.2 字体规格

| 元素 | type_scale | 中文最小值 | 英文最小值 |
|------|-----------|-----------|-----------|
| 页标题 | `headline_medium` | 28pt | 24pt |
| 副标题 | `title_large` | 20pt | 18pt |
| 正文 | `body_large` | 18pt | 16pt |
| 注释/标签 | `body_medium` | 14pt | 12pt |
| 数据标签 | `label_large` | 12pt | 11pt |

**中文排版规则**：
- 行高: ≥ 1.5（中文）, ≥ 1.3（英文）
- 字体: 优先 `Noto Sans SC`，回退 `PingFang SC` / `Microsoft YaHei`
- 中英混排: Noto Sans SC 自动处理基线对齐

### 3.3 间距系统

基于 `spacing_system.base_unit = 4` 的倍数体系：

| 语义 | 计算 | 值 (px → inches) |
|------|------|------------------|
| 页边距 | `margin_horizontal` from grid_system | 80px ≈ 0.83" |
| 内容上边距 | 标题栏高 + gutter | ~1.1" |
| 栏间距 | `gutter` from grid_system | 24px ≈ 0.25" |
| 卡片内边距 | `component_library.card.padding` | 24px ≈ 0.25" |
| 元素间距 | `scale[5]` = 32px | 0.33" |

**px → inches 换算**（基于 96 DPI 标准）：

```python
def px_to_inches(px: int) -> float:
    return px / 96.0

def px_to_emu(px: int) -> int:
    return int(px / 96.0 * 914400)
```

---

## 4. 网格布局系统

### 4.1 12 列网格

基于 `grid_system`：
- 幻灯片宽度: 1920px (13.33")
- 水平边距: 80px (0.83") × 2
- 可用宽度: 1920 - 160 = 1760px (12.22")
- 栏间距: 24px (0.17")
- 单列宽度: (1760 - 24×11) / 12 ≈ 124.67px (0.87")

```python
from pptx.util import Inches, Pt, Emu

class GridSystem:
    def __init__(self, spec: dict):
        grid = spec['grid_system']
        self.slide_w = grid['slide_width_px']
        self.slide_h = grid['slide_height_px']
        self.margin_h = grid['margin_horizontal']
        self.gutter = grid['gutter']
        self.columns = grid['columns']
        self.usable_w = self.slide_w - 2 * self.margin_h
        self.col_w = (self.usable_w - self.gutter * (self.columns - 1)) / self.columns

    def col_span(self, n_cols: int, start_col: int = 0) -> tuple[float, float]:
        """返回 (left_inches, width_inches) 基于跨列数"""
        left_px = self.margin_h + start_col * (self.col_w + self.gutter)
        width_px = n_cols * self.col_w + (n_cols - 1) * self.gutter
        return px_to_inches(left_px), px_to_inches(width_px)

    def content_area(self, title_bar_h_inches: float = 0.75) -> dict:
        """返回内容区域的 top 和 height (inches)"""
        top = title_bar_h_inches + px_to_inches(self.gutter)
        height = px_to_inches(self.slide_h) - top - px_to_inches(self.gutter * 2)
        return {'top': top, 'height': height}
```

### 4.2 标准布局模板

根据 `slide_type` 选择布局：

| slide_type | 布局策略 | 列分配 |
|------------|---------|--------|
| `title` | 居中全宽 | 12 列 |
| `bullet-list` | 左内容 + 右空/图 | 7+5 或 12 |
| `two-column` | 左右均分 | 6+6 |
| `comparison` | 左右均分 | 6+6 |
| `data-heavy` | 上 KPI 卡 + 下图表 | 12 列分区 |
| `matrix` | 全宽矩阵 | 12 列 |
| `flowchart` | 全宽流程图 | 12 列 |
| `timeline` | 全宽时间线 | 12 列 |
| `gantt` | 全宽甘特图 | 12 列 |
| `technical` | 左文字 + 右示意图 | 5+7 |
| `process` | 流程步骤 | 12 列等分 |
| `call_to_action` | 居中突出 | 8 列居中 |
| `decision` | 左请求 + 右矩阵 | 5+7 |

```python
def get_layout(slide_type: str, has_visual: bool) -> dict:
    """根据 slide_type 返回布局规格"""
    layouts = {
        'title': {
            'content': {'start_col': 1, 'span': 10, 'valign': 'middle'},
        },
        'bullet-list': {
            'content': {'start_col': 0, 'span': 7 if has_visual else 12},
            'visual':  {'start_col': 7, 'span': 5} if has_visual else None,
        },
        'two-column': {
            'left':  {'start_col': 0, 'span': 6},
            'right': {'start_col': 6, 'span': 6},
        },
        'comparison': {
            'left':  {'start_col': 0, 'span': 6},
            'right': {'start_col': 6, 'span': 6},
        },
        'data-heavy': {
            'kpi_row':  {'start_col': 0, 'span': 12, 'height_ratio': 0.3},
            'chart':    {'start_col': 0, 'span': 12, 'height_ratio': 0.65},
        },
        'matrix': {
            'content': {'start_col': 0, 'span': 12},
        },
        'flowchart': {
            'content': {'start_col': 0, 'span': 12},
        },
        'timeline': {
            'content': {'start_col': 0, 'span': 12},
        },
        'gantt': {
            'content': {'start_col': 0, 'span': 12},
        },
        'technical': {
            'content': {'start_col': 0, 'span': 5},
            'visual':  {'start_col': 5, 'span': 7},
        },
        'decision': {
            'content': {'start_col': 0, 'span': 5},
            'visual':  {'start_col': 5, 'span': 7},
        },
        'process': {
            'content': {'start_col': 0, 'span': 12},
        },
        'call_to_action': {
            'content': {'start_col': 2, 'span': 8, 'valign': 'middle'},
        },
    }
    return layouts.get(slide_type, layouts['bullet-list'])
```

---

## 5. 页面渲染流程

### 5.1 通用页面结构

每一页幻灯片的渲染遵循统一结构：

```
┌─────────────────────────────────────────────┐
│ 标题栏 (title_bar)                           │ ← primary 背景
│   [Section 标签]         [标题文字]  [页码]   │ ← on_primary 文字
├─────────────────────────────────────────────┤
│                                             │
│  内容区域 (content_area)                     │ ← surface 背景
│  ┌─────────────┐  ┌───────────────────────┐ │
│  │ 文字/要点    │  │ 图表/可视化            │ │
│  │             │  │                       │ │
│  └─────────────┘  └───────────────────────┘ │
│                                             │
├─────────────────────────────────────────────┤
│ 底部装饰线                                   │ ← primary 颜色
└─────────────────────────────────────────────┘
```

### 5.2 渲染函数 — 标题栏

```python
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

def render_title_bar(slide, spec: dict, grid: GridSystem,
                     title: str, slide_id: int, section_label: str = ''):
    """渲染统一标题栏 (深色背景 + 白色文字)"""
    colors = spec['color_system']
    typo = spec['typography']

    bar_h = Inches(0.75)
    slide_w_in = px_to_inches(grid.slide_w)

    # 标题栏背景
    bar = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0), Inches(0),
        Inches(slide_w_in), bar_h
    )
    bar.fill.solid()
    bar.fill.fore_color.rgb = hex_to_rgb(colors['primary'])
    bar.line.fill.background()

    # Section 标签 (可选)
    if section_label:
        tb = slide.shapes.add_textbox(
            Inches(px_to_inches(grid.margin_h)), Inches(0.12),
            Inches(4), Inches(0.25)
        )
        tf = tb.text_frame
        tf.text = section_label
        p = tf.paragraphs[0]
        p.font.size = Pt(get_font_size(spec, 'label_large'))
        p.font.bold = True
        p.font.color.rgb = hex_to_rgb(colors['on_primary'])

    # 标题文字
    margin_left = px_to_inches(grid.margin_h)
    title_top = Inches(0.28) if section_label else Inches(0.18)
    tb = slide.shapes.add_textbox(
        Inches(margin_left), title_top,
        Inches(slide_w_in - 2 * margin_left - 1.0), Inches(0.45)
    )
    tf = tb.text_frame
    tf.text = title
    p = tf.paragraphs[0]
    p.font.size = Pt(get_font_size(spec, 'headline_medium'))
    p.font.bold = get_font_weight(spec, 'headline_medium')
    p.font.color.rgb = hex_to_rgb(colors['on_primary'])

    # 页码
    tb_num = slide.shapes.add_textbox(
        Inches(slide_w_in - 1.2), Inches(0.22),
        Inches(0.8), Inches(0.4)
    )
    tf_num = tb_num.text_frame
    tf_num.text = f"{slide_id:02d}"
    p = tf_num.paragraphs[0]
    p.font.size = Pt(get_font_size(spec, 'headline_medium'))
    p.font.bold = True
    p.font.color.rgb = hex_to_rgb(colors['on_primary'])
    p.alignment = PP_ALIGN.RIGHT

    return bar_h
```

### 5.3 渲染函数 — 底部装饰线

```python
def render_bottom_bar(slide, spec: dict, grid: GridSystem):
    """底部装饰线"""
    slide_w = px_to_inches(grid.slide_w)
    slide_h = px_to_inches(grid.slide_h)
    bar = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE,
        Inches(0), Inches(slide_h - 0.05),
        Inches(slide_w), Inches(0.05)
    )
    bar.fill.solid()
    bar.fill.fore_color.rgb = hex_to_rgb(spec['color_system']['primary'])
    bar.line.fill.background()
```

### 5.4 渲染函数 — 演讲者笔记

```python
def render_speaker_notes(slide, notes_data: dict):
    """将 speaker_notes 字段拼接写入演讲者备注（保留原文）"""
    if not notes_data:
        return
    parts = []
    field_labels = {
        'summary': 'Summary',
        'rationale': 'Rationale',
        'evidence': 'Evidence',
        'audience_action': 'Audience Action',
        'risks': 'Risks'
    }
    for key, label in field_labels.items():
        if key in notes_data and notes_data[key]:
            parts.append(f"{label}: {notes_data[key]}")

    notes_frame = slide.notes_slide.notes_text_frame
    notes_frame.text = '\n\n'.join(parts)
```

---

## 6. 按 slide_type 渲染

### 6.1 title (封面/标题页)

```python
def render_slide_title(slide, slide_data: dict, spec: dict, grid: GridSystem):
    """全宽居中标题页"""
    content_area = grid.content_area()
    margin = px_to_inches(grid.margin_h)

    # 主结论文字 (居中大号)
    for i, bullet in enumerate(slide_data.get('content', [])):
        tb = slide.shapes.add_textbox(
            Inches(margin + 1.0),
            Inches(content_area['top'] + 1.5 + i * 0.6),
            Inches(px_to_inches(grid.slide_w) - 2 * (margin + 1.0)),
            Inches(0.5)
        )
        tf = tb.text_frame
        tf.word_wrap = True
        tf.text = bullet
        p = tf.paragraphs[0]
        p.font.size = Pt(get_font_size(spec, 'title_large'))
        p.font.bold = True
        p.font.color.rgb = hex_to_rgb(spec['color_system']['on_surface'])
        p.alignment = PP_ALIGN.CENTER
```

### 6.2 bullet-list (要点列表)

```python
def render_slide_bullets(slide, slide_data: dict, spec: dict, grid: GridSystem):
    """左侧要点 + 可选右侧可视化"""
    has_visual = slide_data.get('visual') is not None and \
                 slide_data['visual'].get('type') not in (None, 'none')
    layout = get_layout('bullet-list', has_visual)

    content_area = grid.content_area()
    left, width = grid.col_span(
        layout['content']['span'],
        layout['content']['start_col']
    )

    # 渲染 bullet 列表
    bullet_top = content_area['top'] + 0.3
    for i, bullet in enumerate(slide_data.get('content', [])):
        tb = slide.shapes.add_textbox(
            Inches(left + 0.1), Inches(bullet_top + i * 0.55),
            Inches(width - 0.2), Inches(0.5)
        )
        tf = tb.text_frame
        tf.word_wrap = True
        tf.text = f"• {bullet}"
        p = tf.paragraphs[0]
        p.font.size = Pt(get_font_size(spec, 'body_large'))
        p.font.color.rgb = hex_to_rgb(spec['color_system']['on_surface'])
        p.line_spacing = 1.5

    # 如有可视化，在右侧渲染
    if has_visual and layout.get('visual'):
        v_left, v_width = grid.col_span(
            layout['visual']['span'],
            layout['visual']['start_col']
        )
        render_visual(slide, slide_data['visual'], spec, grid,
                      v_left, content_area['top'], v_width, content_area['height'])
```

### 6.3 two-column / comparison (双栏对比)

```python
def render_slide_two_column(slide, slide_data: dict, spec: dict, grid: GridSystem):
    """双栏布局 — 适用于 two-column, comparison, decision"""
    layout = get_layout(slide_data['slide_type'], True)
    content_area = grid.content_area()
    colors = spec['color_system']

    # 左栏内容 (bullets)
    l_left, l_width = grid.col_span(
        layout['left']['span'], layout['left']['start_col']
    )
    for i, bullet in enumerate(slide_data.get('content', [])):
        tb = slide.shapes.add_textbox(
            Inches(l_left + 0.1), Inches(content_area['top'] + 0.3 + i * 0.55),
            Inches(l_width - 0.2), Inches(0.5)
        )
        tf = tb.text_frame
        tf.word_wrap = True
        tf.text = f"• {bullet}"
        p = tf.paragraphs[0]
        p.font.size = Pt(get_font_size(spec, 'body_large'))
        p.font.color.rgb = hex_to_rgb(colors['on_surface'])

    # 右栏可视化
    if slide_data.get('visual'):
        r_left, r_width = grid.col_span(
            layout['right']['span'], layout['right']['start_col']
        )
        render_visual(slide, slide_data['visual'], spec, grid,
                      r_left, content_area['top'], r_width, content_area['height'])
```

### 6.4 data-heavy (数据密集型)

```python
def render_slide_data_heavy(slide, slide_data: dict, spec: dict, grid: GridSystem):
    """上方 KPI 卡片行 + 下方图表"""
    content_area = grid.content_area()
    margin = px_to_inches(grid.margin_h)
    full_width = px_to_inches(grid.usable_w)
    colors = spec['color_system']

    # 上方 bullets 作为 KPI 卡片
    kpi_top = content_area['top'] + 0.2
    kpi_items = slide_data.get('content', [])
    card_w = (full_width - 0.2 * (len(kpi_items) - 1)) / max(len(kpi_items), 1)

    for i, kpi in enumerate(kpi_items):
        card_left = margin + i * (card_w + 0.2)
        # 卡片背景
        card = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE,
            Inches(card_left), Inches(kpi_top),
            Inches(card_w), Inches(0.8)
        )
        card.fill.solid()
        card.fill.fore_color.rgb = hex_to_rgb(colors['primary_container'])
        card.line.fill.background()

        # 卡片文字
        tb = slide.shapes.add_textbox(
            Inches(card_left + 0.15), Inches(kpi_top + 0.15),
            Inches(card_w - 0.3), Inches(0.5)
        )
        tf = tb.text_frame
        tf.word_wrap = True
        tf.text = kpi
        p = tf.paragraphs[0]
        p.font.size = Pt(get_font_size(spec, 'body_medium'))
        p.font.color.rgb = hex_to_rgb(colors['on_primary_container'])

    # 下方可视化
    if slide_data.get('visual'):
        chart_top = kpi_top + 1.2
        render_visual(slide, slide_data['visual'], spec, grid,
                      margin, chart_top, full_width,
                      content_area['height'] - 1.4)
```

### 6.5 matrix (矩阵类)

当 `visual.placeholder_data` 含 `chart_config` 时渲染为数据表格；含 `mermaid_code` 时渲染为象限图。

```python
def render_slide_matrix(slide, slide_data: dict, spec: dict, grid: GridSystem):
    """矩阵布局 — 表格或象限"""
    content_area = grid.content_area()
    margin = px_to_inches(grid.margin_h)
    full_width = px_to_inches(grid.usable_w)

    # bullets
    for i, bullet in enumerate(slide_data.get('content', [])):
        tb = slide.shapes.add_textbox(
            Inches(margin + 0.1), Inches(content_area['top'] + 0.2 + i * 0.45),
            Inches(full_width - 0.2), Inches(0.4)
        )
        tf = tb.text_frame
        tf.word_wrap = True
        tf.text = f"• {bullet}"
        p = tf.paragraphs[0]
        p.font.size = Pt(get_font_size(spec, 'body_large'))
        p.font.color.rgb = hex_to_rgb(spec['color_system']['on_surface'])

    # 可视化
    if slide_data.get('visual'):
        vis_top = content_area['top'] + 0.2 + len(slide_data.get('content', [])) * 0.45 + 0.3
        render_visual(slide, slide_data['visual'], spec, grid,
                      margin, vis_top, full_width,
                      content_area['height'] - vis_top + content_area['top'])
```

### 6.6 timeline / gantt (时间线)

```python
def render_slide_timeline(slide, slide_data: dict, spec: dict, grid: GridSystem):
    """时间线 / 甘特图 — bullets + 全宽可视化"""
    content_area = grid.content_area()
    margin = px_to_inches(grid.margin_h)
    full_width = px_to_inches(grid.usable_w)
    colors = spec['color_system']

    # 里程碑要点
    milestones = slide_data.get('content', [])
    item_w = (full_width - 0.3 * (len(milestones) - 1)) / max(len(milestones), 1)

    for i, ms in enumerate(milestones):
        x = margin + i * (item_w + 0.3)
        y = content_area['top'] + 0.3

        # 圆点
        dot = slide.shapes.add_shape(
            MSO_SHAPE.OVAL,
            Inches(x + item_w / 2 - 0.08), Inches(y),
            Inches(0.16), Inches(0.16)
        )
        dot.fill.solid()
        dot.fill.fore_color.rgb = hex_to_rgb(colors['primary'])
        dot.line.fill.background()

        # 文字
        tb = slide.shapes.add_textbox(
            Inches(x), Inches(y + 0.25),
            Inches(item_w), Inches(1.2)
        )
        tf = tb.text_frame
        tf.word_wrap = True
        tf.text = ms
        p = tf.paragraphs[0]
        p.font.size = Pt(get_font_size(spec, 'body_medium'))
        p.font.color.rgb = hex_to_rgb(colors['on_surface'])
        p.alignment = PP_ALIGN.CENTER

    # 连接线
    if len(milestones) > 1:
        line_y = content_area['top'] + 0.38
        connector = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE,
            Inches(margin + item_w / 2),
            Inches(line_y),
            Inches(full_width - item_w),
            Pt(2)
        )
        connector.fill.solid()
        connector.fill.fore_color.rgb = hex_to_rgb(colors['outline'])
        connector.line.fill.background()

    # 如有甘特图 mermaid
    if slide_data.get('visual') and slide_data['visual'].get('placeholder_data', {}).get('mermaid_code'):
        # mermaid 甘特图需要外部渲染或文本占位
        render_visual(slide, slide_data['visual'], spec, grid,
                      margin, content_area['top'] + 2.0, full_width,
                      content_area['height'] - 2.2)
```

### 6.7 flowchart (流程图)

```python
def render_slide_flowchart(slide, slide_data: dict, spec: dict, grid: GridSystem):
    """流程图 — bullets + 全宽 mermaid 可视化"""
    content_area = grid.content_area()
    margin = px_to_inches(grid.margin_h)
    full_width = px_to_inches(grid.usable_w)

    # 简要说明
    for i, bullet in enumerate(slide_data.get('content', [])):
        tb = slide.shapes.add_textbox(
            Inches(margin + 0.1), Inches(content_area['top'] + 0.2 + i * 0.4),
            Inches(full_width - 0.2), Inches(0.35)
        )
        tf = tb.text_frame
        tf.word_wrap = True
        tf.text = f"• {bullet}"
        p = tf.paragraphs[0]
        p.font.size = Pt(get_font_size(spec, 'body_large'))
        p.font.color.rgb = hex_to_rgb(spec['color_system']['on_surface'])

    # 流程图可视化
    if slide_data.get('visual'):
        vis_top = content_area['top'] + 0.2 + len(slide_data.get('content', [])) * 0.4 + 0.3
        render_visual(slide, slide_data['visual'], spec, grid,
                      margin, vis_top, full_width,
                      content_area['height'] - vis_top + content_area['top'])
```

### 6.8 call_to_action (行动号召)

```python
def render_slide_cta(slide, slide_data: dict, spec: dict, grid: GridSystem):
    """行动号召页 — 居中突出"""
    content_area = grid.content_area()
    colors = spec['color_system']
    layout = get_layout('call_to_action', False)
    left, width = grid.col_span(
        layout['content']['span'], layout['content']['start_col']
    )

    # 强调卡片
    card_top = content_area['top'] + 0.5
    card = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(left), Inches(card_top),
        Inches(width), Inches(content_area['height'] - 1.0)
    )
    card.fill.solid()
    card.fill.fore_color.rgb = hex_to_rgb(colors['primary_container'])
    card.line.fill.background()

    # 内容
    for i, bullet in enumerate(slide_data.get('content', [])):
        tb = slide.shapes.add_textbox(
            Inches(left + 0.4), Inches(card_top + 0.4 + i * 0.65),
            Inches(width - 0.8), Inches(0.55)
        )
        tf = tb.text_frame
        tf.word_wrap = True
        tf.text = bullet
        p = tf.paragraphs[0]
        p.font.size = Pt(get_font_size(spec, 'title_large'))
        p.font.bold = True
        p.font.color.rgb = hex_to_rgb(colors['on_primary_container'])
        p.alignment = PP_ALIGN.LEFT
```

### 6.9 渲染分派器

```python
# slide_type → 渲染函数映射
RENDERERS = {
    'title':          render_slide_title,
    'bullet-list':    render_slide_bullets,
    'two-column':     render_slide_two_column,
    'comparison':     render_slide_two_column,
    'decision':       render_slide_two_column,
    'data-heavy':     render_slide_data_heavy,
    'matrix':         render_slide_matrix,
    'flowchart':      render_slide_flowchart,
    'timeline':       render_slide_timeline,
    'gantt':          render_slide_timeline,
    'technical':      render_slide_two_column,
    'process':        render_slide_flowchart,
    'call_to_action': render_slide_cta,
}

def render_slide(prs, slide_data: dict, spec: dict, grid: GridSystem, section_map: dict):
    """渲染单页幻灯片"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # 空白布局

    # 背景
    bg = slide.background.fill
    bg.solid()
    bg.fore_color.rgb = hex_to_rgb(spec['color_system']['surface'])

    # 标题栏
    section_label = section_map.get(slide_data['slide_id'], '')
    render_title_bar(slide, spec, grid,
                     slide_data['title'], slide_data['slide_id'], section_label)

    # 内容 (按 slide_type 分派)
    renderer = RENDERERS.get(slide_data['slide_type'], render_slide_bullets)
    renderer(slide, slide_data, spec, grid)

    # 底部装饰
    render_bottom_bar(slide, spec, grid)

    # 演讲者笔记
    render_speaker_notes(slide, slide_data.get('speaker_notes'))

    return slide
```

---

## 7. 可视化渲染

### 7.1 统一入口

```python
def render_visual(slide, visual: dict, spec: dict, grid: GridSystem,
                  left: float, top: float, width: float, height: float):
    """根据 visual.type 渲染可视化内容"""
    if not visual or visual.get('type') in (None, 'none'):
        return

    vtype = visual['type']
    pd = visual.get('placeholder_data', {})

    if 'chart_config' in pd:
        render_chart_table(slide, visual, spec, left, top, width, height)
    elif 'mermaid_code' in pd:
        render_mermaid_placeholder(slide, visual, spec, left, top, width, height)
    else:
        render_visual_placeholder(slide, visual, spec, left, top, width, height)
```

### 7.2 chart_config → 数据表格

当 `placeholder_data.chart_config` 存在时，渲染为 Material 风格数据表格：

```python
def render_chart_table(slide, visual: dict, spec: dict,
                       left: float, top: float, width: float, height: float):
    """将 chart_config 渲染为 Material 风格数据表"""
    config = visual['placeholder_data']['chart_config']
    labels = config.get('labels', [])
    series = config.get('series', [])
    colors = spec['color_system']
    palette = spec['component_library'].get('chart_palette', [colors['primary']])

    if not series:
        return

    # 表格维度
    n_rows = len(series) + 1  # header + data
    n_cols = len(labels)
    col_w = width / max(n_cols, 1)
    row_h = min(0.48, height / max(n_rows, 1))

    # 表头
    for j, label in enumerate(labels):
        tb = slide.shapes.add_textbox(
            Inches(left + j * col_w), Inches(top),
            Inches(col_w), Inches(row_h)
        )
        tf = tb.text_frame
        tf.text = label
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.font.size = Pt(get_font_size(spec, 'label_large'))
        p.font.bold = True
        p.font.color.rgb = hex_to_rgb(colors['on_surface'])

    # 表头分割线
    line = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE,
        Inches(left), Inches(top + row_h - 0.02),
        Inches(width), Pt(2)
    )
    line.fill.solid()
    line.fill.fore_color.rgb = hex_to_rgb(colors['primary'])
    line.line.fill.background()

    # 数据行
    for r, s in enumerate(series):
        row_top = top + (r + 1) * row_h
        # 斑马纹
        if r % 2 == 1:
            stripe = slide.shapes.add_shape(
                MSO_SHAPE.RECTANGLE,
                Inches(left), Inches(row_top),
                Inches(width), Inches(row_h)
            )
            stripe.fill.solid()
            stripe.fill.fore_color.rgb = hex_to_rgb(colors.get('surface_variant', '#E1E2EC'))
            stripe.line.fill.background()

        data = s.get('data', [])
        for j, val in enumerate(data):
            tb = slide.shapes.add_textbox(
                Inches(left + j * col_w), Inches(row_top),
                Inches(col_w), Inches(row_h)
            )
            tf = tb.text_frame
            tf.text = str(val)
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.font.size = Pt(get_font_size(spec, 'body_medium'))
            p.font.color.rgb = hex_to_rgb(colors['on_surface'])
            # 数字右对齐
            if isinstance(val, (int, float)):
                p.alignment = PP_ALIGN.RIGHT

    # 表格标题
    if visual.get('title'):
        tb = slide.shapes.add_textbox(
            Inches(left), Inches(top - 0.35),
            Inches(width), Inches(0.3)
        )
        tf = tb.text_frame
        tf.text = visual['title']
        p = tf.paragraphs[0]
        p.font.size = Pt(get_font_size(spec, 'body_medium'))
        p.font.bold = True
        p.font.color.rgb = hex_to_rgb(colors['on_surface'])
```

### 7.3 mermaid_code → 占位渲染

Mermaid 代码需要外部渲染器（如 mermaid-cli）生成 PNG 后嵌入。短期方案为生成结构化文本占位：

```python
def render_mermaid_placeholder(slide, visual: dict, spec: dict,
                               left: float, top: float, width: float, height: float):
    """Mermaid 代码占位 — 显示标题 + 代码预览"""
    colors = spec['color_system']
    mermaid = visual['placeholder_data']['mermaid_code']

    # 占位卡片
    card = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(left), Inches(top + 0.1),
        Inches(width), Inches(height - 0.2)
    )
    card.fill.solid()
    card.fill.fore_color.rgb = hex_to_rgb(colors.get('surface_variant', '#E1E2EC'))
    card.line.color.rgb = hex_to_rgb(colors['outline'])
    card.line.width = Pt(1)

    # 标题
    if visual.get('title'):
        tb = slide.shapes.add_textbox(
            Inches(left + 0.2), Inches(top + 0.2),
            Inches(width - 0.4), Inches(0.3)
        )
        tf = tb.text_frame
        tf.text = f"📊 {visual['title']}"
        p = tf.paragraphs[0]
        p.font.size = Pt(get_font_size(spec, 'body_medium'))
        p.font.bold = True
        p.font.color.rgb = hex_to_rgb(colors['on_surface'])

    # Mermaid 代码预览 (截取前 8 行)
    preview_lines = mermaid.strip().split('\n')[:8]
    preview = '\n'.join(preview_lines)
    if len(mermaid.strip().split('\n')) > 8:
        preview += '\n  ...'

    tb = slide.shapes.add_textbox(
        Inches(left + 0.2), Inches(top + 0.6),
        Inches(width - 0.4), Inches(height - 1.0)
    )
    tf = tb.text_frame
    tf.word_wrap = True
    tf.text = preview
    p = tf.paragraphs[0]
    p.font.size = Pt(11)
    p.font.color.rgb = hex_to_rgb(colors['outline'])
```

### 7.4 Mermaid 外部渲染 (进阶)

当系统安装了 `mmdc` (mermaid-cli) 时，可自动渲染为 PNG 并嵌入：

```python
import subprocess, tempfile, os

def render_mermaid_to_png(mermaid_code: str, output_path: str,
                          width: int = 1200, bg_color: str = 'transparent') -> bool:
    """调用 mmdc 将 mermaid 代码渲染为 PNG"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.mmd', delete=False) as f:
        f.write(mermaid_code)
        mmd_path = f.name
    try:
        result = subprocess.run(
            ['mmdc', '-i', mmd_path, '-o', output_path,
             '-w', str(width), '-b', bg_color, '--scale', '2'],
            capture_output=True, text=True, timeout=30
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    finally:
        os.unlink(mmd_path)

def embed_diagram_image(slide, img_path: str,
                        left: float, top: float, width: float, height: float):
    """嵌入已渲染的图表图片"""
    slide.shapes.add_picture(
        img_path,
        Inches(left), Inches(top),
        width=Inches(width)
        # height 自适应保持比例
    )
```

### 7.5 无占位数据的可视化

```python
def render_visual_placeholder(slide, visual: dict, spec: dict,
                              left: float, top: float, width: float, height: float):
    """无占位数据时的通用占位框"""
    colors = spec['color_system']

    card = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(left), Inches(top + 0.1),
        Inches(width), Inches(min(height - 0.2, 2.5))
    )
    card.fill.solid()
    card.fill.fore_color.rgb = hex_to_rgb(colors.get('surface_variant', '#E1E2EC'))
    card.line.color.rgb = hex_to_rgb(colors['outline'])
    card.line.width = Pt(1)
    card.line.dash_style = 2  # dash

    # 占位文字
    label = visual.get('title', visual.get('type', 'Visual'))
    reqs = visual.get('content_requirements', [])
    text = f"[{label}]"
    if reqs:
        text += '\n' + '\n'.join(f"  • {r}" for r in reqs[:3])

    tb = slide.shapes.add_textbox(
        Inches(left + 0.3), Inches(top + 0.3),
        Inches(width - 0.6), Inches(min(height - 0.6, 2.0))
    )
    tf = tb.text_frame
    tf.word_wrap = True
    tf.text = text
    p = tf.paragraphs[0]
    p.font.size = Pt(get_font_size(spec, 'body_medium'))
    p.font.color.rgb = hex_to_rgb(colors['outline'])
    p.alignment = PP_ALIGN.CENTER
```

---

## 8. 组件库渲染

基于 `design_spec.json` 的 `component_library` 渲染通用组件。

### 8.1 Material 卡片

```python
def render_card(slide, spec: dict, left: float, top: float,
                width: float, height: float, content_text: str,
                title_text: str = '', variant: str = 'surface'):
    """渲染 Material Design 卡片"""
    comp = spec['component_library']['card']
    colors = spec['color_system']
    shape_spec = spec['shape']

    bg_color = colors.get(variant, colors['surface'])
    text_color = colors.get(f'on_{variant}', colors['on_surface'])

    card = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(left), Inches(top),
        Inches(width), Inches(height)
    )
    card.fill.solid()
    card.fill.fore_color.rgb = hex_to_rgb(bg_color)
    card.line.fill.background()

    # 阴影 (level_1)
    add_shadow(card, spec)

    y_offset = top + px_to_inches(comp['padding'])

    if title_text:
        tb = slide.shapes.add_textbox(
            Inches(left + px_to_inches(comp['padding'])),
            Inches(y_offset),
            Inches(width - 2 * px_to_inches(comp['padding'])),
            Inches(0.3)
        )
        tf = tb.text_frame
        tf.text = title_text
        p = tf.paragraphs[0]
        p.font.size = Pt(get_font_size(spec, 'title_large'))
        p.font.bold = True
        p.font.color.rgb = hex_to_rgb(text_color)
        y_offset += 0.4

    tb = slide.shapes.add_textbox(
        Inches(left + px_to_inches(comp['padding'])),
        Inches(y_offset),
        Inches(width - 2 * px_to_inches(comp['padding'])),
        Inches(height - (y_offset - top) - px_to_inches(comp['padding']))
    )
    tf = tb.text_frame
    tf.word_wrap = True
    tf.text = content_text
    p = tf.paragraphs[0]
    p.font.size = Pt(get_font_size(spec, 'body_large'))
    p.font.color.rgb = hex_to_rgb(text_color)

    return card
```

### 8.2 Callout (提示框)

```python
def render_callout(slide, spec: dict, left: float, top: float,
                   width: float, height: float, text: str):
    """渲染 Callout 提示框 (左侧色条 + 浅色背景)"""
    colors = spec['color_system']

    # 背景
    bg = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(left), Inches(top),
        Inches(width), Inches(height)
    )
    bg.fill.solid()
    bg.fill.fore_color.rgb = hex_to_rgb(colors['primary_container'])
    bg.line.fill.background()

    # 左侧色条
    bar = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE,
        Inches(left), Inches(top),
        Inches(0.05), Inches(height)
    )
    bar.fill.solid()
    bar.fill.fore_color.rgb = hex_to_rgb(colors['primary'])
    bar.line.fill.background()

    # 文字
    tb = slide.shapes.add_textbox(
        Inches(left + 0.2), Inches(top + 0.1),
        Inches(width - 0.3), Inches(height - 0.2)
    )
    tf = tb.text_frame
    tf.word_wrap = True
    tf.text = text
    p = tf.paragraphs[0]
    p.font.size = Pt(get_font_size(spec, 'body_large'))
    p.font.color.rgb = hex_to_rgb(colors['on_primary_container'])
```

### 8.3 阴影辅助

```python
from pptx.oxml.ns import qn

def add_shadow(shape, spec: dict, blur_pt: int = 6, offset_pt: int = 2):
    """为形状添加 Material elevation Level 1 阴影"""
    spPr = shape._element.spPr
    effectLst = spPr.makeelement(qn('a:effectLst'), {})
    outerShdw = effectLst.makeelement(qn('a:outerShdw'), {
        'blurRad': str(int(Pt(blur_pt))),
        'dist': str(int(Pt(offset_pt))),
        'dir': '5400000',
        'algn': 'bl',
        'rotWithShape': '0',
    })
    srgbClr = outerShdw.makeelement(qn('a:srgbClr'), {'val': '000000'})
    alpha = srgbClr.makeelement(qn('a:alpha'), {'val': '20000'})
    srgbClr.append(alpha)
    outerShdw.append(srgbClr)
    effectLst.append(outerShdw)
    spPr.append(effectLst)
```

---

## 9. 中文字体处理

### 9.1 基本模式（推荐）

使用系统已安装的 Noto Sans SC / PingFang SC：

```python
def apply_chinese_font(paragraph, spec: dict):
    """对段落应用中文字体"""
    zh_font = spec['typography']['font_families']['zh'].split(',')[0].strip()
    en_font = spec['typography']['font_families']['en'].split(',')[0].strip()

    for run in paragraph.runs:
        run.font.name = en_font
        # 设置东亚字体
        rPr = run._r.get_or_add_rPr()
        ea = rPr.makeelement(qn('a:ea'), {'typeface': zh_font})
        rPr.append(ea)
```

### 9.2 字体子集模式（可选，减小文件体积）

当 PPTX 需要嵌入字体时（跨平台分发），使用 fonttools 生成子集：

```python
def extract_chinese_chars(semantic_json: dict) -> set:
    """从 slides_semantic.json 提取所有使用的中文字符"""
    import re
    chars = set()
    text = json.dumps(semantic_json, ensure_ascii=False)
    for ch in text:
        if '\u4e00' <= ch <= '\u9fff':
            chars.add(ch)
    return chars

def generate_font_subset(font_path: str, chars: set, output_path: str):
    """用 fonttools 生成字体子集"""
    import subprocess
    unicodes = ','.join(f'U+{ord(c):04X}' for c in chars)
    subprocess.run([
        'pyftsubset', font_path,
        f'--unicodes={unicodes}',
        '--layout-features=*',
        f'--output-file={output_path}'
    ], check=True)
```

---

## 10. 主生成流程

> ⛔ **CRITICAL IMPLEMENTATION NOTE**: 本 skill 的所有渲染逻辑已实现在 `skills/ppt-generator/bin/generate_pptx.py`（~1477 行，自包含，无外部 skill 模块依赖）。
> **必须直接运行该脚本**，不得使用 `python -m skills.ppt_generator.generate`（该模块不存在）。
> 本节的伪代码仅用于说明流程逻辑，**不可直接执行**。

```python
# ⚠️ 以下为流程说明伪代码，实际实现见 skills/ppt-generator/bin/generate_pptx.py
def generate_pptx(semantic_path, design_spec_path, output_path):
    # 1. 加载 slides_semantic.json + design_spec.json
    # 2. 初始化 GridSystem、Presentation
    # 3. 逐页渲染（14+ slide-type renderers, 8 component renderers）
    # 4. 保存 PPTX
    pass
```

### 10.1 CLI 使用

```bash
# ⛔ 唯一正确的命令 — 必须使用 skills/ppt-generator/bin/generate_pptx.py
python3 skills/ppt-generator/bin/generate_pptx.py \
  --semantic output/MFT_slides_semantic.json \
  --design output/MFT_design_spec.json \
  --output docs/presentations/mft-20260206/MFT.pptx
```

> ❌ **已废弃 — 绝对不要使用以下命令**:
> - `python -m skills.ppt_generator.generate ...` （模块不存在，会报 ModuleNotFoundError）
> - `python3 scripts/generate_pptx_ci.py ...` （168 行，功能不完整）
> - `from skills.ppt_generator import generate_pptx` （phantom import，不存在）

### 10.2 参数说明

| 参数 | 说明 | 示例 |
|------|------|------|
| `--semantic` | slides_semantic.json 路径 | `output/MFT_slides_semantic.json` |
| `--design` | design_spec.json 路径 | `output/MFT_design_spec.json` |
| `--output` | 输出 PPTX 路径 | `docs/presentations/mft-20260206/MFT.pptx` |

> ⚠️ 注意：参数名是 `--design`（不是 `--design-spec`）

---

## 11. QA 验证

### 11.1 6 阶段 QA Pipeline

| 阶段 | 检查项 | 权重 | 自动修复 |
|------|--------|------|---------|
| 1. Schema 验证 | semantic JSON 结构完整性 | 10% | ❌ 拒绝 |
| 2. 内容质量 | bullets ≤ 5/页, speaker notes ≥ 80% | 25% | ✅ 拆分 |
| 3. 设计合规 | 颜色/字体/间距均来自 token | 20% | ✅ 替换 |
| 4. 无障碍 | WCAG AA 对比度, 最小字号 | 25% | ✅ 升级 |
| 5. 性能预算 | PPTX ≤ 50MB, 图片 ≤ 5MB | 10% | ✅ 压缩 |
| 6. 技术验证 | PPTX 完整性, 布局边界 | 10% | ❌ 重建 |

### 11.2 实现

```python
def run_qa(pptx_path: str, semantic: dict, spec: dict) -> dict:
    """运行 QA pipeline，返回 qa_report"""
    report = {
        'overall_score': 0,
        'quality_gate_status': 'PENDING',
        'issues': [],
        'stage_results': {}
    }

    # Stage 1: Schema
    s1 = validate_schema(semantic, spec)
    report['stage_results']['schema'] = s1

    # Stage 2: Content quality
    s2 = validate_content(semantic)
    report['stage_results']['content'] = s2

    # Stage 3: Design compliance (需要已生成的 PPTX)
    s3 = validate_design_compliance(pptx_path, spec)
    report['stage_results']['design'] = s3

    # Stage 4: Accessibility
    s4 = validate_accessibility(pptx_path, spec)
    report['stage_results']['accessibility'] = s4

    # Stage 5: Performance
    s5 = validate_performance(pptx_path)
    report['stage_results']['performance'] = s5

    # Stage 6: Technical
    s6 = validate_technical(pptx_path, semantic)
    report['stage_results']['technical'] = s6

    # 计算总分
    weights = [0.10, 0.25, 0.20, 0.25, 0.10, 0.10]
    stages = [s1, s2, s3, s4, s5, s6]
    report['overall_score'] = sum(
        s.get('score', 0) * w for s, w in zip(stages, weights)
    )

    critical = sum(1 for i in report['issues'] if i.get('severity') == 'critical')
    report['quality_gate_status'] = 'PASS' if (
        report['overall_score'] >= 70 and critical == 0
    ) else 'FAIL'

    return report

def validate_content(semantic: dict) -> dict:
    """Stage 2: 内容质量检查"""
    issues = []
    slides = semantic.get('slides', [])

    for s in slides:
        bullets = s.get('content', [])
        if len(bullets) > 5:
            issues.append({
                'slide_id': s['slide_id'],
                'severity': 'major',
                'issue': f'Bullet count {len(bullets)} > 5',
                'auto_fixable': True
            })

        notes = s.get('speaker_notes', {})
        if not notes or not notes.get('summary'):
            issues.append({
                'slide_id': s['slide_id'],
                'severity': 'minor',
                'issue': 'Missing speaker notes',
                'auto_fixable': False
            })

    coverage = sum(1 for s in slides if s.get('speaker_notes', {}).get('summary')) / max(len(slides), 1)
    score = 100 if coverage >= 0.8 and not issues else max(60, 100 - len(issues) * 10)

    return {'score': score, 'issues': issues, 'notes_coverage': coverage}
```

---

## 12. 完整示例 — 运行预构建脚本

> ⛔ **不要编写新的生成脚本。** 直接运行已有的 `skills/ppt-generator/bin/generate_pptx.py`。

```bash
# 完整示例：生成 MFT PPTX
python3 skills/ppt-generator/bin/generate_pptx.py \
  --semantic output/MFT_slides_semantic.json \
  --design output/MFT_design_spec.json \
  --output docs/presentations/mft-20260206/MFT.pptx

# 脚本特性（~1600 行，自包含）：
# - 14+ slide-type renderers (title, section_divider, bullet-list, comparison, etc.)
# - 8 component renderers (kpis, comparison_items, decisions, risks, etc.)
# - GridSystem 12-column layout
# - Material Design token system
# - Speaker notes, bottom bar, title bar
# - chart_config → data table rendering
# - mermaid_code → styled placeholder rendering
# - 自适应高度布局：组件根据可用空间自动扩展，避免大面积留白
#   - render_comparison_items(avail_h=) → 卡片高度自适应
#   - render_decisions(avail_h=) → 决策卡片铺满可用区域
#   - render_chart_table → 行高自适应可用高度
#   - content_fill 策略：读取 design_spec.slide_type_layouts[type].content_fill
#     "expand" = 组件扩展填满 | "center" = 居中 | "top-align" = 固定顶部
```

如果脚本缺少某种 slide_type 的渲染器，**编辑 `skills/ppt-generator/bin/generate_pptx.py` 添加**，不要创建新脚本。

---

## 13. 约束与边界

### MUST（必须）

- ✅ 所有颜色来自 `design_spec.color_system` token
- ✅ 所有字号来自 `design_spec.typography.type_scale`
- ✅ 所有间距基于 `design_spec.spacing_system` 或 `grid_system`
- ✅ 演讲者笔记逐字保留（不重写、不删减）
- ✅ `slides_semantic.json` 中每个 slide 都必须渲染
- ✅ 中文正文 ≥ 18pt，中文标题 ≥ 28pt
- ✅ 行高：中文 ≥ 1.5，英文 ≥ 1.3

### MUST NOT（禁止）

- ❌ 硬编码颜色值（必须从 token 获取）
- ❌ 硬编码位置坐标（必须从网格系统计算）
- ❌ 修改或重写 slide 内容（content / speaker_notes)
- ❌ 添加 semantic JSON 中不存在的 slide
- ❌ 做设计决策（颜色搭配、布局选择由 design_spec 定义）
- ❌ 自行生成 alt text 或 diagram 内容
