# Reveal design-spec (reveal-specific)

目的：定义 Reveal.js 专用的设计规范（design-spec.reveal.json），为 `reveal-builder` 提供可机器读取的视觉 tokens、布局模板与可访问性要求。

该文档包含：
- 字段说明（类型与语义）
- 映射指南（到 CSS 变量 / layout classes）
- 最小校验规则（what must be present for POC）

---

## 文件位置与命名
- 推荐位置： `docs/specs/design-spec.reveal.json`
- 校验文件： `docs/specs/design-spec.reveal.schema.json`

---

## 顶层结构（概览）

{
  "meta": {...},
  "tokens": {...},
  "layouts": {...},
  "chart_palette": {...},
  "animation_tokens": {...},
  "accessibility": {...},
  "fonts": {...}
}

---

## 字段详解

### meta
- session_id (string) — 可选，用于 trace
- design_system_version (string) — 语义化版本
- base_system (string) — e.g., "Material Design 3"
- designer (string)

### tokens
- primary (hex string) — 主色
- on_primary (hex)
- secondary (hex)
- tertiary (hex)
- surface (hex)
- on_surface (hex)
- surface_variant (hex)
- outline (hex)
- shadow (css color)
- Optional: semantic colors (error/warning/success/info)

用途：这些 token 会被注入到 `theme.css` 作为 CSS 变量，例如 `--color-primary: #2563EB;`。

### typography
- type_scale: map of named steps to {size: number, weight: number, line_height: number}
  - e.g., display_large (96), headline_large (60), headline_medium (44), body_large (20)

用途：用于 mapping 到 CSS `font-size` 与 heading classes.

### spacing
- base_unit: number (px)
- scale: array of numbers (px)

### layouts
- Map of layout_name → description & grid spec
- Each layout must provide `columns` (e.g., 12), `gutter` (px), `margin_horizontal` (px) and named regions with column indices, e.g.: 

```
"two-column-6040": {
  "description": "60% content (left) + 40% visual (right)",
  "content_columns": [1,7],
  "image_columns": [8,12]
}
```

用途：reveal-builder 使用此信息将 slide metadata -> CSS positioning（基于 predefined column widths）

### chart_palette
- Named colors for charts, colorblind-safe recommended
- Example: {"primary": "#2563EB", "accent1": "#10B981", ...}

要求：至少包含 `primary`、`accent1`、`accent2`。

### animation_tokens
- entrance: {effect: "fade_slide_up", duration_ms: 300, easing: "ease-out"}
- emphasis: {effect: "scale", from:1.0, to:1.05, duration_ms:200}

用途：render templates will add data attributes or classes (e.g. `data-entrance=