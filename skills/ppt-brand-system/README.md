# ppt-brand-system

Multi-brand design system for PPT HTML slides — single source of truth for 5 consulting brands.

## Overview

This skill centralizes all brand-related definitions that were previously scattered across the `ppt-html-generator` agent file in 3 separate locations (narrative spec, CSS variables, JS color map). It provides a **single YAML data source** (`brands.yml`) from which CSS and JS can be derived.

### Supported Brands

| Brand | Primary Color | Sidebar | Key Trait |
|-------|--------------|---------|-----------|
| KPMG (default) | `#00338D` | Left | 简洁专业、数据驱动 |
| McKinsey | `#00A3A0` | None (full-width) | 极简主义、权威感 |
| BCG | `#009A44` | None (top navbar) | 专业严谨、绿色主调 |
| Bain | `#DC291E` | Right | 红色为主、结果导向 |
| Deloitte | `#86BC25` | None | 绿色、现代、科技感 |

> **C1 fix**: BCG primary changed from `#00A3A0` → `#009A44` to differentiate from McKinsey.

## File Structure

```
skills/ppt-brand-system/
├── manifest.yml      # Skill metadata & integration config
├── brands.yml        # Single source of truth (brands + semantic colors + borders)
├── README.md         # This file
└── examples.yml      # CSS / JS / HTML implementation examples
```

## Data Schema (`brands.yml`)

| Section | Purpose |
|---------|---------|
| `brands.*` | 5 brand definitions: colors, fonts, font sizes, design traits, layout features |
| `switching` | Brand switching mechanism: CSS class pattern, CSS variables, rules |
| `design_tokens` | Common spacing, breakpoints, accessibility requirements |
| `semantic_colors` | 5 semantic color mappings (red/amber/sky/emerald/indigo) |
| `semantic_rules` | Color-content consistency rules |
| `border` | Border token defaults and uniformity contract |

## Usage

### In Agent Prompt

Reference with:
```
> 品牌色彩、字体、语义色、边框 token 定义见 `skills/ppt-brand-system/brands.yml`。
> CSS/JS 实现示例见 `skills/ppt-brand-system/examples.yml`。
```

### Generating CSS Variables

Read `brands.yml → brands.{brand_id}.colors` and map to CSS:
```css
.brand-{brand_id} {
  --brand-primary: {colors.primary};
  --brand-secondary: {colors.secondary};
  --brand-accent: {colors.accent};
  font-family: {fonts.stack};
}
```

### Semantic Color Lookup

Read `brands.yml → semantic_colors.{color}.tailwind` to get the Tailwind class prefix, then apply to card borders, badges, and labels.

## Integration

- **Owner**: `ppt-html-generator`
- **Consumers**: `ppt-content-planner`, `ppt-creative-director`
