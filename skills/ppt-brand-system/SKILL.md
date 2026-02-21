---
name: ppt-brand-system
description: Multi-brand design system for PPT HTML slides - single source of truth for 5 consulting brands (KPMG/McKinsey/BCG/Bain/Deloitte), semantic color palette, border tokens, and brand switching implementation.
metadata: 
  - version: 1.0.0
  - author: ppt-html-generator
---

# PPT Brand System

## Overview

This skill serves as the single source of truth for multi-brand design system, containing color, font, layout characteristics, semantic color system, and border tokens for 5 consulting brands.

## When to Use This Skill

- Get brand colors and font definitions
- Implement CSS variables and brand switching
- Query semantic color mappings
- Apply border tokens
- Ensure design consistency

## Supported Brands

| Brand | Primary Color | Sidebar | Characteristics |
|-------|--------------|---------|-----------------|
| KPMG (Default) | `#00338D` | Left | Clean professional, data-driven |
| McKinsey | `#00A3A0` | None (full-width) | Minimalist, authoritative |
| BCG | `#009A44` | None (top navigation) | Professional rigorous, green-based |
| Bain | `#DC291E` | Right | Red-focused, results-oriented |
| Deloitte | `#86BC25` | None | Green, modern, tech-savvy |

## Brand Detailed Definitions

### KPMG (Default)

- **Primary Color**: `#00338D` (KPMG Blue)
- **Secondary Color**: `#0091DA` (Light Blue)
- **Accent Color**: `#483698` (Purple)
- **Fonts**: Georgia, Arial, Noto Sans SC
- **Layout**: Left brand sidebar

### McKinsey

- **Primary Color**: `#00A3A0` (McKinsey Teal)
- **Secondary Color**: `#1A1A1A` (Dark Gray)
- **Accent Color**: `#FF6B35` (Orange)
- **Fonts**: Helvetica Neue, PingFang SC
- **Layout**: No sidebar, full-width design

### BCG

- **Primary Color**: `#009A44` (BCG Green) - **C1 Fix**
- **Secondary Color**: `#1D428A` (Navy Blue)
- **Accent Color**: `#FF671F` (Orange)
- **Fonts**: Arial, Microsoft YaHei
- **Layout**: Top navigation bar

> **C1 Fix**: BCG primary color changed from `#00A3A0` to `#009A44` to differentiate from McKinsey.

### Bain

- **Primary Color**: `#DC291E` (Bain Red)
- **Secondary Color**: `#1A1A1A` (Dark Gray)
- **Accent Color**: `#F5A623` (Yellow)
- **Fonts**: Gotham, Arial, Source Han Sans SC
- **Layout**: Right insight sidebar

### Deloitte

- **Primary Color**: `#86BC25` (Deloitte Green)
- **Secondary Color**: `#0033A0` (Blue)
- **Accent Color**: `#FF671F` (Orange)
- **Fonts**: Arial, Noto Sans SC
- **Layout**: Rounded corners, gradient background

## Brand Switching Mechanism

### CSS Class Pattern

```css
/* Add brand-{brand_id} class on <body> */
<body class="brand-kpmg">
```

### CSS Variables

```css
.brand-kpmg {
  --brand-primary: #00338D;
  --brand-secondary: #0091DA;
  --brand-accent: #483698;
}
```

### Switching Rules

1. Production `slide-*.html` files should not contain brand switching controls
2. Debug controls only retained in `presentation.html`
3. After switching brand, must delay 50ms before calling `chart.resize()`

## Semantic Color System

| Color | Meaning | Examples | Tailwind |
|-------|---------|----------|----------|
| red | Risk/Block/Negative variance | Delays, Gaps, Failed items | `red` |
| amber | Warning/Transition/Pending | Fluctuations, Threshold ranges, Watch items | `amber` |
| sky | Information/Progress/Neutral state | Progress, Fact descriptions, Status updates | `sky` |
| emerald | Achieved/Positive results | Completed, Benefits, Improvements | `emerald` |
| indigo | Phase/Architecture/Milestone | Structural layers and phase anchors | `indigo` |

### Semantic Rules

- Card titles, conclusion keywords, and numerical direction must align with color semantics
- Forbidden: "Negative content using emerald" or "Achieved content using red" mismatches
- Semantic expression diversity: left border, top border, light background, colored icons, shadows, gradients, badges
- Forbidden: Mechanically using only single left border or top border throughout

## Border System

### Semantic Emphasis Options

- `border-l-2`: Left border
- `border-t-4`: Top border
- `bg-tint`: Light background
- `icon-accent`: Icon emphasis
- `gradient-bg`: Gradient background
- `badge-accent`: Badge emphasis

### Structural Default

Basic card and container borders uniformly use `border border-slate-200`

### Border Rules

1. **Same Group Unified**: Cards in same group must use same emphasis paradigm
2. **Structural Consistency**: Base borders should not drift to heavier strokes
3. **Exception Protocol**: Using non-default width requires specific conditions
4. **Maximum Intensity**: `border-l-4` only for cover decoration or high-risk alert cards
5. **Intra-group Consistency**: Process, roadmap, KPI group cards must use same border width

## Design Tokens

### Spacing

- `base`: 1rem (16px)
- `content_padding`: px-16 py-8
- `card_padding`: p-4 or p-6

### Runtime Layout Tokens

| Token | Compact | Default | Relaxed |
|-------|---------|---------|---------|
| card_padding | 12px | 16px | 24px |
| chart_height | 180px | 220px | 260px |

### Breakpoints

- `sm`: 640px
- `md`: 768px
- `lg`: 1024px
- `xl`: 1280px

## Accessibility

- Color contrast complies with WCAG 2.1 AA standard
- Keyboard navigation support
- Screen reader friendly
- Automatic text color inversion (based on background brightness)

## Resource Files

- `assets/brands.yml`: Complete brand definitions, semantic colors, border tokens
- `examples/examples.md`: CSS/JS/HTML implementation examples
