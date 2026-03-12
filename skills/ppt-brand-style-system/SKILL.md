---
name: ppt-brand-style-system
description: Brand-style design system for PPT HTML slides - single source of truth for 6 reusable style profiles, including consulting-style and editorial-briefing styles.
metadata: 
  - version: 1.0.0
  - author: ppt-html-generator
---

# PPT Brand-Style System

## Overview

This skill serves as the single source of truth for a brand-style design system, containing color, font, layout characteristics, semantic color system, and border tokens for reusable PPT style profiles.

The repository path is `ppt-brand-style-system`. Its intended meaning is a **brand-style system**: a library of brand-inspired or report-style visual profiles rather than a registry of literal corporate brands.

`brand_id` is retained as the technical identifier for compatibility, but in practice it means a **style profile**, not necessarily a real-world corporate identity.

## When to Use This Skill

- Get style colors and font definitions
- Implement CSS variables and style switching
- Query semantic color mappings
- Resolve component semantic payloads into style-compatible class payloads
- Apply border tokens
- Ensure design consistency

## Supported Style Profiles

| Style Profile | Primary Color | Sidebar | Characteristics |
|--------------|--------------|---------|-----------------|
| KPMG (Default) | `#00338D` | Left | Clean professional, data-driven |
| McKinsey | `#00A3A0` | None (full-width) | Minimalist, authoritative |
| BCG | `#009A44` | None (top navigation) | Professional rigorous, green-based |
| Bain | `#DC291E` | Right | Red-focused, results-oriented |
| Deloitte | `#86BC25` | None | Green, modern, tech-savvy |
| Editorial Briefing | `#0F172A` | None (minimalist) | Weight-contrast typography, deep navy, restrained background, editorial briefing feel |

## Style Profile Definitions

### KPMG (Default Style)

- **Primary Color**: `#00338D` (KPMG Blue)
- **Secondary Color**: `#0091DA` (Light Blue)
- **Accent Color**: `#483698` (Purple)
- **Fonts**: Georgia, Arial, Noto Sans SC
- **Layout**: Left style sidebar

### McKinsey Style

- **Primary Color**: `#00A3A0` (McKinsey Teal)
- **Secondary Color**: `#1A1A1A` (Dark Gray)
- **Accent Color**: `#FF6B35` (Orange)
- **Fonts**: Helvetica Neue, PingFang SC
- **Layout**: No sidebar, full-width design

### BCG Style

- **Primary Color**: `#009A44` (BCG Green) - **C1 Fix**
- **Secondary Color**: `#1D428A` (Navy Blue)
- **Accent Color**: `#FF671F` (Orange)
- **Fonts**: Arial, Microsoft YaHei
- **Layout**: Top navigation bar

> **C1 Fix**: BCG primary color changed from `#00A3A0` to `#009A44` to differentiate from McKinsey.

### Bain Style

- **Primary Color**: `#DC291E` (Bain Red)
- **Secondary Color**: `#1A1A1A` (Dark Gray)
- **Accent Color**: `#F5A623` (Yellow)
- **Fonts**: Gotham, Arial, Source Han Sans SC
- **Layout**: Right insight sidebar

### Deloitte Style

- **Primary Color**: `#86BC25` (Deloitte Green)
- **Secondary Color**: `#0033A0` (Blue)
- **Accent Color**: `#FF671F` (Orange)
- **Fonts**: Arial, Noto Sans SC
- **Layout**: Rounded corners, gradient background

### Editorial Briefing Style

- **Primary Color**: `#0F172A` (Deep Navy)
- **Secondary Color**: `#64748B` (Muted Slate)
- **Accent Color**: `#E2E8F0` (Structural Lines)
- **Fonts**: Sans-based weight-contrast system (Inter / Noto Sans SC or equivalent)
- **Layout**:
  - **No Sidebars/Navbars**: Absolute canvas cleanliness.
  - **Restrained Background**: White or very light neutral background, no heavy style blocks.
  - **Typography Driven**: Hierarchy established by weight contrast and scale, not by saturated color blocks.
  - **Structural Elements**: Use thin vertical/horizontal lines for separation.
  - **Use Case**: Board briefings, strategy narratives, geopolitics or macro decks where editorial typography and restraint matter more than literal corporate identity.

`Strategic Report` is treated as a use case, not the canonical style name. The visual language here is closer to an editorial briefing or editorial-report cover system than to a generic strategy-report category.

## Style Profile Switching Mechanism

Preferred API naming should now use `style profile` semantics such as `getStyleProfileColors()` and `switchStyleProfile()`. Legacy names like `getBrandColors()` and `switchBrand()` may be retained only as backward-compatible aliases.

### CSS Class Pattern

```css
/* Add brand-{brand_id} class on <body>. brand_id is a style-profile id. */
<body class="brand-kpmg">
```

### CSS Variables

```css
.brand-kpmg {
  --brand-primary: #00338D;
  --brand-secondary: #0091DA;
  --brand-accent: #483698;
}

.brand-strategic,
.brand-strategic-report {
  --brand-primary: #0F172A;   /* Slate-900: Deep Navy/Black for Headlines */
  --brand-secondary: #64748B; /* Slate-500: Muted for Subtitles */
  --brand-accent: #E2E8F0;    /* Slate-200: Subtle Borders/Lines */
  --font-heading: 'Inter', sans-serif;
  --font-body: 'Noto Sans SC', sans-serif; 
}
```

### Switching Rules

1. Production `slide-*.html` files should not contain style-switching controls
2. Debug controls only retained in `presentation.html`
3. After switching style profile, must delay 50ms before calling `chart.resize()`
4. New examples and new runtime code should prefer style-profile API names; `brand-*` naming is a compatibility contract, not the recommended public vocabulary

## Typography System (New in v1.1)

### Visual Hierarchy Principles

Standard business reports often rely on size alone for hierarchy (H1 > H2). The `Editorial Briefing` style profile introduces **Weight Contrast** as a primary tool:

- **Display Title**: `text-6xl font-black tracking-tight leading-none` (e.g., "Geopolitics 2026")
- **Subtitle / Essence**: `text-4xl font-light tracking-wide text-slate-500` (e.g., "New Order and Strategic Inflection")
- **Eyebrow / Meta**: `text-xm font-bold tracking-widest uppercase text-slate-400` (e.g., "CONFIDENTIAL BOARD BRIEFING")

### Font Pairings (`.brand-strategic` / `.brand-strategic-report`)

- **Primary Headings**: `font-weight: 900` (Inter Black / Noto Sans Black)
- **Secondary Headings**: `font-weight: 300` (Inter Light / Noto Sans Light)
- **Body Text**: `font-weight: 400` with `leading-relaxed` for readability.

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

## Component Semantic Resolution

If a component example exposes a `semantic_payload`, resolve it through `assets/component_semantic_mappings.yml` before relying on raw fallback classes.

Use this order:

1. Determine the style profile (`brand_id` compatibility id).
2. Read the component family contract in `component_semantic_mappings.yml` to confirm the correct `slot_set`, allowed semantic fields, and resolution order.
3. Read component-level semantic roles such as `emphasis_role`, `surface_role`, `value_role`, or `timeline_role`.
4. Merge neutral tokens first (`primary_text`, `secondary_text`, `muted_text`, `neutral_structure`).
5. Merge emphasis-role tokens second (`primary`, `info`, `positive`, `warning`, `critical`, `neutral`).
6. Only then apply page-local exceptions.

This keeps component instances aligned with the style profile while avoiding hardcoded dark defaults inside each component payload.

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
4. **Maximum Intensity**: `border-l-4` only for cover decoration or SINGLE high-risk alert card. Default insight cards should use `border-l-2` or icon accents.
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

- `assets/brands.yml`: Complete style-profile definitions, semantic colors, border tokens
- `assets/component_semantic_mappings.yml`: Semantic role -> component-ready class payload resolver, including component family contracts and slot-set bindings
- `examples/examples.md`: CSS/JS/HTML implementation examples
