# ppt-design-system Skill

## Purpose

Provides a comprehensive Material Design 3 design system for PPT creation. Used by `ppt-visual-designer` to ensure consistent visual language across slides with accessibility, performance, and quality standards.

## Core Features

- **Design tokens**: Material Design 3 color palette, typography scale, spacing system
- **Component library**: Text boxes, charts, diagrams, containers with accessibility guidelines
- **Layout system**: 3 master layouts (title, content, closing) with safe zones and grid
- **Chart encoding**: Data-to-visual mappings for 24+ chart types
- **Performance budgets**: Object limits, file size targets, rendering optimization
- **Quality checklist**: 15-item visual quality validation before handoff

## Usage

Agent references this skill when:
1. Generating Material Design-compliant slide JSON
2. Selecting colors, fonts, and spacing tokens
3. Encoding data into charts with accessible visual mappings
4. Validating output against performance budgets and quality standards
5. Creating reusable component instances with consistent styling

---

## Core Design Tokens

### 1.1 Color System (Material Design 3 Semantic Colors)

| Role | Purpose | Example |
|---|---|---|
| `primary` | Main brand color | #2563EB (trust/professionalism) |
| `on_primary` | Text on primary | #FFFFFF |
| `primary_container` | Tinted backgrounds | #D6E3FF |
| `on_primary_container` | Text on tinted BG | #001B3D |
| `secondary` | Supporting color | #10B981 (success/positive) |
| `on_secondary` | Text on secondary | #FFFFFF |
| `secondary_container` | Secondary tinted BG | #C7F4E2 |
| `tertiary` | Accent/warning | #F59E0B (highlights) |
| `on_tertiary` | Text on tertiary | #FFFFFF |
| `surface` | Background | #FFFFFF |
| `on_surface` | Body text | #1A1C1E |
| `surface_variant` | Alt background | #E1E2EC |
| `surface_dim` | Muted background | #D3D4DE |
| `surface_container_low` | Subtle container | #ECEDF6 |
| `on_surface_variant` | Secondary text | #44464F |
| `error` | Error/risk | #EF4444 |
| `on_error` | Text on error | #FFFFFF |
| `outline` | Borders | #74777F |
| `shadow` | Shadow color | #000000 |

### 1.2 Typography System (Material Type Scale)

| Scale | Size | Weight | Line Height | Use |
|---|---|---|---|---|
| Display Large | 96pt | 700 | 1.1 | Hero titles, full-screen impact |
| Headline Large | 60pt | 700 | 1.2 | Section dividers |
| Headline Medium | 24-32pt | 700 | 1.2 | Standard slide titles (**24pt for 7.5" slides**) |
| Title Large | 20-24pt | 600 | 1.3 | Sub-section headers |
| Body Large | 16-20pt | 400 | 1.5 | Main content (**16pt for 7.5" slides**) |
| Body Medium | 14-16pt | 400 | 1.5 | Supporting content |
| Label Large | 12-14pt | 600 | 1.4 | Data labels, captions |

**Font Families:**
- EN: Roboto, Arial, Helvetica, sans-serif
- ZH: Noto Sans SC, PingFang SC, Microsoft YaHei, sans-serif

**Explicit Sizes (design-spec.json `explicit_sizes`):**

```json
{
  "slide_title": 24, "slide_subtitle": 18, "body_text": 16, "body_text_zh": 18,
  "bullet_text": 16, "table_header": 14, "table_cell": 13, "label": 12,
  "section_label": 11, "kpi_value": 28, "kpi_label": 12, "callout_text": 15,
  "speaker_note": 12, "page_number": 9
}
```

> **Typography Blocker**: `explicit_sizes` MUST be present in `typography_system`. Missing → design_spec.json blocked. Every text run must have deterministic Pt(N).

**字体规格设计原则：**
- **标题栏文字**: slide 高度 3-4%（7.5" → 22-24pt）
- **正文文字**: slide 高度 2-2.5%（7.5" → 15-19pt），最小 14pt，推荐 16pt
- **中文文字**: 比英文大 1.2 倍（中文 16pt ≈ 英文 14pt 视觉等效）
- **行高**: 中文 1.5-1.6，英文 1.3-1.5
- **画布适配**: 字体大小关联 `slide_height_inches`；非标准 7.5" 按比例缩放

### 1.3 Spacing System (4pt Base Grid)

**Scale**: 4, 8, 12, 16, 24, 32, 48, 64, 96 pt

**推荐值（16:9 slides, 13.333" × 7.5"）：**

| Token | Value | Description |
|---|---|---|
| `slide_margin` | 40-48px | 水平边距（slide 宽度 3-4%） |
| `content_padding` | 20-24px | 内容内边距 |
| `title_bar_height` | 0.50-0.55" | slide 高度 7-7.3%，推荐 0.55" |
| `title_bar_height_narrow` | 0.40" | data-heavy slide |
| `content_top` | title_bar + 0.12" | 标题栏后留间距 |
| `bottom_bar_height` | 0.25" | 页码 + section 进度条；**≥0.20"** |
| `bottom_margin` | 0.15-0.20" | 底部留白 |

**layout_zones (REQUIRED in design_spec.json, INCHES):**

```json
"layout_zones": {
  "title_bar_height_default": 0.55,
  "title_bar_height_narrow": 0.40,
  "bottom_bar_height": 0.25,
  "content_margin_top": 0.12,
  "content_bottom_margin": 0.20,
  "progress_bar": true
}
```

> **重要**: 值为英寸，渲染脚本直接读取，禁止在脚本中硬编码任何 bar 高度。

**空间利用率目标**:
- 内容可用高度：≥80%
- 水平留白：≤10%
- 避免过度留白导致内容区域 <70% 可用空间

### 1.4 Elevation System

| Level | Shadow | Use |
|---|---|---|
| 0 | none | Flat backgrounds |
| 1 | 0 1px 3px rgba(0,0,0,0.12) | Cards, containers |
| 2 | 0 2px 6px rgba(0,0,0,0.15) | Floating elements |
| 3 | 0 4px 12px rgba(0,0,0,0.18) | Modals, emphasis |

### 1.5 Shape System

| Size | Corner Radius | Use |
|---|---|---|
| Small | 4pt | Chips, small tags |
| Medium | 8pt | Cards, containers |
| Large | 16pt | Large cards, panels |
| Extra Large | 24pt | Hero containers |

### 1.6 Grid System

- 12-column grid, 24pt gutter, 48pt margin
- Min whitespace ratio: 0.3

**Responsive Breakpoints:**

| Format | Resolution | Aspect | Adaptation |
|---|---|---|---|
| Projector Standard | 1920×1080 | 16:9 | Default typography |
| Projector Large | 2560×1440 | 16:9 | Scale ×1.1 |
| Classic | 1024×768 | 4:3 | Single-column, 8 columns |
| Print Handout | A4 landscape | 1.414:1 | Display 48pt / Headline 24pt / Body 12pt |

---

## Material Component Library

### 2.1 Cards

| Property | Value |
|---|---|
| Padding | 24pt |
| Corner radius | 8pt (medium) |
| Elevation | Level 1 |
| Background | surface |
| **Use cases** | Bullet lists, data groups, key takeaways |

### 2.2 Callouts

| Property | Value |
|---|---|
| Border left | 4px solid primary |
| Background | primary_container (tinted) |
| Padding | 16px 24px |
| **Use cases** | Key insights, warnings, quotes |

### 2.3 Data Tables

| Property | Value |
|---|---|
| Header weight | 600 |
| Header color | on_surface_variant |
| Row height | 48pt |
| Number alignment | Right |
| Text alignment | Left |
| Zebra striping | surface_variant at 0.3 opacity |
| Dividers | outline at 0.12 opacity |

### 2.4 Chips

| Property | Value |
|---|---|
| Height | 32pt |
| Padding | 8px 16px |
| Corner radius | 16pt (fully rounded) |
| **Use cases** | Tags, filters, metadata |

---

## Layout Templates

### title-only
- **Purpose**: Hero slides, section dividers, impactful single messages
- **Typography**: Display Large (96pt) + heavy whitespace (≥50%)
- **Elevation**: Level 0 (flat)
- No title bar — full-screen visual impact

### bullet-list
- **Purpose**: Main content slides with ≤5 points
- **Typography**: Headline Medium title (44pt) + Body Large content (20pt)
- **Component**: Card container with max 5 bullets

### two-column
- **Purpose**: Content + supporting visual (diagram, chart, image)
- **Split**: 40:60 or 50:50
- **Left**: Body Large text, **Right**: visual with elevation 1

### full-bleed
- **Purpose**: Emotional impact, storytelling moments
- **Design**: Background image + scrim overlay (rgba(0,0,0,0.4))
- **Typography**: Display Large with on-primary color for contrast

### data-heavy
- **Purpose**: Complex data presentation
- **Components**: Material Data Table + chart visualization

---

## Chart & Visual Encoding Guidelines

### 4.1 Cleveland & McGill Hierarchy

1. **Position** (most accurate): scatter plots, dot plots
2. **Length**: bar charts, column charts
3. **Angle**: pie charts (use sparingly)
4. **Area**: avoid unless necessary
5. **Volume**: avoid (least accurate)

### 4.2 Chart Type Selection

| Chart Type | Use Case | Key Constraints |
|---|---|---|
| Bar / Column | Categorical comparison | Y-axis starts at 0, primary for main series |
| Line | Temporal trends | Direct labeling preferred over legend |
| Scatter | Correlation (position rank 1) | ≥5 data points |
| Pie | Part-to-whole | ≤5 categories ONLY |
| Waterfall | Running totals, breakdowns | Loss decomposition, cost build-up |
| Tornado | Sensitivity analysis | Diverging horizontal bars, symmetric axis |
| Radar / Spider | Multi-attribute comparison | ≤8 axes |
| Sankey | Flow/allocation visualization | ≥2 stages, ≥3 flow paths, proportional width |
| Bubble | 3-variable comparison | Max 20 bubbles, area encoding |
| Treemap | Hierarchical proportions | 2-level max |
| Pareto | Frequency + cumulative % | 80/20 analysis, dual Y-axis |
| Funnel | Staged process conversion | Gradient fill |

### 4.3 Cognitive Intent → Design Token Mapping

| Intent | Emotional Tone | Design Token Direction |
|---|---|---|
| `compare` | analytical | Parallel layout, contrasting colors |
| `trend` | confidence/concern | Sequential palette, trend emphasis |
| `composition` | informative | Harmonious palette, proportional |
| `distribution` | analytical | Gradient palette, density mapping |
| `relationship` | explanatory | Graph layout, directional indicators |
| `inform` | neutral | Bold typography, dashboard |
| `persuade` | urgency/aspiration | Highlight preferred, red/green |

**Emotional Tone → Accent Mapping:**
- urgency → error accent
- confidence → primary accent
- analytical → neutral/surface
- aspirational → gradient

### 4.4 Design Requirements

- **Color encoding**: Material color system, max 5 colors per chart, colorblind-safe
- **Resolution**: ≥200 DPI for raster, SVG preferred for diagrams
- **Direct labeling**: Label Large (14pt bold) for data labels
- **Data honesty**: No chartjunk. Tufte's Data-Ink Ratio.

---

## Layout System Specification

### 5.1 Grid System Configuration

> **CRITICAL**: `grid_system` MUST include `slide_width_inches` and `slide_height_inches` (16:9 = 13.333 × 7.5). `slide_width_px` / `slide_height_px` are reference resolution only. Renderer uses inches, NOT px/96.

```json
{
  "layout_system": {
    "grid_columns": 12,
    "slide_width_px": 1920,
    "slide_height_px": 1080,
    "dpi": 144,
    "slide_width_inches": 13.333,
    "slide_height_inches": 7.5,
    "aspect_ratio": "16:9",
    "margin_horizontal": 80,
    "gutter": 24,
    "column_width": 124.67,
    "layouts": {
      "two-column-6040": {
        "description": "60% content (left) + 40% image (right)",
        "content_columns": [1, 7],
        "image_columns": [8, 12]
      },
      "two-column-5050": {
        "description": "50% content + 50% image (equal split)",
        "content_columns": [1, 6],
        "image_columns": [7, 12]
      },
      "bullets": {
        "description": "Full-width content (centered with margins)",
        "content_columns": [2, 11]
      },
      "title-slide": {
        "description": "Centered title with large margins",
        "content_columns": [3, 10]
      },
      "chart-focused": {
        "description": "Small annotation (left) + large chart (right)",
        "content_columns": [1, 3],
        "image_columns": [4, 12]
      }
    },
    "responsive": {
      "4:3": {
        "slide_width_px": 1024,
        "slide_height_px": 768,
        "grid_columns": 8,
        "margin_horizontal": 48
      }
    }
  }
}
```

### 5.2 Layout Selection Rules

| Condition | Layout |
|---|---|
| `slide_type == 'title'` | title-slide |
| `bullet-list AND requires_diagram == true` | two-column-6040 |
| `slide_type == 'chart'` | chart-focused |
| `bullet-list AND requires_diagram == false` | bullets |
| `slide_type == 'section-divider'` | section-divider (custom) |

### 5.3 Column Position Calculation

```python
content_width_px = slide_width_px - 2 * margin_horizontal
total_gutter_px = gutter * (grid_columns - 1)
col_width_px = (content_width_px - total_gutter_px) / grid_columns

start_col, end_col = [1, 7]
left_px = margin_horizontal + (start_col - 1) * (col_width_px + gutter)
width_px = (end_col - start_col + 1) * col_width_px + (end_col - start_col) * gutter
```

---

## Performance Budgets

| Category | Budget |
|---|---|
| Total PPTX | ≤50MB |
| Per image | ≤5MB |
| Per font family | ≤500KB (subsetted) |
| Image: PNG | 8-bit simple / 24-bit photos |
| Image: JPEG | 85% quality |
| Image: SVG | Preferred for icons & diagrams |
| Image DPI | ≥200 (min), 300 (hero) |
| Font subset | EN A-Za-z0-9 + ZH 常用 3500 字 |
| Font fallback | system-ui, sans-serif |
| Animation FPS | 60fps |
| Animation properties | transform, opacity only |
| Animation duration | 200-400ms |
| Forbidden animation | width, height, color, background-color |

---

## Testing Strategy

### 7.1 Visual Regression
- Baseline slides: 1, 6, 8, 12 (hero, complex data, diagrams)
- Diff tolerance: 0.05%
- Validate: layout integrity, color accuracy, typography rendering

### 7.2 Accessibility
- **Automated**: WCAG 2.1 AA minimum, AAA for contrast
- **Manual**: VoiceOver (macOS), NVDA (Windows) for alt text
- Checks: contrast ratios, colorblind simulation, text readability

### 7.3 Cross-Format
- Formats: 16:9 (1920×1080), 4:3 (1024×768), PDF, A4 print
- Acceptance: 95% visual fidelity across all formats
- Validate: no clipping, font rendering, color accuracy, image quality

### 7.4 Component Consistency
- Color: only design system colors (no custom hex)
- Typography: only Material Type Scale (no custom sizes)
- Spacing: only scale values (no arbitrary padding)
- Elevation: only levels 0-3

---

## Design Review Checklist

### Design System Compliance
- [ ] Material Design 3 principles applied (color, typography, spacing, elevation)
- [ ] Design tokens consistent: color roles properly assigned
- [ ] Typography: Material Type Scale used
- [ ] Spacing: 4pt base grid maintained
- [ ] Elevation: Level 0-3 only
- [ ] Shape: Corner radius from shape system (4/8/16/24pt)

### Component Library
- [ ] All components follow Material patterns
- [ ] Component specifications complete (padding, colors, typography, elevation)
- [ ] Component usage documented per slide
- [ ] Components adapt to content with responsive rules

### Accessibility
- [ ] Color contrast ≥4.5:1 normal text, ≥3:1 large text (WCAG AA)
- [ ] Colorblind-safe palette tested
- [ ] i18n typography: Roboto/Noto Sans SC stacks
- [ ] Alt text specified for all diagrams and charts

### Visual Hierarchy & Layout
- [ ] Visual flow designed (F-pattern or appropriate)
- [ ] Hierarchy clear: Display > Headline > Body
- [ ] Grid system: 12-column specs provided
- [ ] Whitespace ≥30%
- [ ] Emphasis: scale, color, position used systematically

### Charts & Diagrams
- [ ] Chart types follow Cleveland Hierarchy
- [ ] All 3 taxonomy levels supported (Level 1: 10, Level 2: 8, Level 3: 6)
- [ ] Visual encodings specified
- [ ] No misleading scales (Y-axis at 0 for bar)
- [ ] Max 5 colors per chart

### Cognitive Intent
- [ ] `cognitive_intent` parsed from slides_semantic.json
- [ ] `primary_message` applied as chart title or annotation
- [ ] `emotional_tone` mapped to design tokens
- [ ] `attention_flow` mapped to layout reading order
- [ ] `key_contrast` applied as contrasting encodings
- [ ] `visual_hierarchy` mapped to elevation and typography

### Animation
- [ ] Material Motion: entrance (fade+slide), exit (fade), emphasis (scale)
- [ ] Duration 200-400ms
- [ ] Easing: ease-out / ease-in / ease-in-out
- [ ] Purposeful only, no decorative animations

### Design Rationale
- [ ] Design decisions documented
- [ ] Audience adaptation explained
- [ ] 2-3 alternative options provided for Creative Director

### Performance & Testing
- [ ] Performance budgets set (§6)
- [ ] Testing strategy defined (§7)
- [ ] Visual regression baselines identified
- [ ] Cross-format validation criteria

### Deliverable Completeness
- [ ] design-spec.json complete with all required sections
- [ ] Chart design specs ready for handoff
- [ ] Implementation notes for specialist

---

## Output Schema (design-spec.json)

Complete example with all required sections. Use as template — adapt values per project.

```json
{
  "meta": {
    "session_id": "YYYYMMDD-project",
    "timestamp": "ISO-8601",
    "design_system_version": "1.0.0",
    "base_system": "Material Design 3",
    "designer": "ppt-visual-designer",
    "status": "pending_creative_director_review"
  },

  "version_management": {
    "changelog": {
      "1.0.0": {
        "date": "YYYY-MM-DD",
        "changes": ["Initial design system"],
        "breaking_changes": []
      }
    },
    "deprecations": [],
    "next_version_plan": ""
  },

  "quality_metrics": {
    "target_token_compliance": 0.95,
    "target_accessibility_score": 1.0,
    "target_component_reuse_rate": 0.8,
    "actual_metrics": {
      "token_compliance": 1.0,
      "wcag_aa_compliance": 1.0,
      "component_reuse_rate": 0.85
    }
  },

  "design_philosophy": {
    "primary": "...",
    "principles": ["Material Design", "Tufte Data-Ink Ratio", "Cleveland Hierarchy"],
    "audience_persona": "...",
    "complexity_level": "medium"
  },

  "design_system": {
    "color_system": "... (see §1.1 for full token list)",
    "typography_system": {
      "type_scale": "... (see §1.2)",
      "font_families": {"en": "Roboto, Arial, ...", "zh": "Noto Sans SC, PingFang SC, ..."},
      "explicit_sizes": "... (see §1.2 — REQUIRED, blocks delivery if missing)"
    },
    "spacing_system": {"base_unit": 4, "scale": [4, 8, 12, 16, 24, 32, 48, 64, 96]},
    "elevation_system": "... (see §1.4)",
    "shape_system": {"corner_radius": {"small": 4, "medium": 8, "large": 16, "extra_large": 24}},
    "grid_system": {"columns": 12, "gutter": 24, "margin": 48, "min_whitespace_ratio": 0.3}
  },

  "layout_zones": "... (see §1.3 — REQUIRED, values in INCHES)",

  "layout_system": "... (see §5.1 — full grid config with slide_width/height_inches)",

  "responsive_design": {
    "breakpoints": "... (see §1.6 for all formats)"
  },

  "interaction_states": {
    "default": {"background": "surface", "elevation": "level_0"},
    "hover": {"background": "surface_variant", "elevation": "level_1", "transition": "200ms ease-out"},
    "focus": {"outline": "2px solid primary", "outline_offset": 2},
    "active": {"elevation": "level_0", "scale": 0.98},
    "disabled": {"opacity": 0.38}
  },

  "component_library": "... (see §2 for full component specs)",

  "slide_specifications": [
    {
      "slide_number": 1,
      "title": "...",
      "layout": "title-only",
      "components": [{"type": "hero_title", "typography": "display_large", "color": "primary"}],
      "visual_hierarchy": "single-focus",
      "whitespace_ratio": 0.6
    }
  ],

  "chart_design_specs": [
    {
      "slide_number": 8,
      "chart_type": "bar",
      "encoding": "length",
      "color_mapping": {"main_series": "primary"},
      "y_axis_start": 0,
      "direct_labeling": true,
      "label_typography": "label_large"
    }
  ],

  "performance_specs": "... (see §6)",
  "testing_strategy": "... (see §7)",
  "animation_specs": {
    "entrance": {"effect": "fade_slide_up", "duration": 300, "easing": "ease-out"},
    "emphasis": {"effect": "scale", "from": 1.0, "to": 1.05, "duration": 200}
  },

  "accessibility_specs": {
    "contrast_ratios": {"primary_on_surface": 7.2, "body_on_surface": 12.6, "wcag_level": "AAA"},
    "colorblind_safe": true,
    "alt_text_required": ["all diagrams", "all charts", "all images"]
  },

  "design_rationale": {
    "color_choice": "...",
    "typography_choice": "...",
    "layout_strategy": "...",
    "audience_adaptation": "..."
  },

  "component_usage_guidelines": "... (see §2-§4 for detailed guidelines)",

  "design_decisions_adr": [
    {
      "id": "ADR-001",
      "title": "...",
      "status": "accepted",
      "context": "...",
      "decision": "...",
      "rationale": "...",
      "alternatives_considered": [],
      "consequences": "..."
    }
  ],

  "design_debt": [],

  "alternative_options": [
    {"option": "A", "description": "...", "use_case": "..."},
    {"option": "B", "description": "...", "use_case": "..."}
  ]
}
```

> **Note**: In the actual output, expand all `"... (see §X)"` references with real values. The placeholders above show structure only.

### Optional Deliverables

| Deliverable | When |
|---|---|
| component-usage-guide.md | Recommended for all projects |
| design-decisions-adr/ | Recommended for complex projects |
| visual-prototype-specs.md | For complex designs |
| design-system-changelog.md | For version tracking |
