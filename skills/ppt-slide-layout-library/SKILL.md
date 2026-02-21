---
name: ppt-slide-layout-library
description: Slide layout library for PPT HTML slides - 12 layout types with HTML templates, per-layout constraints, selection guide, dedup rules, Notion skeleton, and page-level budget/overflow constraints.
metadata: 
  - version: 1.0.0
  - author: ppt-layout-library
---

# PPT Slide Layout Library

## Overview

This skill provides 12 professional HTML slide layout templates, including complete HTML template code, layout constraints, selection guide, and page-level budget constraints.

## When to Use This Skill

- Create new HTML slides
- Select appropriate layout type for content
- Apply layout constraints and specifications
- Handle page vertical budget
- Ensure layout deduplication and visual balance

## General Constraints

Each page must contain "Title Area + Main Content Area + Insight Area + Footer Area" four-part structure (except cover/ending pages). Missing any part requires explanation and visual equivalent alternative.

## Layout Type Quick Reference

| # | Layout Type | YAML Key | Use Cases |
|---|-------------|----------|-----------|
| 1 | Cover | `cover` | Homepage, brand showcase |
| 2 | Data Chart | `data_chart` | Deep analysis of single data point |
| 3 | Dashboard Grid | `dashboard_grid` | Multi-dimensional KPI + trend chart combo |
| 4 | Side by Side | `side_by_side` | A/B testing, competitor comparison |
| 5 | Full Width | `full_width` | Strategic vision, trend display |
| 6 | Hybrid | `hybrid` | Multi-layer data display |
| 7 | Process | `process` | Workflow, step instructions |
| 8 | Dashboard | `dashboard` | Real-time monitoring, KPI tracking |
| 9 | Milestone Timeline | `milestone_timeline` | Annual event evolution, phase transitions |
| 10 | Pillar | `pillar` | Executive Summary, core pillars |
| 11 | Process Steps | `process_steps` | Timeline, evolution process |
| 12 | Comparison | `comparison` | Competitor analysis, multi-dimensional comparison |

## Layout Selection Guide

| Content Type | Recommended Layout | Scenario Description |
|--------------|-------------------|---------------------|
| Single insight | `data_chart` | Deep analysis of single data point |
| Comparative analysis | `side_by_side` | A/B testing, competitor comparison |
| Panoramic display | `full_width` | Strategic vision, overall overview |
| Complex analysis | `hybrid` | Multi-layer data display |
| Process explanation | `process` | Workflow, timeline display |
| Milestone narrative | `milestone_timeline` | Annual event evolution, key nodes |
| Monitoring report | `dashboard` | Real-time monitoring, KPI tracking |
| Comprehensive data view | `dashboard_grid` | Complex multi-dimensional data analysis |
| Core pillars | `pillar` | Executive Summary |
| Simple steps | `process_steps` | Simple timeline |
| Competitor comparison | `comparison` | Competitor analysis, solution comparison |

## Core Layout Details

### 1. Cover Layout (cover)

**Constraint Rules:**
- Homepage must use cover layout by default
- Body Suppression: Cover must not contain large analysis text (≤2 short sentences, total ≤80 characters)
- Hierarchy Structure: Must include eyebrow + main title + subtitle/English title + meta info four layers
- Visual Focus: Title and brand elements as focal point
- Footer Strategy: Simplified footer may be retained, analysis-type footnotes not allowed

### 2. Data Chart Layout (data_chart)

**Constraint Rules:**
- Chart Priority: Left chart area width must be ≥ 58% (w-7/12)
- Right Alignment: Right insight cards recommended 2-3, total height auto-fill

**Page Budget:**
- Maximum Vertical Budget: 582px
- Default Chart Height: 220px
- Maximum Right Cards: 3
- Max List Items Per Card: 5

### 3. Side by Side Layout (side_by_side)

**Constraint Rules:**
- Default Same Height: Both main chart containers must have consistent height
- Primary/Secondary Exception: Only when explicitly marked 'primary/secondary' chart, height difference ≤15% allowed
- Bottom Alignment: Side by side cards must be bottom-aligned
- Top/Bottom Whitespace Threshold: Within each card difference ≤24px
- Chart Area Ratio Floor: Chart container height occupies 70%-82% of card available height

**Page Budget:**
- Per Column Chart Height: 210px
- Maximum Bottom KPI Rows: 3

### 4. Dashboard Grid Layout (dashboard_grid)

**Constraint Rules:**
- Grid Alignment: Must strictly follow 12-column grid system
- Density Red Line: When cards ≥6 and each card has <30 characters, must downgrade to list layout
- Font Restraint: Except core KPI numbers, body text must not exceed text-base
- Whitespace Mandatory: 2x3 or 3x2 grid must use gap-6 or gap-8
- Chart-Text Ratio: At least 1/3 area must be data charts

### 5. Full Width Layout (full_width)

**Constraint Rules:**
- Main Chart Semantic Anchor: Full-width trend page main chart must have title or口径 short note
- KPI Card Minimum Fields: Metric name + time point + value + comparison baseline at least three items
- Full Width Fill Rate ≥ 86%
- Bottom Half Budget Range: Insight cards + KPI columns total height occupies 44%-52% of main content area

## Layout Deduplication Rules

1. Consecutive Same Structure Forbidden: Any two adjacent pages must not use same main layout type
2. Main Layout Determination: Determined by largest proportion layout module in main content area
3. Conflict Priority: Second page must switch to visual equivalent alternative
4. Cover and Ending Exception: Not参与 consecutive page same structure validation

## Page-Level Constraints

### Layout Balance Hard Constraints

- Left/Right Column Occupancy: Main chart column and narrative column content height both need ≥ 85%
- Whitespace Difference Control: Left/Right column visible whitespace rate difference must not exceed 10%
- Trigger Failure Priority: Increase main chart container height or add structured items

### Vertical Budget

- Single Page Height Budget: header + main + footer <= slide_height
- Main Area Safety Upper Limit: 1280×720 canvas main available height not exceeding 540px
- Footer Safety Zone: Main content area bottom reserve at least 8px safety margin

### Content Overflow Handling Strategy

1. Card Overflow Strategy Required: Long text cards must explicitly declare overflow-auto
2. Body Line Limit: Each card default ≤ 5 lines of body text
3. Overlimit Downgrade Order: First compress text → reduce auxiliary blocks → lower chart container height

## Page Budget Archive

### Global Configuration

- Canvas Size: 1280×720px
- Header Height: 80px
- Footer Height: 50px
- Main Padding: 80px
- Main Area External Available: 590px
- Main Area Internal Available: 510px

### Per-Layout Budget

| Layout | Max Vertical Budget | Default Chart Height | Max Cards |
|--------|---------------------|---------------------|-----------|
| data_chart | 582px | 220px | 3 |
| side_by_side | 582px | 210px | - |
| full_width | 582px | - | 4 KPI |
| hybrid | 582px | 230px | 3 |
| process | 582px | - | 5 steps |
| dashboard | 582px | 232px | 4 KPI |
| milestone_timeline | 582px | - | 6 cards |

## Dependencies

- **ppt-brand-system**: Brand colors/fonts/CSS variables
- **ppt-chart-engine**: Chart containers and rendering rules

## Resource Files

For detailed layout definitions, HTML templates, and complete constraints, refer to `assets/layouts.yml`.
