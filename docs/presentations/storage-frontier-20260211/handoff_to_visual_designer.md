# Handoff to `ppt-visual-designer`

Status: **Ready for visual design** ✅

Summary:
- Content planning completed and self-checked (MO-0..MO-12) — all checks passed (see `content_qa_report.json`).
- Visual style: **BCG** (see `design_spec.json` for tokens & accessibility constraints).

Files for design:
- `slides_semantic.json` — 25 slides with components, visuals placeholder_data (charts/diagrams)
- `slides.md` — readable slide outline with speaker notes
- `reports/figures/*` and `reports/data/*` — source charts & CSVs
- `content_qa_report.json` — QA checklist & pass status

Deliverables requested from visual-designer (priority order):
1. **Sample cover** rendered in BCG tokens + one data-heavy content page mockup (market chart) + one comparison page mockup (vendor cards). Export SVG + PNG (>=150 DPI) + PPTX master.
2. Apply BCG color palette to charts in `slides_semantic.json` visual.placeholder_data (use `chart_color_palette` from `design_spec.json`).
3. Ensure text & chart contrast meets `accessibility.min_color_contrast` (4.5:1).
4. Produce a `.pptx` template with master slide and two content layouts (data-heavy, comparison/cards).

Acceptance criteria before handoff back to content-planner:
- Sample mockups comply with `design_spec.json` tokens and accessibility rules.
- Charts rendered from `placeholder_data` (no value changes) and exported as SVG/PNG.
- Provide slide exports and an updated `slides_semantic.json` where `visual.placeholder_data` includes `rendered_svg_path` and `rendered_png_path` for each chart.

Escalation policy:
- If any MO-0..MO-12 checks fail during visual integration, **escalate to the stakeholder (user)** with details.
- Otherwise, deliver mockups and request content-visual integration QA.

Notes:
- Visual designer should *not* modify numeric data; any visual design questions should be logged as issues in this session directory.

Contact: `ppt-content-planner` (auto-assigned).