To: ppt-visual-designer
From: ppt-content-planner
Session: MFT-20260210
Visual Style: mckinsey

Status: Content self-check PASSED (see `content_qa_report.json` — all MO-0..MO-12 checks PASS).

Deliverables requested (priority order):
1. Critical visuals (high priority, deliver first, editable SVG + PDF):
   - Slide 2 (Decision matrix / quick-vote card) — priority: critical — cognitive_intent: support-concise-decision — source: `slides_semantic.json` `components.decisions` (visual: summary matrix). Note: DO NOT duplicate decisions data; use components as single source.
   - Slide 3 (Executive summary comparison) — priority: high — cognitive_intent: quick-consensus — placeholder: `chart_config` in `slides_semantic.json`.
   - Slide 5 (Market segmentation ranges) — priority: high — cognitive_intent: compare-market-opportunity — placeholder: `chart_config`.
   - Slide 11 (Winding comparison) — priority: high — cognitive_intent: technical-comparison — placeholder: `chart_config` (qualitative labels).
   - Slide 13 (Cooling options comparison) — priority: critical — cognitive_intent: compare-thermal-options — placeholder: `chart_config`.
   - Slide 17/18 (Demonstration data flow + KPI dashboard) — priority: critical — cognitive_intent: evidence-chain + monitoring-dashboard — mermaid + dashboard spec provided.
   - Slide 25 (Risk matrix) — priority: critical — cognitive_intent: highlight-high-impact-risks — mermaid placeholder available.
   - Slide 28/29 (Roadmap/Gantt) — priority: high — cognitive_intent: show timeline & critical path — mermaid gantt placeholders.

2. Visual requirements & constraints:
   - Follow `mckinsey` visual hierarchy: assertion-first headline, single visual per slide where possible, restrained color palette.
   - Ensure all visuals include `cognitive_intent` and `priority` annotation in the asset metadata.
   - Use the `placeholder_data` in `slides_semantic.json` as the single source of truth. If numbers are missing, flag for content-team—do not invent numeric values.
   - For comparison cards, each card must show >=3 attributes (MO-10).

3. Output formats:
   - Provide editable SVGs + production PNGs (1920x1080) and a short design_spec.md listing fonts, colors, spacing and any data-transform rules used.

4. Timeline:
   - First-pass previews for critical visuals in 48h; full visual package in 72h.

Attachments & references:
- `slides_semantic.json` — authoritative semantic structure and placeholder_data
- `content_qa_report.json` — self-check results
- `slides.md` — presenter-facing slide outline & speaker notes

If any placeholder data is insufficient for a visual, please open an issue in the session dir (`docs/presentations/MFT-20260210/`) and tag `ppt-content-planner` for quick clarifications. No escalation to Creative Director needed unless visual constraints cannot be resolved within 48h.

Thank you — looking forward to the visual previews.
