# Handoff to `ppt-content-planner`

Context:
- Source report: `reports/storage-frontier-2026.md` (draft)
- Figures: located under `reports/figures/` (PNG)
- CSV data: `reports/data/` (for chart regeneration)

Task for `ppt-content-planner`:
1. Expand `docs/presentations/storage-frontier-20260211/slides.md` into a full `slides_semantic.json` following SCQA / hierarchical structure.
2. Fill in slide `content[]` and `visual.placeholder_data` for charts and tables. Use existing PNGs; regenerate SVG/PNG if needed using `reports/figures/scripts/*.py`.
3. Add speaker notes for each slide (≥2 sentences) and action-oriented recommendations for CTO/architect audience.
4. Mark critical visuals with `cognitive_intent` (at least 3 visuals).
5. Create `design_spec.json` requirements for `ppt-visual-designer` (color tokens / typography / accessibility constraints).
6. Produce first draft `slides.md` (expanded) + `slides_semantic.json` and request design handoff to `ppt-visual-designer`.

Priority & timeline:
- Immediate start requested by stakeholder. **First draft `slides_semantic.json` (完整 25 页草稿，含讲者笔记) due within 24 hours.**
- Visual style: **BCG** — 请在 `design_spec.json` 中明确色彩/版式/可访问性约束并生成样例封面与 2 个内容页（含图表样式）。
- Expand vendor cases: **AWS / 阿里云 / 华为云 / Dell** → each vendor to have dedicated slides (建议每厂商 2–3 slides: 架构要点、试点建议、成本/风险要点)。
- Deliverables: `slides_semantic.json`, expanded `slides.md` (25 slides), speaker_notes (>=2 sentences per slide), `design_spec.json` (BCG), updated figures (high-res PNG/SVG with BCG styling if needed).
- QA checks before handoff to `ppt-visual-designer`:
  - total slides == 25
  - per-slide bullets <= 5
  - speaker_notes >= 2 sentences per slide
  - mark >= 3 critical visuals with `cognitive_intent`
  - ensure figure resolution >=150 DPI and provide SVG where feasible
- Deliver artifacts to: `docs/presentations/storage-frontier-20260211/` and notify `ppt-visual-designer` on completion for style handoff.

Open questions for content-planner (answer before starting):
- Confirm target slide count (default 12–16) and language (中文)
- Confirm whether to expand individual vendor cases into separate slides (Yes/No)

Deliver artifacts to: `docs/presentations/storage-frontier-20260211/` and create `qa_report.json` after `ppt-specialist` review later.

