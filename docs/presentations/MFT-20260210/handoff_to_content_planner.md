To: ppt-content-planner
From: ppt-creative-director
Session: MFT-20260210

Objective: 请基于已存在的 `slides.md`（草案）与 creative brief 完成内容规划任务，输出 `slides_semantic.json` 与 12–15 张面向技术评审的 `slides.md` 可交付稿（保留关键决策 Slide 2）。

Requirements / Acceptance Criteria:
1. Follow **McKinsey Pyramid** structure (conclusion first, then supporting arguments).
2. SCQA mapping must align with `story_structure` in brief; ensure evidence slides correspond to KPIs and demo plan.
3. Produce `slides_semantic.json` with:
   - `cognitive_intent` annotations for ≥3 critical visuals
   - `sections` array and per-slide `components`
   - completeness (entries for all slides in `slides.md`)
4. Speaker notes coverage ≥ 80% and provide presenter cues for decision slides.
5. Bullet limit ≤5 per slide; prioritize visuals for comparison/priority slides.
6. Self-check using Content Strategy Review Checklist; include content_qa.json summary with submission.

Deliverables & Timeline:
- D+2 working days: `slides_semantic.json` + `slides.md` draft
- D+4 working days: handoff to Visual Designer (visual placeholders & design_spec recommendations)

If any checklist fails or ambiguous trade-offs arise, escalate to `ppt-creative-director` for rapid decision (limit 2 iterations before escalation).

Notes:
- Visual style: `mckinsey` (ensure slide layout and hierarchy reflect this)
- Maintain KPI traceability and linkage to evidence slides (示范 KPI 必须在证据幻灯内可验证)
- Keep the session directory updated under `docs/presentations/MFT-20260210/` with all artifacts
