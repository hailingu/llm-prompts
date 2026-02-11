# ppt-exhibit-design Skill

## Purpose

This skill provides practical methods, templates, and heuristics for the Exhibit Architect (EA) to make safe, explainable decisions about slide merging, assertion extraction, visual upgrades, and layout choices. It is intended for use by the EA agent and human designers who need consistent, auditable transformations of `slides_semantic.json` into higher-quality v2 deliverables.

## Contents
- So-What? Assertion Extraction (three-question method)
- Page Merge Rules & Matrix
- Visual Upgrade Mapping (recommendations + rationales)
- Layout Template Decision Tree
- Information Density & Thresholds (heuristics for merging / splitting)
- Examples and smoke tests

---

## 1. Assertion Extraction — So-What? (Three questions)
When generating an assertion for a slide, prefer short, decisive text (≤ 10 words). Use the following ordered evidence priorities:
1. Title (if already assertion-style)
2. First bullet or prominent KPI
3. Speaker notes or component headings

Three-question method (apply in order):
- What changed? (observed fact from data)
- So what? (implication or consequence)
- What should the audience do/decide? (action or recommendation)

Produce:
- `assertion` (≤ 10 words)
- `assertion_source` (one of: title, bullet[i], speaker_notes, components)
- `ea_confidence` (0.0–1.0)

## 2. Page Merge Rules & Matrix
Only merge slides when it preserves traceability and keeps the deck readable.

Primary merge heuristics:
- Word-count threshold: combined_word_count ≤ 200
- Evidence overlap threshold: ≥ 33% overlap in key tokens (nouns/metrics)
- Component conflict: do not merge if two slides have mutually exclusive key decisions
- Max deck compression by EA: 30% (configurable)

Merge decision matrix (example):
- Adjacent summary slide + evidence slide → merge (if combined_word_count ≤ 200)
- Two short callout slides (< 3 bullets each) → merge (if topics related)
- Data heavy chart + method slide → do NOT merge (keeps evidence clean)

When merging, produce `merged_from: [ids]` and keep `original_*` snapshots for audit and reversibility.

## 3. Visual Upgrade Mapping
Rules for recommending visual upgrades. Always check renderer support (EA-4).

Examples:
- `table_data` with numeric columns → recommend `bar_chart` or `stacked_bar` when comparing categories.
- `single_series` trend values → recommend `line_chart` if x-axis time-like.
- `two_numeric_series` → recommend `grouped_bar` or `line+bar` (flag as `combo_candidate`).
- `percent_share` across categories → recommend `stacked_bar` or `100% stacked` rather than multiple pie charts.

Each suggestion includes:
- `candidate_renderer` (e.g., `line_chart`)
- `rationale` (short explanation)
- `confidence` (0.0–1.0)

## 4. Layout Template Decision Tree
Choose `template` based on content density and component mix:
- If visual is `data-heavy` and assertion present → `visual-left/detail-right` or `visual-full` if single key visual
- If multiple small components → `two-column` or `tiered-cards` with `regions` for each component
- If assertion + small evidence → `assertion-top, visual-center, insight-bottom`

Produce `layout_intent` with `regions[]` where each region includes: `id`, `renderer`, `data_source`, `position_hint`.

## 5. Information Density & Thresholds
Defaults (tuned by EA config):
- Max text per slide before recommending split: 220 words
- Max distinct components per slide before suggesting multi-region layout: 4
- Merge thresholds: combined_word_count ≤ 200 and evidence_overlap ≥ 0.33

Heuristic scoring for layout selection is a simple weighted sum of:
- component_count (weight 0.4)
- word_count (weight 0.3)
- visual_complexity (weight 0.3)

## 6. Examples and Smoke Tests
- Example: PMem slide
  - Title: "PMem/SCM 與 介質份額"
  - Bullets: ["PMem 在低延遲場景增長顯著", "SSD/HDD 份額變化"]
  - EA outputs:
    - assertion: "PMem 在低延遲場景將實現快速增長"
    - insight: "在日志/元数据场景开展 PMem 试点并记录恢复验证"
    - visual_upgrades: []

Smoke tests:
- Run a small script to assert that `assertion` appears for ≥ 25% of data-heavy slides in `storage-frontier` sample.
- Run a merge-safety test: provide two slides likely candidate for merge and assert `merged_from` only when thresholds satisfied.

## 7. QA and Human Review
- EA should emit `ea_audit.json` with per-slide changes and confidence scores.
- Human in the loop: UI or PR review should show before/after snapshots and allow revert of individual changes.

## 8. Implementation notes
- Start with rule-based heuristics and add LLM prompts for improved candidates.
- Keep all transformations reversible and auditable.
- Provide a `--dry-run` mode that generates `ea_audit.json` without writing v2 JSON files.

## Contact & Ownership
- Owner: ppt pipeline maintainers (see repo README)
- For questions or improvements: open an issue and tag `area/exhibit-architect`
