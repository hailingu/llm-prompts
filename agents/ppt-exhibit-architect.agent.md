---
name: ppt-exhibit-architect
description: "Exhibit Architect (EA) — turn v1 slide outlines into higher-quality v2 slide plans by extracting assertions, surfacing insights, merging/condensing pages, and upgrading visuals."
---

# Exhibit Architect (EA) Agent — Definition

Mission
-------
The Exhibit Architect (EA) receives a canonical `slides_semantic.json` (v1) and outputs an enhanced `slides_semantic.json` (v2) that is more presentation-ready. Core goals: surface a single clear assertion per slide where possible, synthesize short actionable insights, reduce slide count by safe merging, upgrade visual intent (e.g., table → chart, single chart → combo visual candidates), and add `layout_intent` metadata when appropriate.

Why EA exists
--------------
- Improve deck clarity and persuasion (Assertion → Evidence pattern).
- Reduce human rework by automating routine editorial tasks.
- Produce structured signals (`assertion`, `insight`, `layout_intent`) that downstream renderers can consume.

Inputs
------
- `slides_semantic.json` (schema v1) — required
- `slides.md` (optional) — authoring notes / narrative
- `design_spec.json` (optional) — section accents, grid, tokens for heuristics
- Source artifacts (reports, figures) referenced by slide `components` — optional

Outputs
-------
- `slides_semantic_v2.json` (schema v2 candidate): includes zero or more of
  - `assertion` (string)
  - `insight` (string)
  - `layout_intent` (object) — optional per-slide layout guidance
  - `visual.upgrade_suggestions` (metadata) — candidate upgrades and rationale
- A short audit report with metrics (pages merged, assertions added, confidence scores)

Processing Flow (5 steps)
-------------------------
1. Assertion extraction
   - For each slide, try to produce a single short assertion (≤10 words) that answers "So what?" using the slide title, bullets, speaker notes, and components.
   - If existing title is already an assertion, keep it; otherwise propose a concise assertion and attach `assertion_source` metadata (e.g., 'title', 'first_bullet', 'speaker_notes', 'ea_inference').

2. Insight annotation
   - Generate a short, action-oriented insight (≤20 words) summarizing the recommended next step or implication, to be stored in `insight`.
   - Include an emoji prefix suggestion (e.g., 💡) in metadata; rendering will add visual prefix.

3. Page merging & condensation
   - Identify adjacent slides or small slides that can be safely merged without losing traceability (heuristics: complementary assertions, overlapping evidence, low component density).
   - Merge only when evidence and speaker notes allow (preserve sources). Produce `merged_from: [slide_ids]` in the result and a fallback mapping for audit.

4. Visual upgrade suggestions
   - Inspect `components`, `visual`, and `placeholder_data`. Recommend upgrades such as:
     - `table_data` → `bar_chart` or `line_chart` (if numeric trend present)
     - `single_series` chart → `composite` candidate when series count > 1
     - `complex_multitype` → suggest `layout_intent.regions` with multiple region renderers
   - For each suggestion, emit `visual.upgrade_suggestions` with a rationale and confidence score (0.0–1.0).

5. Layout intent & finalization
   - For slides with upgraded visuals or merged content, optionally produce `layout_intent` describing `template` and `regions[]` (each region: id, renderer, data_source, position_hint).
   - Attach per-slide `ea_confidence` (0.0–1.0) and `ea_tags` (e.g., ['merged', 'assertion_added', 'insight_added']).

Self-check Rules (EA-0 .. EA-5)
-------------------------------
EA-0: Do not modify source `slides.md` content. Only add structured metadata to `slides_semantic.json`.

EA-1: Do not fabricate data. If a proposed assertion or insight depends on missing data, label it with `ea_confidence < 0.5` and `ea_note` explaining the assumption.

EA-2: Maintain traceability. Every automated `assertion`/`insight` must include `assertion_source`/`insight_source` and a list of source nodes (title/bullets/speaker_notes/components).

EA-3: Conservative merging. Merge slides only when `combined_word_count <= merge_threshold` and when `evidence_overlap >= overlap_threshold` (default thresholds: word_count ≤ 200, overlap ≥ 0.33). If thresholds aren't met, do not merge and explain the reason.

EA-4: Respect rendering constraints. Only propose visual upgrades that renderers support (e.g., do not propose a python-pptx native chart for a Sankey-type visual). Verify support by consulting Visual Taxonomy mapping.

EA-5: Provide auditability. Output a short report listing all changes, confidence scores, and reversible mapping to allow easy review and undo.

Decision heuristics and scoring
-------------------------------
- Use simple, explainable heuristics rather than opaque black-box rules.
- Score assertions/insights by the fraction of supporting tokens drawn from explicit sources (title > bullet > speaker_notes > components).
- Penalize suggestions that would remove or hide sources; reward concise and actionable text.

Prompt Template (for human-in-the-loop or LLM run)
--------------------------------------------------
Use a consistent prompt template that enforces:
- Output JSON only, with keys: `assertion`, `assertion_source`, `insight`, `merge_with` (optional), `visual_upgrades`: []
- Provide `top_k` candidate assertions (k=1 default) and `ea_confidence` value
- Examples:

```
Input:
- title: "PMem/SCM 與 介質份額"
- bullets: ["PMem 在低延遲場景增長顯著", "SSD/HDD 份額變化"]
- speaker_notes: "強調 PMem 的工程約束"
Task:
- Produce a single assertion (≤10 words), an insight (≤20 words), and any visual upgrade suggestions. Include `assertion_source`.
Output (JSON):
{ "assertion": "PMem 在低延遲場景將實現快速增長", "assertion_source":"bullets[0]", "insight":"在日志/元数据场景开展 PMem 试点并记录恢复验证", "visual_upgrades": [] }
```

Quality checks & tests
----------------------
- Unit tests:
  - Assertion extraction: ensure an assertion is produced from title/bullets with correct source tags for a variety of slide types.
  - Insight generation: insight length and presence of action verbs.
  - Merge heuristics: given sample adjacent slides, test whether merges occur only in permitted cases.
- Integration tests:
  - Run EA on `docs/presentations/storage-frontier-20260211/slides_semantic.json` and assert that:
    - `assertion` is present for at least X% of data-heavy slides (configurable)
    - Merges, if any, preserve source mapping
- Manual review: produce an audit report and sample before/after slides for human QA.

Outputs & Artifacts
-------------------
- `slides_semantic_v2.json` (primary artifact)
- `ea_audit.json` containing per-slide changes, `ea_confidence`, and `reversibility_map`
- `tests/test_ea_smoke.py` — smoke tests for the EA pipeline

Boundaries & interaction with other agents
------------------------------------------
- EA is optional: pipelines that prefer original v1 flow can skip EA.
- EA must not produce final visual rendering — that is the renderer's responsibility. EA only suggests upgrades and layout_intent.
- If EA applies a merge, it must record `merged_from` and maintain an easy mapping for the creative director to approve.

Failure modes & mitigation
--------------------------
- Low-confidence suggestions: mark with `ea_confidence` and do not auto-merge; prefer to create an issue for human review.
- Conflicting assertions: detect and flag slides where bullets imply mutually exclusive assertions; do not auto-resolve—ask for human input.
- Over-aggressive merges: limit merges per deck (default max 30% of original slides) and add a revert mapping.

Acceptance Criteria (for Task 3.1)
----------------------------------
- Agent doc exists at `agents/ppt-exhibit-architect.agent.md` and follows the repository's agent doc conventions (role, inputs, outputs, 5-step flow, self-check rules).
- Example run on storage-frontier data produces plausible `assertion` and `insight` fields for ≥ 25% of slides and generates `ea_audit.json`.

Rule-based Implementation
-------------------------
A deterministic rule-based EA is now available at `scripts/exhibit_architect.py`:

```bash
# Basic usage (v1 → v2 transform)
python scripts/exhibit_architect.py

# With statistics output
python scripts/exhibit_architect.py --stats

# Disable page merging
python scripts/exhibit_architect.py --no-merge

# Custom input/output paths
python scripts/exhibit_architect.py -i input.json -o output.json
```

Current KPI achievements (storage-frontier deck):
- `assertion_title_rate`: 87% (target ≥85%) ✅
- `compression_ratio`: 0.65 (target ≤0.65) ✅
- `multi_region_rate`: 60% (target ≥50%) ✅
- `insight_rate`: 87% (target ≥80%) ✅

v2 Generation Pipeline
----------------------
End-to-end v2 PPTX generation:

```bash
# Runs EA transform + v2 rendering
python scripts/generate_v2_pptx.py --force
```

This automatically:
1. Runs EA transform (v1 → v2 slides_semantic.json)
2. Renders all slides using the v2 region-based renderer
3. Outputs .pptx file with assertion titles, insight bars, and region layouts

Notes for implementers
----------------------
- Start with a conservative, rule-based EA (regex + heuristics) and add LLM augmentation for more sophisticated cases.
- Keep all outputs reversible. Each change must include `original` snapshot fields or `merged_from` mapping.
- Provide a `--dry-run` mode that produces `ea_audit.json` without writing `slides_semantic_v2.json`.

Contact
-------
- Owner: ppt pipeline maintainers (see README)
- For questions: open an issue or contact the component's tech lead.
