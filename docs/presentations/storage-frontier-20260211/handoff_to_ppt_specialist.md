# Handoff to `ppt-specialist`

Status: **Design complete (BCG)** — ready for PPT rendering ✅

Included assets:
- `slides_semantic.json` — 25 slides with inline visual placeholder_data
- `slides.md` — readable slide outline with speaker notes
- `design_spec.json` — BCG visual tokens, typography, layout rules
- `images/cover_bg.jpg` — cover image placeholder (landscape 1920x1080 svg content saved as .jpg)
- `content_qa_report.json` & `visual_qa_report.json` — content and visual QA pass reports

Instructions for `ppt-specialist`:
1. Use `design_spec.json` tokens to set master slides and apply BCG palette/typography.
2. Render charts from `slides_semantic.json.visual.placeholder_data` as SVG (apply `chart_colors` palette) and export high-res PNGs (>=150 DPI). Place the rendered files under `images/` and update `slides_semantic.json` by adding `rendered_svg_path` and `rendered_png_path` for each visual.
3. Build the PPTX using the provided master slide, check all slide backgrounds, and ensure title/section_divider slides use `primary` background with white text.
4. Validate final PPTX for accessibility (contrast >=4.5:1 for normal text) and for layout consistency (max 5 bullets/slide, speaker notes present).
5. When finished, return updated `slides_semantic.json` + `pptx` in the session directory and run the visual-content integration QA.

If any issues arise during rendering or token application, open an issue in this session directory and escalate to the stakeholder only if MV-1..MV-13 checks fail.

Owner: `ppt-specialist` — Please acknowledge and provide ETA for PPTX generation.