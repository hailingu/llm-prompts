"""CLI entrypoint for ppt_generator package."""
import argparse
import json
import os
import re

from pptx import Presentation
from pptx.util import Inches

from .grid import GridSystem
from .renderers import render_slide


def generate_pptx(semantic_path: str, design_spec_path: str, output_path: str) -> str:
    """Load semantic + design JSON, render slides, and save PPTX.

    Returns the output path on success.
    """
    with open(semantic_path, encoding='utf-8') as f:
        semantic = json.load(f)
    with open(design_spec_path, encoding='utf-8') as f:
        spec = json.load(f)

    grid = GridSystem(spec)

    prs = Presentation()
    prs.slide_width = Inches(grid.slide_w)
    prs.slide_height = Inches(grid.slide_h)

    sections = semantic.get('sections', [])
    slides_data = semantic.get('slides', [])
    total = len(slides_data)
    deck_title = semantic.get('title', '')
    deck_title = re.sub(r'[（(]\s*\d+\s*分钟\s*[）)]\s*$', '', deck_title).strip()

    for i, sd in enumerate(slides_data, 1):
        if sd.get('slide_type') == 'title':
            sd['_deck_title'] = deck_title
        render_slide(prs, sd, spec, grid, sections, i, total)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    prs.save(output_path)
    print(f"✅ PPTX saved: {output_path} ({total} slides)")

    # Compute and persist metrics (best-effort)
    try:
        from .metrics import compute_deck_metrics, write_metrics, audit_metrics
        metrics = compute_deck_metrics(semantic)
        # attach some meta
        metrics['deck_id'] = deck_title or os.path.basename(output_path)
        # pick schema version heuristically: 2 if any slide is v2
        schema_version = 1
        for sd in slides_data:
            if sd and isinstance(sd, dict) and 'layout_intent' in sd and isinstance(sd.get('layout_intent'), dict) and 'regions' in sd.get('layout_intent'):
                schema_version = 2
                break
        metrics['schema_version'] = schema_version
        # Run audit and attach warnings
        warnings = audit_metrics(metrics)
        if warnings:
            metrics['warnings'] = warnings
        metrics_file = write_metrics(metrics, os.path.dirname(os.path.abspath(output_path)), deck_id=metrics['deck_id'])
        print(f"🔢 Metrics appended to: {metrics_file}")
    except Exception as e:
        print(f"⚠️ Metrics persistence failed: {e}")

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate PPTX from semantic JSON + design spec')
    parser.add_argument('--semantic', required=True, help='Path to slides_semantic.json')
    parser.add_argument('--design', required=True, help='Path to design_spec.json')
    parser.add_argument('--output', required=True, help='Output PPTX path')
    args = parser.parse_args()
    generate_pptx(args.semantic, args.design, args.output)


if __name__ == '__main__':
    main()
