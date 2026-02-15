#!/usr/bin/env python3
"""Generate a PPTX from an existing v2 semantic deck (v2-only).

This script:
1. Loads `slides_semantic_v2.json` from the session directory
2. Renders slides using the region-based v2 renderer only
3. Outputs a PPTX file

Usage:
    python scripts/generate_v2_pptx.py --base-dir <session-dir> [--output file.pptx]
"""
import argparse
import json
import os
import sys

# Package import context
sys.path.insert(0, os.path.abspath('skills/ppt-generator'))

from pptx import Presentation
from pptx.util import Inches

from ppt_generator import renderers


DEFAULT_BASE_DIR = 'docs/presentations/storage-frontier-20260211'


def _paths(base_dir: str):
    sem_v2 = os.path.join(base_dir, 'slides_semantic_v2.json')
    des = os.path.join(base_dir, 'design_spec.json')
    out = os.path.join(base_dir, os.path.basename(base_dir) + '-v2.pptx')
    return sem_v2, des, out


def main():
    parser = argparse.ArgumentParser(description='Generate v2 PPTX (v2-only)')
    parser.add_argument('--base-dir', default=DEFAULT_BASE_DIR,
                        help='Session directory containing slides_semantic_v2.json and design_spec.json')
    parser.add_argument('--output', '-o', default=None, help='Output PPTX path (default: <base-dir>/<session>-v2.pptx)')
    args = parser.parse_args()

    base_dir = args.base_dir
    SEM_V2, DES, OUT_DEFAULT = _paths(base_dir)
    output_path = args.output or OUT_DEFAULT

    if not os.path.exists(SEM_V2):
        raise FileNotFoundError(
            f"Missing v2 semantic file: {SEM_V2}. "
            "Run `python scripts/exhibit_architect.py` first to produce slides_semantic_v2.json."
        )
    if not os.path.exists(DES):
        raise FileNotFoundError(f"Missing design spec file: {DES}")

    print(f'[Input] Using existing v2 JSON: {SEM_V2}')

    # Step 2: Load v2 data and design spec
    with open(SEM_V2, 'r', encoding='utf-8') as f:
        semantic = json.load(f)
    with open(DES, 'r', encoding='utf-8') as f:
        design = json.load(f)

    # Step 3: Render
    grid = renderers.GridSystem(design)
    prs = Presentation()
    prs.slide_width = Inches(grid.slide_w)
    prs.slide_height = Inches(grid.slide_h)

    sections = semantic.get('sections', [])
    slides = semantic.get('slides', [])
    total = len(slides)

    # Statistics tracking
    v2_count = 0

    for i, sd in enumerate(slides, 1):
        # Inject deck title for title slide (semantic uses deck_title)
        if sd.get('slide_type') == 'title':
            sd['_deck_title'] = semantic.get('deck_title', semantic.get('title', ''))

        stype = sd.get('slide_type')
        regions = ((sd.get('layout_intent') or {}).get('regions') or []) if isinstance(sd.get('layout_intent'), dict) else []
        if stype not in ('title', 'section_divider') and not regions:
            raise ValueError(f"Slide {i} is missing layout_intent.regions (v2-only mode)")
        v2_count += 1

        renderers.render_slide(prs, sd, design, grid, sections, i, total)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    prs.save(output_path)

    print(f'\n[Output] Saved: {output_path}')
    print(f'[Stats] Total slides: {total}')
    print(f'[Stats] v2 rendered (region-based): {v2_count}')

    # Metrics
    assertion_count = sum(1 for s in slides if s.get('assertion'))
    insight_count = sum(1 for s in slides if s.get('insight'))
    multi_region = sum(1 for s in slides if s.get('layout_intent') and len(s['layout_intent'].get('regions', [])) >= 2)
    print(f'[Stats] Assertion titles: {assertion_count}/{total}')
    print(f'[Stats] Insight bars:     {insight_count}/{total}')
    print(f'[Stats] Multi-region:     {multi_region}/{total}')


if __name__ == '__main__':
    main()
