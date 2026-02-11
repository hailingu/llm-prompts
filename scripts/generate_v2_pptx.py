#!/usr/bin/env python3
"""Generate a v2 PPTX using the EA-transformed slides_semantic_v2.json.

This script:
1. Runs the EA transform (v1 → v2) if v2 JSON doesn't exist or --force is used
2. Renders the v2 slides using the region-based v2 renderer
3. Outputs a PPTX file

Usage:
    python scripts/generate_v2_pptx.py [--force] [--no-merge]
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

# Also import the EA transform
sys.path.insert(0, os.path.abspath('scripts'))
from exhibit_architect import transform_v1_to_v2


BASE_DIR = 'docs/presentations/storage-frontier-20260211'
SEM_V1 = os.path.join(BASE_DIR, 'slides_semantic.json')
SEM_V2 = os.path.join(BASE_DIR, 'slides_semantic_v2.json')
DES = os.path.join(BASE_DIR, 'design_spec.json')
OUT = os.path.join(BASE_DIR, 'storage-frontier-v2.pptx')


def main():
    parser = argparse.ArgumentParser(description='Generate v2 PPTX')
    parser.add_argument('--force', action='store_true', help='Force regenerate v2 JSON even if it exists')
    parser.add_argument('--no-merge', action='store_true', help='Disable page merging in EA transform')
    parser.add_argument('--output', '-o', default=OUT, help='Output PPTX path')
    args = parser.parse_args()

    # Step 1: EA Transform (v1 → v2)
    if args.force or not os.path.exists(SEM_V2):
        print('[EA] Transforming v1 → v2...')
        with open(SEM_V1, 'r', encoding='utf-8') as f:
            v1_data = json.load(f)
        v2_data = transform_v1_to_v2(v1_data, enable_merge=not args.no_merge)
        with open(SEM_V2, 'w', encoding='utf-8') as f:
            json.dump(v2_data, f, ensure_ascii=False, indent=2)
        meta = v2_data.get('ea_transform', {})
        print(f'[EA] {meta.get("original_slide_count")} → {meta.get("output_slide_count")} slides')
    else:
        print(f'[EA] Using existing v2 JSON: {SEM_V2}')

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
    v1_count = 0

    for i, sd in enumerate(slides, 1):
        # Inject deck title for title slide
        if sd.get('slide_type') == 'title':
            sd['_deck_title'] = semantic.get('title', '')

        # Detect which path will be used
        version = renderers.detect_schema_version(sd)
        if version == 2:
            v2_count += 1
        else:
            v1_count += 1

        renderers.render_slide(prs, sd, design, grid, sections, i, total)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    prs.save(args.output)

    print(f'\n[Output] Saved: {args.output}')
    print(f'[Stats] Total slides: {total}')
    print(f'[Stats] v2 rendered (region-based): {v2_count}')
    print(f'[Stats] v1 rendered (fallback):     {v1_count}')

    # Metrics
    assertion_count = sum(1 for s in slides if s.get('assertion'))
    insight_count = sum(1 for s in slides if s.get('insight'))
    multi_region = sum(1 for s in slides if s.get('layout_intent') and len(s['layout_intent'].get('regions', [])) >= 2)
    print(f'[Stats] Assertion titles: {assertion_count}/{total}')
    print(f'[Stats] Insight bars:     {insight_count}/{total}')
    print(f'[Stats] Multi-region:     {multi_region}/{total}')


if __name__ == '__main__':
    main()
