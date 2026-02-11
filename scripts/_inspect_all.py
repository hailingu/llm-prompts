#!/usr/bin/env python3
"""Inspect data-heavy slides' visual info and content, plus section dividers."""
import json

sem = json.load(open('docs/presentations/storage-frontier-20260211/slides_semantic.json', encoding='utf-8'))

# Check data-heavy slides (5,6,7,13,17) and flowchart (18)
for idx in [4, 5, 6, 12, 16, 17]:
    sd = sem['slides'][idx]
    print(f"=== Slide {idx+1} ({sd['slide_type']}) ===")
    print(f"  title: {sd.get('title','')[:70]}")
    vis = sd.get('visual', {})
    print(f"  visual.type: {vis.get('type','none')}")
    if vis.get('type') and vis.get('type') != 'none':
        print(f"  visual.chart_type: {vis.get('chart_type','')}")
        series = vis.get('series', [])
        print(f"  visual.series: {len(series)} items")
        for s in series[:2]:
            print(f"    {json.dumps(s, ensure_ascii=False)[:120]}")
        labels = vis.get('labels', [])
        print(f"  visual.labels: {labels[:6]}")
        if vis.get('mermaid'):
            print(f"  visual.mermaid: {vis['mermaid'][:100]}")
    comps = sd.get('components', {})
    for k, v in comps.items():
        if v:
            print(f"  components.{k}: {len(v) if isinstance(v, list) else 'val'}")
    print()

# Section dividers
for idx in [1, 3, 7, 11, 15, 18, 20]:
    sd = sem['slides'][idx]
    print(f"=== Slide {idx+1} ({sd['slide_type']}) ===")
    print(f"  title: {sd.get('title','')[:70]}")
    print(f"  content: {sd.get('content', [])}")
    couts = sd.get('components', {}).get('callouts', [])
    if couts:
        print(f"  callouts: {len(couts)}")
    print()

# bullet-list slides
for idx in [2, 14, 19]:
    sd = sem['slides'][idx]
    print(f"=== Slide {idx+1} ({sd['slide_type']}) ===")
    print(f"  title: {sd.get('title','')[:70]}")
    comps = sd.get('components', {})
    for k, v in comps.items():
        if v:
            n = len(v) if isinstance(v, list) else 'val'
            print(f"  components.{k}: {n}")
    print()
