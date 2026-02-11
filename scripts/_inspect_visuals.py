#!/usr/bin/env python3
"""Check visual.placeholder_data for data-heavy slides."""
import json

sem = json.load(open('docs/presentations/storage-frontier-20260211/slides_semantic.json', encoding='utf-8'))

for idx in [4, 5, 6, 12, 16, 17]:
    sd = sem['slides'][idx]
    vis = sd.get('visual', {})
    pd = vis.get('placeholder_data', {})
    print(f"=== Slide {idx+1} ({sd['slide_type']}) ===")
    print(f"  visual.type: {vis.get('type','none')}")
    print(f"  visual.title: {vis.get('title','')[:60]}")
    if pd:
        cc = pd.get('chart_config', {})
        mc = pd.get('mermaid_code', '')
        print(f"  placeholder_data.chart_config: {bool(cc)} labels={cc.get('labels',[])[:5]}")
        if cc.get('series'):
            for s in cc['series'][:2]:
                print(f"    series: {json.dumps(s, ensure_ascii=False)[:100]}")
        if mc:
            print(f"  placeholder_data.mermaid_code: {mc[:100]}")
    else:
        print(f"  placeholder_data: EMPTY")
    # Also check top-level visual keys
    for k in vis:
        if k not in ('type', 'title', 'placeholder_data', 'content_requirements'):
            v = vis[k]
            if v:
                print(f"  visual.{k}: {json.dumps(v, ensure_ascii=False)[:100]}")
    print()
