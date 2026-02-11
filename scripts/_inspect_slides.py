#!/usr/bin/env python3
"""Inspect semantic data for slides 20, 22, 23."""
import json

sem = json.load(open('docs/presentations/storage-frontier-20260211/slides_semantic.json', encoding='utf-8'))

for idx in [19, 21, 22]:
    sd = sem['slides'][idx]
    print(f'=== Slide {idx+1} ({sd["slide_type"]}) ===')
    print(f'  title: {sd.get("title","")[:80]}')
    print(f'  assertion: {sd.get("assertion","")[:80]}')
    comps = sd.get('components', {})
    for k, v in comps.items():
        if v:
            if isinstance(v, list):
                print(f'  components.{k}: [{len(v)} items]')
                for item in v[:4]:
                    if isinstance(item, dict):
                        s = json.dumps(item, ensure_ascii=False)
                        print(f'    {s[:150]}')
                    else:
                        print(f'    {str(item)[:120]}')
            else:
                print(f'  components.{k}: {str(v)[:120]}')
    print(f'  content: {sd.get("content",[])}')
    vis = sd.get('visual', {})
    print(f'  visual: {json.dumps(vis, ensure_ascii=False)[:200]}')
    print()
