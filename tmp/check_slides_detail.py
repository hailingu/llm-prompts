#!/usr/bin/env python3
"""检查问题slide的详细semantic数据"""
import json

with open('docs/presentations/MFT-20260210/slides_semantic.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

for sn in [1, 2, 9, 14, 15, 27, 28, 29]:
    s = data['slides'][sn-1]
    print(f'\n=== Slide {sn}: {s.get("title", "")} ===')
    print(f'  type: {s.get("type", "N/A")}')
    keys = [k for k in s.keys() if k not in ('speaker_notes',)]
    print(f'  keys: {keys}')
    
    if 'bullets' in s and s['bullets']:
        print(f'  bullets ({len(s["bullets"])}):')
        for b in s['bullets'][:3]:
            print(f'    - {str(b)[:80]}')
    
    if 'items' in s and s['items']:
        print(f'  items ({len(s["items"])}):')
        for item in s['items'][:3]:
            print(f'    - {str(item)[:80]}')
    
    if 'decisions' in s and s['decisions']:
        print(f'  decisions ({len(s["decisions"])}):')
        for d in s['decisions'][:3]:
            print(f'    - {str(d)[:80]}')

    if 'sections' in s:
        print(f'  sections ({len(s["sections"])}):')
        for sec in s['sections'][:3]:
            print(f'    - {str(sec)[:100]}')

    visual = s.get('visual', {})
    if visual:
        print(f'  visual.type: {visual.get("type", "?")}')
        pd = visual.get('placeholder_data', {})
        if pd:
            for k, v in pd.items():
                if k == 'mermaid_code':
                    print(f'  visual.pd.mermaid_code: {v[:80]}...')
                elif isinstance(v, list):
                    print(f'  visual.pd.{k} ({len(v)}): {str(v[:2])[:100]}...')
                else:
                    print(f'  visual.pd.{k}: {str(v)[:80]}')
