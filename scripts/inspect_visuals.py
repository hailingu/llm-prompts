#!/usr/bin/env python3
import json, os
repo_root = '/Users/guhailin/Git/llm-prompts'
slides_path=os.path.join(repo_root,'docs','presentations','storage-frontier-20260211')
sem=json.load(open(os.path.join(slides_path,'slides_semantic.json'),'r',encoding='utf-8'))
for sid in [7,17]:
    sd = next(s for s in sem['slides'] if s.get('id')==sid)
    vis=sd.get('visual')
    print('slide',sid,'type',vis.get('type'))
    print('keys',list(vis.keys()))
    # Print top-level fields and types
    for k,v in vis.items():
        print(' ',k,':',type(v).__name__)
    # If has "series" or "sub_visuals", print summary
    if 'series' in vis:
        print(' Series count:', len(vis['series']))
        for i,s in enumerate(vis['series']):
            print('  series',i,':', {kk:type(vv).__name__ for kk,vv in s.items()})
    if 'sub_visuals' in vis:
        print(' Sub-visuals count:', len(vis['sub_visuals']))
        for i,s in enumerate(vis['sub_visuals']):
            print('  sub',i,':', s.get('type'), 'keys', list(s.keys()))
    if 'placeholder_data' in vis:
            pd=vis['placeholder_data']
            print(' placeholder_data keys:', list(pd.keys()))
            if 'chart_config' in pd:
                cc = pd['chart_config']
                print(' chart_config keys:', list(cc.keys()))
                # Print a compact representation of the chart_config (avoid huge dumps)
                for k,v in cc.items():
                    if isinstance(v, dict):
                        print(f"  {k}: dict keys={list(v.keys())}")
                    elif isinstance(v, list):
                        print(f"  {k}: list len={len(v)}")
                    else:
                        print(f"  {k}: {type(v).__name__}")
                # If composite, print children
                if 'charts' in cc:
                    print('  charts count', len(cc['charts']))
                    for i,c in enumerate(cc['charts']):
                        print(f"   chart[{i}] type={c.get('type')} keys={list(c.keys())}")
                        if 'series' in c:
                            print('    series count',len(c['series']))
                # bar_line combo
                if 'series' in cc:
                    print(' series count', len(cc['series']))
                    for i,s in enumerate(cc['series'][:5]):
                        print('  series',i,':', {kk:(type(vv).__name__ if not isinstance(vv,(list,dict)) else f"{type(vv).__name__} len={len(vv)}") for kk,vv in s.items()})
    print('\n')
