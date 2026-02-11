#!/usr/bin/env python3
import importlib.util, os, json
repo_root = '/Users/guhailin/Git/llm-prompts'
mod_path = os.path.join(repo_root,'skills','ppt-generator','bin','generate_pptx.py')
spec = importlib.util.spec_from_file_location('gen', mod_path)
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)
slides_path = os.path.join(repo_root,'docs','presentations','storage-frontier-20260211')
sem = json.load(open(os.path.join(slides_path,'slides_semantic.json'),'r',encoding='utf-8'))
for sid in [7,17]:
    sd = next(s for s in sem['slides'] if s.get('id')==sid)
    vis = sd.get('visual')
    print('\n---\n')
    print('slide',sid,'vis type',vis.get('type'))
    pd = vis.get('placeholder_data',{})
    config = pd.get('chart_config',{})
    print('initial config keys', list(config.keys()))
    # mimic logic
    if not config.get('labels') and any(isinstance(v, dict) for v in config.values()):
        first_key = next(iter(config.keys()))
        print('composite: first key', first_key)
        config = config[first_key]
    labels = config.get('labels') or config.get('x') or []
    series = config.get('series', [])
    print('labels len', len(labels), 'series len', len(series))
    if series:
        print(' series[0] keys', list(series[0].keys()))
        print(' series[0] sample types', {k:type(v).__name__ for k,v in series[0].items()})
        # print a small sample of data
        for k in ('data','y','x'):
            if k in series[0]:
                v = series[0][k]
                if isinstance(v,list):
                    print(f"  {k} sample len={len(v)} first={v[:3]}")
                else:
                    print(f"  {k} type {type(v).__name__}")
    chart_type = vis.get('type','line_chart')
    ALIAS_TYPE_MAP = {'bar_line_chart':'column_chart','composite_charts':'line_chart'}
    if chart_type in ALIAS_TYPE_MAP:
        chart_type = ALIAS_TYPE_MAP[chart_type]
        print('mapped chart_type to', chart_type)
    xl_type = gen.NATIVE_CHART_TYPE_MAP.get(chart_type)
    print('xl_type', xl_type)
    if chart_type != 'scatter_chart' and not labels:
        print('no labels => abort')
    else:
        try:
            if xl_type == gen.XL_CHART_TYPE.XY_SCATTER:
                print('would construct XY chart')
            else:
                print('would construct Category chart with categories', labels[:5])
        except Exception as e:
            print('error checking XL_CHART_TYPE', e)
