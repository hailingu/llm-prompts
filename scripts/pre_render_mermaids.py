#!/usr/bin/env python3
"""Pre-render mermaid diagrams only using npx @mermaid-js/mermaid-cli"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def render_mermaid(mermaid_code: str, output_path: str, width: int = 1600, height: int = 900):
    tmp_dir = Path('.').resolve() / 'tmp_mermaid'
    tmp_dir.mkdir(parents=True, exist_ok=True)
    mmd_path = tmp_dir / (Path(output_path).stem + '.mmd')
    with open(mmd_path, 'w', encoding='utf-8') as f:
        f.write(mermaid_code)
    cmd = ['npx', '--yes', '@mermaid-js/mermaid-cli', '-i', str(mmd_path), '-o', str(output_path), '--width', str(width), '--height', str(height)]
    print('Running:', ' '.join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print('mmdc error:', res.stderr)
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', required=True)
    args = parser.parse_args()
    spec_path = Path(args.spec)
    if not spec_path.exists():
        print('Design spec not found:', spec_path)
        sys.exit(1)
    with open(spec_path, encoding='utf-8') as f:
        spec = json.load(f)

    output_folder = Path(spec.get('visual_assets_manifest', {}).get('output_folder', 'docs/presentations/mft-20260206/images'))
    output_folder.mkdir(parents=True, exist_ok=True)

    assets = spec.get('visual_assets_manifest', {}).get('assets', [])
    updated = False
    # Map slide_id to visual_spec
    vis_map = {v['slide_id']: v for v in spec.get('visual_specs', [])}

    for asset in assets:
        slide_id = asset.get('slide_id')
        vis = vis_map.get(slide_id)
        if not vis:
            continue
        chart_cfg = vis.get('chart_config', {})
        is_mermaid = asset.get('source_type') == 'mermaid' or 'mermaid_code' in chart_cfg or (vis.get('render_instructions', {}).get('chart_type', '').startswith('MERMAID'))
        if not is_mermaid:
            continue
        path = output_folder / Path(asset['path']).name
        print('Rendering mermaid for slide', slide_id, '->', path)
        mermaid_code = chart_cfg.get('mermaid_code') or vis.get('chart_config', {}).get('mermaid_code') or vis.get('placeholder_data', {}).get('mermaid_code')
        if not mermaid_code:
            print('  no mermaid code found for slide', slide_id)
            continue
        width, height = asset.get('dimensions', [1600,900])
        ok = render_mermaid(mermaid_code, str(path), width=width, height=height)
        if ok:
            asset['status'] = 'rendered'
            updated = True
            print('  rendered ->', path)
        else:
            print('  failed to render slide', slide_id)

    if updated:
        with open(spec_path, 'w', encoding='utf-8') as f:
            json.dump(spec, f, ensure_ascii=False, indent=2)
        print('Updated design_spec with rendered asset statuses.')

    print('Done')

if __name__ == '__main__':
    main()
