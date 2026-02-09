#!/usr/bin/env python3
"""Pre-render mermaid diagrams and simple charts for MFT deck.

Usage: python3 scripts/pre_render_assets.py --spec output/MFT_design_spec.json

This script will:
- Read design_spec.visual_specs and visual_assets_manifest
- For mermaid visuals, call npx @mermaid-js/mermaid-cli to generate PNG
- For chart visuals (column_clustered, line_multi_series, radar, bar_clustered) render via matplotlib
- Update visual_assets_manifest statuses to 'rendered' and save updated design_spec.json
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def render_mermaid(mermaid_code: str, output_path: str, width: int = 1600, height: int = 900):
    # write temp mmd
    tmp_dir = Path('.').resolve() / 'tmp_mermaid'
    tmp_dir.mkdir(parents=True, exist_ok=True)
    mmd_path = tmp_dir / (Path(output_path).stem + '.mmd')
    with open(mmd_path, 'w', encoding='utf-8') as f:
        f.write(mermaid_code)
    # use npx mermaid-cli
    cmd = ['npx', '--yes', '@mermaid-js/mermaid-cli', '-i', str(mmd_path), '-o', str(output_path), '--width', str(width), '--height', str(height)]
    print('Running:', ' '.join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print('mmdc error:', res.stderr)
        return False
    return True


def render_bar_chart(chart_config, output_path):
    labels = chart_config['labels']
    series = chart_config['series']
    x = np.arange(len(labels))
    width = 0.7 / max(1, len(series))
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, s in enumerate(series):
        data = s['data']
        ax.bar(x + i * width, data, width, label=s.get('name'))
    ax.set_xticks(x + width * (len(series)-1)/2)
    ax.set_xticklabels(labels, rotation=0)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=144)
    plt.close()
    return True


def render_line_multi_series(chart_config, output_path):
    labels = chart_config['labels']
    series = chart_config['series']
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(12, 6))
    for s in series:
        ax.plot(x, s['data'], marker='o', label=s.get('name'))
    ax.set_xticks(x);
    ax.set_xticklabels(labels, rotation=0)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=144)
    plt.close()
    return True


def render_radar(chart_config, output_path):
    labels = chart_config['labels']
    series = chart_config['series']
    N = len(labels)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8,8), subplot_kw=dict(polar=True))
    for s in series:
        data = s['data']
        d = data + data[:1]
        ax.plot(angles, d, label=s.get('name'))
        ax.fill(angles, d, alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(output_path, dpi=144)
    plt.close()
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
    for asset in assets:
        path = output_folder / Path(asset['path']).name
        print('Asset:', asset.get('slide_id'), asset.get('path'), 'status:', asset.get('status'))
        if asset.get('status') == 'rendered':
            print('  already rendered')
            continue
        # find visual_spec by slide_id
        vis = next((v for v in spec.get('visual_specs', []) if v.get('slide_id') == asset.get('slide_id')), None)
        if not vis:
            print('  no visual spec found for', asset)
            continue
        chart_cfg = vis.get('chart_config', {})
        render_ok = False
        if asset.get('source_type') == 'mermaid' or 'mermaid_code' in chart_cfg:
            mermaid_code = chart_cfg.get('mermaid_code')
            if not mermaid_code and vis.get('visual_type', '').startswith('mermaid'):
                mermaid_code = vis.get('chart_config', {}).get('mermaid_code')
            if mermaid_code:
                render_ok = render_mermaid(mermaid_code, str(path), width=asset.get('dimensions', [1600,900])[0], height=asset.get('dimensions', [1600,900])[1])
        else:
            # handle chart types by chart_config
            ct = vis.get('render_instructions', {}).get('chart_type') or vis.get('chart_type')
            if ct in ('COLUMN_CLUSTERED','BAR_CLUSTERED','column_clustered','bar_clustered'):
                render_ok = render_bar_chart(chart_cfg, str(path))
            elif ct in ('LINE_MULTI_SERIES','line_multi_series'):
                render_ok = render_line_multi_series(chart_cfg, str(path))
            elif ct in ('RADAR','radar'):
                render_ok = render_radar(chart_cfg, str(path))
            else:
                # try basic bar
                render_ok = render_bar_chart(chart_cfg, str(path))

        if render_ok:
            asset['status'] = 'rendered'
            updated = True
            print('  rendered ->', path)
        else:
            print('  failed to render', asset)

    if updated:
        # write back spec
        with open(spec_path, 'w', encoding='utf-8') as f:
            json.dump(spec, f, ensure_ascii=False, indent=2)
        print('Updated spec with rendered assets.')

    print('Done')

if __name__ == '__main__':
    main()
