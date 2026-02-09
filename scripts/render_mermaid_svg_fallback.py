#!/usr/bin/env python3
"""Fallback SVG renderer for simple mermaid flowcharts/sequence/gantt.
Creates basic box-and-arrow SVGs for each mermaid asset when mmdc is unavailable.
"""
import argparse
import json
import os
from pathlib import Path


def parse_mermaid_nodes_edges(code: str):
    # Very simple parser: find lines like A[Text] --> B[Text]
    nodes = {}
    edges = []
    for line in code.splitlines():
        line = line.strip()
        if not line or line.startswith('%'):
            continue
        if '-->' in line or '->>' in line:
            parts = line.replace('->>','-->').split('-->')
            left = parts[0].strip()
            right = parts[1].strip()
            def extract_id_text(token):
                if '[' in token and ']' in token:
                    id_ = token.split('[')[0].strip()
                    text = token.split('[')[1].split(']')[0].strip()
                else:
                    id_ = token.strip()
                    text = id_
                return id_, text
            lid, ltext = extract_id_text(left)
            rid, rtext = extract_id_text(right)
            nodes[lid] = ltext
            nodes[rid] = rtext
            edges.append((lid, rid))
        elif line.startswith('gantt') or line.startswith('sequenceDiagram'):
            # fallback, treat as single node
            pass
    return nodes, edges


def render_svg(nodes, edges, output_path: Path, title: str = ''):
    # layout nodes horizontally
    width = 1600
    height = 900
    node_w = 220
    node_h = 70
    margin_x = 80
    spacing = 40
    n = len(nodes)
    ids = list(nodes.keys())
    total_w = n * node_w + (n-1) * spacing
    start_x = (width - total_w) // 2
    y = height // 2 - node_h // 2

    svg_lines = []
    svg_lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    svg_lines.append('<style>text{font-family: "Noto Sans", Arial, sans-serif; font-size:16px; fill:#111}</style>')
    if title:
        svg_lines.append(f'<text x="{width/2}" y="40" text-anchor="middle" font-size="20" fill="#2563EB">{title}</text>')
    positions = {}
    for i, id_ in enumerate(ids):
        x = start_x + i * (node_w + spacing)
        positions[id_] = (x, y)
        svg_lines.append(f'<rect x="{x}" y="{y}" width="{node_w}" height="{node_h}" rx="8" ry="8" fill="#F2F6FF" stroke="#2563EB" stroke-width="2"/>')
        text = nodes[id_]
        svg_lines.append(f'<text x="{x + node_w/2}" y="{y + node_h/2 +6}" text-anchor="middle">{text}</text>')
    # draw arrows
    for (a,b) in edges:
        ax, ay = positions[a]
        bx, by = positions[b]
        x1 = ax + node_w
        y1 = ay + node_h/2
        x2 = bx
        y2 = by + node_h/2
        # line
        svg_lines.append(f'<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" fill="#2563EB"/></marker></defs>')
        svg_lines.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#2563EB" stroke-width="2" marker-end="url(#arrow)"/>')
    svg_lines.append('</svg>')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(svg_lines))
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', required=True)
    args = parser.parse_args()
    spec_path = Path(args.spec)
    spec = json.loads(spec_path.read_text(encoding='utf-8'))
    output_folder = Path(spec.get('visual_assets_manifest', {}).get('output_folder', 'docs/presentations/mft-20260206/images'))
    assets = spec.get('visual_assets_manifest', {}).get('assets', [])
    changed = False
    for asset in assets:
        slide_id = asset.get('slide_id')
        if asset.get('status') == 'rendered':
            continue
        vis = next((v for v in spec.get('visual_specs', []) if v.get('slide_id') == slide_id), None)
        if not vis:
            continue
        chart_cfg = vis.get('chart_config', {})
        mermaid_code = chart_cfg.get('mermaid_code') or vis.get('chart_config', {}).get('mermaid_code') or vis.get('placeholder_data', {}).get('mermaid_code')
        if not mermaid_code:
            continue
        nodes, edges = parse_mermaid_nodes_edges(mermaid_code)
        if not nodes:
            # create simple placeholder SVG with the mermaid code snippet
            title = vis.get('title', f'Slide {slide_id}') if vis else f'Slide {slide_id}'
            svg_path = output_folder / (Path(asset['path']).stem + '.svg')
            svg_path.parent.mkdir(parents=True, exist_ok=True)
            svg_content = f'<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="900"><rect width="100%" height="100%" fill="#FFFFFF"/><text x="80" y="120" font-size="16">{mermaid_code.replace('<','&lt;').replace('>','&gt;')}</text></svg>'
            svg_path.write_text(svg_content, encoding='utf-8')
            asset['path'] = str(Path(asset['path']).with_suffix('.svg'))
            asset['format'] = 'svg'
            asset['status'] = 'rendered'
            changed = True
            print('Wrote svg placeholder for slide', slide_id, '->', svg_path)
            continue
        svg_path = output_folder / (Path(asset['path']).stem + '.svg')
        render_svg(nodes, edges, svg_path, title=vis.get('title', f'Slide {slide_id}'))
        asset['path'] = str(Path(asset['path']).with_suffix('.svg'))
        asset['format'] = 'svg'
        asset['status'] = 'rendered'
        changed = True
        print('Rendered svg for slide', slide_id, '->', svg_path)
    if changed:
        spec['visual_assets_manifest']['assets'] = assets
        spec_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding='utf-8')
        print('Updated spec with SVG assets marked as rendered')
    else:
        print('No mermaid assets updated')

if __name__ == '__main__':
    main()
