#!/usr/bin/env python3
"""Fallback SVG renderer for simple charts (table matrix and column clustered)"""
import argparse
import json
from pathlib import Path


def render_table_matrix(chart_config, output_path: Path):
    labels = chart_config.get('labels', [])
    series = chart_config.get('series', [])
    cols = len(labels)
    rows = len(series) + 1  # header + series
    cell_w = 220
    cell_h = 48
    width = cols * cell_w + 40
    height = rows * cell_h + 40
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}">'.format(width, height)]
    svg.append('<style>text{font-family: "Noto Sans", Arial, sans-serif; font-size:14px; fill:#111}</style>')
    # header
    x0 = 20
    y0 = 20
    for j, label in enumerate(labels):
        x = x0 + j * cell_w
        svg.append(f'<rect x="{x}" y="{y0}" width="{cell_w}" height="{cell_h}" fill="#2563EB"/>')
        svg.append(f'<text x="{x+cell_w/2}" y="{y0+cell_h/2+6}" text-anchor="middle" fill="#fff">{label}</text>')
    # rows
    for i, s in enumerate(series):
        y = y0 + (i+1) * cell_h
        for j, val in enumerate(s.get('data', [])):
            x = x0 + j * cell_w
            svg.append(f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" fill="#FFF" stroke="#E6E6E6"/>')
            svg.append(f'<text x="{x+cell_w/2}" y="{y+cell_h/2+6}" text-anchor="middle">{val}</text>')
    svg.append('</svg>')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text('\n'.join(svg), encoding='utf-8')
    return True


def render_column_clustered(chart_config, output_path: Path):
    labels = chart_config.get('labels', [])
    series = chart_config.get('series', [])
    cols = len(labels)
    series_count = len(series)
    width = 1200
    height = 600
    margin = 80
    plot_w = width - 2*margin
    plot_h = height - 2*margin
    bar_group_w = plot_w / cols
    bar_w = bar_group_w * 0.6 / max(1, series_count)
    max_val = 0
    for s in series:
        for v in s.get('data', []):
            try:
                max_val = max(max_val, float(v))
            except:
                pass
    if max_val == 0:
        max_val = 1
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}">'.format(width, height)]
    svg.append('<style>text{font-family: "Noto Sans", Arial, sans-serif; font-size:14px; fill:#111}</style>')
    # axes
    origin_x = margin
    origin_y = margin + plot_h
    svg.append(f'<line x1="{origin_x}" y1="{margin}" x2="{origin_x}" y2="{origin_y}" stroke="#ccc"/>')
    svg.append(f'<line x1="{origin_x}" y1="{origin_y}" x2="{origin_x+plot_w}" y2="{origin_y}" stroke="#ccc"/>')
    # bars
    for i, label in enumerate(labels):
        group_x = origin_x + i * bar_group_w
        for j, s in enumerate(series):
            data = s.get('data', [])
            val = 0
            if i < len(data):
                try:
                    val = float(data[i])
                except:
                    val = 0
            h = (val / max_val) * (plot_h * 0.9)
            x = group_x + (j * bar_w) + (bar_group_w * 0.2 / 2)
            y = origin_y - h
            color = '#2563EB' if j == 0 else '#10B981'
            svg.append(f'<rect x="{x}" y="{y}" width="{bar_w}" height="{h}" fill="{color}"/>')
        svg.append(f'<text x="{group_x + bar_group_w/2}" y="{origin_y + 20}" text-anchor="middle">{label}</text>')
    # legend
    legend_x = origin_x + plot_w + 20
    ly = margin
    for j, s in enumerate(series):
        color = '#2563EB' if j == 0 else '#10B981'
        svg.append(f'<rect x="{legend_x}" y="{ly}" width="14" height="14" fill="{color}"/>')
        svg.append(f'<text x="{legend_x+20}" y="{ly+12}">{s.get("name","series")}</text>')
        ly += 24
    svg.append('</svg>')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text('\n'.join(svg), encoding='utf-8')
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', required=True)
    args = parser.parse_args()
    spec_path = Path(args.spec)
    spec = json.loads(spec_path.read_text(encoding='utf-8'))
    output_folder = Path(spec.get('visual_assets_manifest', {}).get('output_folder', 'docs/presentations/mft-20260206/images'))
    assets = spec.get('visual_assets_manifest', {}).get('assets', [])
    vis_map = {v['slide_id']: v for v in spec.get('visual_specs', [])}
    changed = False
    for asset in assets:
        if asset.get('status') == 'rendered':
            continue
        slide_id = asset.get('slide_id')
        vis = vis_map.get(slide_id)
        if not vis:
            continue
        chart_cfg = vis.get('chart_config', {})
        ct = vis.get('render_instructions', {}).get('chart_type') or vis.get('visual_type')
        out_svg = output_folder / (Path(asset['path']).stem + '.svg')
        ok = False
        if 'TABLE_MATRIX' in ct or vis.get('visual_type') == 'matrix':
            ok = render_table_matrix(chart_cfg, out_svg)
        elif 'COLUMN' in ct or vis.get('visual_type') == 'comparison':
            ok = render_column_clustered(chart_cfg, out_svg)
        else:
            # fallback to column
            ok = render_column_clustered(chart_cfg, out_svg)
        if ok:
            asset['path'] = str(Path(asset['path']).with_suffix('.svg'))
            asset['format'] = 'svg'
            asset['status'] = 'rendered'
            changed = True
            print('Rendered chart svg for slide', slide_id, '->', out_svg)
    if changed:
        spec['visual_assets_manifest']['assets'] = assets
        spec_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding='utf-8')
        print('Updated spec with chart SVGs marked as rendered')
    else:
        print('No chart assets updated')

if __name__ == '__main__':
    main()
