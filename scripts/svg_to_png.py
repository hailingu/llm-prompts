#!/usr/bin/env python3
"""Convert SVG assets in design_spec to PNG for PPT embedding."""
import argparse
import json
from pathlib import Path
from cairosvg import svg2png


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', required=True)
    args = parser.parse_args()
    spec_path = Path(args.spec)
    spec = json.loads(spec_path.read_text(encoding='utf-8'))
    assets = spec.get('visual_assets_manifest', {}).get('assets', [])
    output_folder = Path(spec.get('visual_assets_manifest', {}).get('output_folder', 'docs/presentations/mft-20260206/images'))
    updated = False
    for a in assets:
        path = a.get('path')
        if not path:
            continue
        p = output_folder / Path(path).name
        if p.suffix.lower() == '.svg':
            png_path = p.with_suffix('.png')
            print('Converting', p, '->', png_path)
            svg2png(url=str(p), write_to=str(png_path))
            a['path'] = str(Path(a['path']).with_suffix('.png'))
            a['format'] = 'png'
            a['status'] = 'rendered'
            updated = True
    if updated:
        spec['visual_assets_manifest']['assets'] = assets
        spec_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding='utf-8')
        print('Updated design_spec with PNG assets.')
    else:
        print('No SVG assets to convert.')

if __name__ == '__main__':
    main()
