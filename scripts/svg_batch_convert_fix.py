#!/usr/bin/env python3
from pathlib import Path
from cairosvg import svg2png
from PIL import Image, ImageDraw, ImageFont
import json
spec_path = Path('output/MFT_design_spec.json')
spec = json.loads(spec_path.read_text(encoding='utf-8'))
assets = spec.get('visual_assets_manifest', {}).get('assets', [])
output_folder = Path(spec.get('visual_assets_manifest', {}).get('output_folder', 'docs/presentations/mft-20260206/images'))
font = None
try:
    font = ImageFont.truetype('DejaVuSans-Bold.ttf', 24)
except Exception:
    font = ImageFont.load_default()

for a in assets:
    p = Path(a['path'])
    full = output_folder / p.name
    if full.suffix.lower() == '.svg':
        pngp = full.with_suffix('.png')
        try:
            svg2png(url=str(full), write_to=str(pngp))
            print('Converted', full, '->', pngp)
            a['path'] = str(Path(a['path']).with_suffix('.png'))
            a['format'] = 'png'
            a['status'] = 'rendered'
        except Exception as e:
            print('Failed to convert', full, '->', e)
            # create placeholder PNG
            w,h = a.get('dimensions', [1600,900])
            img = Image.new('RGB', (w,h), color=(245,245,245))
            d = ImageDraw.Draw(img)
            title = f"Preview unavailable: {p.stem}"
            try:
                # Pillow 9.2+ supports textbbox
                bbox = d.textbbox((0,0), title, font=font)
                tw = bbox[2]-bbox[0]
                th = bbox[3]-bbox[1]
            except Exception:
                try:
                    tw,th = font.getsize(title)
                except Exception:
                    tw,th = (len(title)*8, 16)
            d.text(((w-tw)/2, h*0.45), title, fill=(40,40,40), font=font)
            img.save(pngp)
            a['path'] = str(Path(a['path']).with_suffix('.png'))
            a['format'] = 'png'
            a['status'] = 'placeholder_generated'
            print('Wrote placeholder', pngp)

# write back spec
spec['visual_assets_manifest']['assets'] = assets
spec_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding='utf-8')
print('Spec updated with PNGs/placeholders')
