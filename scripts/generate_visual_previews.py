#!/usr/bin/env python3
"""Generate preview PNG placeholders for visual assets described in design_spec.json."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import json

SPEC_PATH = Path('output/MFT_design_spec.json')
OUT_DIR = Path('docs/presentations/mft-20260206/images')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Simple font fallback
try:
    FONT = ImageFont.truetype('DejaVuSans-Bold.ttf', 24)
    FONT_L = ImageFont.truetype('DejaVuSans.ttf', 18)
except Exception:
    FONT = ImageFont.load_default()
    FONT_L = ImageFont.load_default()

with open(SPEC_PATH, encoding='utf-8') as f:
    spec = json.load(f)

assets = spec.get('visual_assets_manifest', {}).get('assets', [])
report = {'assets': []}

for asset in assets:
    path = OUT_DIR / Path(asset['path']).name
    w, h = asset.get('dimensions', [1600,900])
    fmt = 'JPEG' if path.suffix.lower() in ('.jpg', '.jpeg') else 'PNG'

    img = Image.new('RGB', (w,h), color=(250,250,250))
    draw = ImageDraw.Draw(img)

    # Background gradient-ish
    for i in range(h):
        ratio = i / h
        r = int(245 - 20*ratio)
        g = int(250 - 40*ratio)
        b = int(255 - 60*ratio)
        draw.line([(0,i),(w,i)], fill=(r,g,b))

    title = f"Slide {asset.get('slide_id') if asset.get('slide_id')!=0 else 'Cover'} — {asset.get('source_type')}"
    subtitle = asset.get('purpose', '')

    # big title
    tw, th = draw.textsize(title, font=FONT)
    draw.text(((w-tw)/2, h*0.35), title, font=FONT, fill=(20,30,48))
    sw, sh = draw.textsize(subtitle, font=FONT_L)
    draw.text(((w-sw)/2, h*0.35+th+10), subtitle, font=FONT_L, fill=(60,70,90))

    # small mock chart area
    box_w = int(w*0.6)
    box_h = int(h*0.25)
    box_x = int((w-box_w)/2)
    box_y = int(h*0.6)
    draw.rectangle([box_x,box_y,box_x+box_w,box_y+box_h], outline=(100,110,140), width=2)
    # simple mock lines
    for s in range(3):
        points = [(box_x + i*(box_w//8), box_y + box_h - ((i*10 + s*20) % box_h)) for i in range(9)]
        draw.line(points, fill=(40+s*40, 120, 180), width=3)

    # Save
    img.save(path, format=fmt)

    asset_entry = {
        'slide_id': asset.get('slide_id'),
        'path': str(path),
        'dimensions': [w,h],
        'format': fmt,
        'status': 'generated'
    }
    report['assets'].append(asset_entry)

# Write visual_report.json
REPORT_PATH = Path('output/visual_report.json')
with open(REPORT_PATH, 'w', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print('Generated', len(report['assets']), 'preview assets. Report:', REPORT_PATH)
