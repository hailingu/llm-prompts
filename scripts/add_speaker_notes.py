#!/usr/bin/env python3
import json
from pathlib import Path

files = [Path('output/slides_semantic.json'), Path('output/MFT_slides_semantic.json')]
for f in files:
    data = json.loads(f.read_text(encoding='utf-8'))
    updated = False
    for s in data.get('slides', []):
        if not s.get('speaker_notes'):
            s['speaker_notes'] = 'See MFT_slides_structured.md'
            updated = True
    if updated:
        f.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'Updated {f}')
    else:
        print(f'No change needed for {f}')
