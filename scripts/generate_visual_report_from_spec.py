#!/usr/bin/env python3
import json
from pathlib import Path
spec_path = Path('output/MFT_design_spec.json')
spec = json.loads(spec_path.read_text(encoding='utf-8'))
assets = spec.get('visual_assets_manifest', {}).get('assets', [])
report = {'generated_at': __import__('datetime').datetime.utcnow().isoformat()+'Z', 'assets': []}
for a in assets:
    slide_id = a.get('slide_id')
    path = 'docs/presentations/mft-20260206/images/' + Path(a.get('path')).name
    report['assets'].append({'slide_id': slide_id, 'path': path, 'dimensions': a.get('dimensions', []), 'format': a.get('format', 'png'), 'status': a.get('status')})
report['summary'] = {'total': len(report['assets']), 'rendered': sum(1 for a in report['assets'] if a['status'] in ('rendered','placeholder_generated','png')), 'pending': sum(1 for a in report['assets'] if a['status'] not in ('rendered','placeholder_generated','png'))}
out1 = Path('output/visual_report.json')
out2 = Path('docs/presentations/mft-20260206/visual_report.json')
out1.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
out2.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print('Wrote visual_report to', out1, 'and', out2)
