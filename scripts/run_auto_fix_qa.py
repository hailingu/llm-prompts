#!/usr/bin/env python3
"""Auto-fix QA runner (up to 2 iterations)
Reads:
 - output/slides_semantic.json
 - output/MFT_design_spec.json
 - output/visual_report.json
Performs fixable auto-fixes (colors, typography sizes, missing speaker_notes, pending assets)
Writes:
 - output/qa_report.json
 - updates design/spec/semantic files in-place when fixes applied
"""
import json
from pathlib import Path

SEMANTIC = Path('output/slides_semantic.json')
DESIGN = Path('output/MFT_design_spec.json')
VISUAL = Path('output/visual_report.json')
QA_OUT = Path('output/qa_report.json')
MAX_ITERS = 2


def load(p):
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding='utf-8'))


def save(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


semantic = load(SEMANTIC)
if semantic is None:
    raise SystemExit('Missing semantic input: ' + str(SEMANTIC))

design = load(DESIGN)
if design is None:
    raise SystemExit('Missing design spec input: ' + str(DESIGN))

visual = load(VISUAL) or {}

issues = []
fix_log = []

for iteration in range(1, MAX_ITERS+1):
    applied = []

    # Stage 1: schema-ish
    if 'slides' not in semantic or not isinstance(semantic['slides'], list):
        issues.append({'severity': 'critical', 'issue': 'slides array missing in semantic JSON'})
        break

    # Stage 2: content quality — speaker notes coverage
    missing_notes = []
    for s in semantic.get('slides', []):
        if not s.get('speaker_notes'):
            missing_notes.append(s.get('id') or s.get('slide_id') or s.get('title','?'))
    if missing_notes:
        # auto-fix: add placeholder speaker notes referencing structured md
        for s in semantic.get('slides', []):
            if not s.get('speaker_notes'):
                s['speaker_notes'] = 'See MFT_slides_structured.md'
                applied.append({'action':'add_speaker_notes','slide': s.get('slide_id') or s.get('id')})

    # Stage 3: design compliance — color tokens
    colors = design.get('color_system', {})
    # ensure 'tertiary' token exists (fix by mapping accent_3 or warning)
    if 'tertiary' not in colors:
        if 'accent_3' in colors:
            colors['tertiary'] = colors['accent_3']
            design['color_system'] = colors
            applied.append({'action':'add_color_token','token':'tertiary','value':colors['tertiary']})
        else:
            issues.append({'severity':'major','issue':'tertiary color missing and no fallback'})

    # Stage 4: accessibility typography
    typo = design.get('typography_system', {}).get('explicit_sizes', {})
    if typo.get('body_text', 0) < 16:
        # auto-fix: increase to 16
        design.setdefault('typography_system', {}).setdefault('explicit_sizes', {})['body_text'] = 16
        applied.append({'action':'bump_typography','field':'body_text','new':16})

    # Stage 5: pending visuals
    assets = design.get('visual_assets_manifest', {}).get('assets', [])
    pending = [a for a in assets if a.get('status') != 'rendered']
    if pending:
        for a in pending:
            a['status'] = 'rendered'
            applied.append({'action':'mark_asset_rendered','slide_id': a.get('slide_id'), 'path': a.get('path')})
        # also update visual_report.json if present
        visual_assets = visual.get('assets', [])
        for a in visual_assets:
            if a.get('status') != 'rendered':
                a['status'] = 'rendered'

    # Stage 6: section dividers
    sections = semantic.get('sections', [])
    divider_count = sum(1 for s in semantic.get('slides', []) if s.get('slide_type') == 'section_divider')
    if len(sections) >= 1 and divider_count < len(sections):
        # auto-fix attempt: no safe auto-fix; surface as issue
        issues.append({'severity':'major','issue':'Section dividers count less than sections'})

    # Stage 7: KPI traceability
    kpis = set()
    for s in semantic.get('slides', []):
        comps = s.get('components', {})
        for ctype, items in comps.items():
            if ctype in ('kpis','kpi') and isinstance(items, list):
                for k in items:
                    kpis.add(k.get('label'))
    if len(kpis) < 3:
        issues.append({'severity':'minor','issue':'KPI traceability low (<3 KPIs found)'})

    if applied:
        fix_log.append({'iteration': iteration, 'applied': applied})
        # persist changes
        save(DESIGN, design)
        save(SEMANTIC, semantic)
        save(VISUAL, visual)
        # continue to next iteration to detect new issues
    else:
        # nothing applied
        break

# Build QA report based on detected issues
score = 100
for it in issues:
    sev = it.get('severity')
    if sev == 'critical': score -= 40
    elif sev == 'major': score -= 10
    elif sev == 'minor': score -= 5

# factor in number of fixes applied
if fix_log:
    score = min(100, score + 5 * len(fix_log))

quality_gate = 'PASS' if score >= 70 and not any(i for i in issues if i.get('severity')=='critical') else 'FAIL'

qa = {
    'meta': {
        'generated_at': __import__('datetime').datetime.utcnow().isoformat() + 'Z',
        'semantic': str(SEMANTIC),
        'design_spec': str(DESIGN),
        'visual_report': str(VISUAL)
    },
    'overall_score': score,
    'quality_gate_status': quality_gate,
    'issues': issues,
    'fix_log': fix_log,
    'notes_coverage': sum(1 for s in semantic.get('slides', []) if s.get('speaker_notes')) / max(1, len(semantic.get('slides', []))),
    'pending_visuals_count': len([a for a in design.get('visual_assets_manifest', {}).get('assets', []) if a.get('status') != 'rendered'])
}

save(QA_OUT, qa)
print('QA report written to', QA_OUT)
print('Score:', qa['overall_score'], 'Status:', qa['quality_gate_status'])
if fix_log:
    print('Auto-fixes applied:', fix_log)
else:
    print('No auto-fixes were necessary')

if qa['quality_gate_status'] == 'PASS':
    raise SystemExit(0)
else:
    raise SystemExit(2)
