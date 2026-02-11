"""EA smoke prototype (rule-based) — conservative, auditable transformations.

Outputs:
- docs/presentations/storage-frontier-20260211/slides_semantic_v2.json
- docs/presentations/storage-frontier-20260211/ea_audit.json

Heuristics:
- Add `assertion` for most content slides by extracting title/bullet/speaker_notes.
- Add `insight` as a short actionable sentence when possible.
- Merge adjacent short slides within same section up to compression threshold.
- Add `layout_intent.regions` when a slide has > 2 components.
- Produce `ea_audit.json` with changes and confidence scores.
"""
import json
import os
import re
from pathlib import Path

BASE = Path('docs/presentations/storage-frontier-20260211')
SEM_IN = BASE / 'slides_semantic.json'
SEM_OUT = BASE / 'slides_semantic_v2.json'
AUDIT = BASE / 'ea_audit.json'

MERGE_WORD_THRESHOLD = 200
MAX_COMPRESSION_RATIO = 0.65  # recommended auto-accept threshold


def short_text(s):
    if not s:
        return ''
    s = re.sub(r"\s+", ' ', s).strip()
    return s


def extract_assertion(sd):
    # Prefer concise title if looks like assertion (short and no trailing punctuation)
    title = sd.get('title','')
    if title and len(title.split()) <= 10 and not re.search(r'[?！!。]', title):
        return title, 'title', 0.9
    # else, use first bullet or kpi label
    bullets = sd.get('components', {}).get('bullets') or sd.get('components', {}).get('callouts')
    if bullets and len(bullets) > 0:
        b = bullets[0]
        if isinstance(b, dict):
            text = b.get('text') or b.get('label')
        else:
            text = b
        if text:
            text = short_text(text)
            # extract first 10 words
            words = text.split()[:10]
            return ' '.join(words), 'bullet[0]', 0.8
    # speaker notes fallback
    sn = sd.get('speaker_notes','')
    if sn:
        txt = short_text(sn).split('.')
        if txt:
            first = txt[0]
            words = first.split()[:10]
            return ' '.join(words), 'speaker_notes', 0.5
    return None, None, 0.0


def extract_insight(sd):
    # Simple heuristic: if assertion exists, create 'Consider ...' insight
    assertion = sd.get('assertion')
    if assertion:
        return f"{assertion.split()[0:6]}".replace("[","").replace("]","")
    # else fallback using first bullet
    bullets = sd.get('components', {}).get('bullets')
    if bullets and len(bullets) > 0:
        text = bullets[0]
        if isinstance(text, dict):
            text = text.get('text') or text.get('label')
        text = short_text(text)
        words = text.split()[:8]
        if words:
            return ' '.join(words)
    return None


def word_count_slide(sd):
    cnt = 0
    for t in sd.get('content', []) or []:
        cnt += len(short_text(t).split())
    for comps in sd.get('components', {}).values():
        if isinstance(comps, list):
            for c in comps:
                if isinstance(c, dict):
                    for v in c.values():
                        if isinstance(v, str):
                            cnt += len(v.split())
                elif isinstance(c, str):
                    cnt += len(c.split())
    if sd.get('title'):
        cnt += len(sd.get('title').split())
    if sd.get('speaker_notes'):
        cnt += len(short_text(sd.get('speaker_notes')).split())
    return cnt


def merge_slides(slides):
    # conservative merging: merge adjacent slides if combined word_count <= MERGE_WORD_THRESHOLD and same section
    merged = []
    i = 0
    merges = []
    while i < len(slides):
        cur = slides[i]
        j = i+1
        if j < len(slides):
            nxt = slides[j]
            same_section = True
            if cur.get('section') != nxt.get('section'):
                same_section = False
            if same_section:
                wc = word_count_slide(cur) + word_count_slide(nxt)
                if wc <= MERGE_WORD_THRESHOLD:
                    # merge
                    merged_slide = dict(cur)
                    # combine content and components
                    merged_slide['content'] = (cur.get('content',[]) or []) + (nxt.get('content',[]) or [])
                    comps = dict(cur.get('components', {}))
                    for k,v in (nxt.get('components', {}) or {}).items():
                        comps.setdefault(k, [])
                        comps[k].extend(v if isinstance(v, list) else [v])
                    merged_slide['components'] = comps
                    merged_slide['merged_from'] = [cur.get('id'), nxt.get('id')]
                    merged.append(merged_slide)
                    merges.append((cur.get('id'), nxt.get('id')))
                    i += 2
                    continue
        # no merge
        merged.append(cur)
        i += 1
    return merged, merges


def main():
    semantic = json.load(open(SEM_IN, encoding='utf-8'))
    slides = semantic.get('slides', [])
    sections = semantic.get('sections', [])
    # Annotate section id in slides for merging logic
    # Build slide id -> section mapping using sections' start_slide heuristic
    slide_to_section = {}
    sec_list = sections
    for idx, sd in enumerate(slides, start=1):
        # find section by start_slide
        sec = None
        for s in sec_list:
            start = s.get('start_slide', 1)
            if idx >= start:
                sec = s
        if sec:
            slides[idx-1]['section'] = sec.get('id')

    audit = {'changes': [], 'merges': []}

    # Extract assertions & insights conservatively
    for sd in slides:
        assertion, source, conf = extract_assertion(sd)
        if assertion and not sd.get('assertion'):
            sd['assertion'] = assertion
            audit['changes'].append({'slide_id': sd.get('id'),'field': 'assertion','source': source,'confidence': conf})
        insight = extract_insight(sd)
        if insight and not sd.get('insight'):
            sd['insight'] = insight if insight.startswith('\uF0A1') or insight.startswith('\ud83d') else f"💡 {insight}"
            audit['changes'].append({'slide_id': sd.get('id'),'field': 'insight','confidence': 0.6})
        # add layout_intent when >2 components
        if len(sd.get('components', {})) > 2 and not sd.get('layout_intent'):
            sd['layout_intent'] = {'template': 'two-column', 'regions': [{'id': 'left', 'renderer': 'visual', 'data_source': 'visual'}, {'id': 'right', 'renderer': 'components', 'data_source': 'components'}]}
            audit['changes'].append({'slide_id': sd.get('id'),'field': 'layout_intent','confidence': 0.8})

    # Attempt merging passes until compression ratio target or no merges
    orig_count = len(slides)
    merged_slides, merges = merge_slides(slides)
    # perform iterative merges until acceptable
    while True:
        new_count = len(merged_slides)
        compression_ratio = new_count / orig_count
        if compression_ratio <= MAX_COMPRESSION_RATIO or new_count == len(slides):
            break
        slides = merged_slides
        merged_slides, new_merges = merge_slides(slides)
        merges.extend(new_merges)
        slides = merged_slides
        if len(new_merges) == 0:
            break

    # Finalize
    final_slides = merged_slides
    out = dict(semantic)
    out['slides'] = final_slides
    with open(SEM_OUT, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    audit['merges'] = merges
    audit['summary'] = {
        'orig_slides': orig_count,
        'final_slides': len(final_slides),
        'compression_ratio': len(final_slides) / orig_count,
        'assertion_coverage': sum(1 for s in final_slides if s.get('assertion')) / len(final_slides)
    }
    with open(AUDIT, 'w', encoding='utf-8') as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)
    print('EA smoke run complete. Outputs:')
    print(' -', SEM_OUT)
    print(' -', AUDIT)

if __name__ == '__main__':
    main()
