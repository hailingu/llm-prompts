#!/usr/bin/env python3
"""Exhibit Architect (EA) — Rule-based v1 → v2 Schema Transform.

Transforms a v1 slides_semantic.json into v2 by:
  1. Extracting assertion titles (So What?) from bullets/notes
  2. Merging low-density adjacent pages within the same section
  3. Annotating insights from speaker_notes
  4. Designing layout_intent (regions) based on slide_type + components
  5. Upgrading visual types where appropriate

Usage:
    python scripts/exhibit_architect.py [--input FILE] [--output FILE] [--no-merge]
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Maximum pages after merging (target)
TARGET_MAX_PAGES = 15

# Minimum content items to avoid merge (a page with fewer is merge-candidate)
MIN_DENSITY_ITEMS = 2

# Merge budget: at most 25% slides can be removed by merge in one run.
# This keeps pacing close to source narrative and avoids over-compression.
MAX_MERGE_RATIO = 0.25

# Slide types that should not be merged with adjacent content slides.
# These types carry strong narrative/visual intent and are easy to degrade
# when merged into generic containers.
MERGE_PROTECTED_TYPES = {
    'decision',
    'comparison',
    'timeline',
    'matrix',
    'call_to_action',
}


def _text_overlap(a: str, b: str) -> bool:
    """Check if two strings have high character overlap (>80%), indicating duplication."""
    if not a or not b:
        return False
    if a == b:
        return True
    # Normalize: strip whitespace and compare
    a_norm = a.strip()
    b_norm = b.strip()
    if a_norm == b_norm:
        return True
    # Check if one contains the other
    if a_norm in b_norm or b_norm in a_norm:
        return True
    # Character-level overlap ratio
    shorter = min(len(a_norm), len(b_norm))
    if shorter == 0:
        return False
    common = sum(1 for c in a_norm if c in b_norm)
    return common / shorter > 0.80


# ---------------------------------------------------------------------------
# Step 1: Assertion Extraction
# ---------------------------------------------------------------------------

def _extract_assertion_from_bullets(bullets: List[str]) -> str:
    """Pick the most assertive bullet or synthesize from multiple."""
    if not bullets:
        return ''
    # Prefer bullets with numbers/percentages (more assertive)
    scored: List[Tuple[int, str]] = []
    for b in bullets:
        score = 0
        if re.search(r'\d+', b):
            score += 2
        if re.search(r'%|USD|亿|万|Bn', b):
            score += 3
        if any(kw in b for kw in ('增长', '下降', '领先', '超过', '达到', '提升', '主导',
                                   '快速', '显著', '关键', '核心', '趋势', '风险')):
            score += 2
        if len(b) <= 60:
            score += 1  # prefer concise
        scored.append((score, b))
    scored.sort(key=lambda x: -x[0])
    best = scored[0][1]
    # Trim to max 80 chars
    if len(best) > 80:
        best = best[:77] + '…'
    return best


def _merge_items_by_base_label(items: List[Dict]) -> List[Dict]:
    """Merge comparison items that share a base label but have disjoint attributes.

    Example: ["AWS 架构"{亮点,场景}, "AWS 试点"{建议,风险}] → ["AWS"{亮点,场景,建议,风险}]

    Only merges when:
    - There are >= 4 items
    - Label suffix grouping produces exactly 2 groups
    - Each base label appears in both suffix groups
    - Attribute keys between suffix groups have zero overlap
    """
    if len(items) < 4:
        return items

    import re

    # Extract suffix from each label
    suffixes: Dict[str, List[int]] = {}
    base_labels: List[str] = []
    for i, it in enumerate(items):
        label = it.get('label', '').strip()
        m = re.search(r'[\s_]([^\s_]+)$', label)
        if m:
            suffix = m.group(1)
            base = label[:m.start()].strip()
        else:
            suffix = label
            base = label
        suffixes.setdefault(suffix, []).append(i)
        base_labels.append(base)

    # Need exactly 2 suffix groups
    if len(suffixes) != 2:
        return items

    groups = list(suffixes.values())
    # Check attribute key disjointness between the two groups
    keys_a = set(k for idx in groups[0] for k in (items[idx].get('attributes') or {}).keys())
    keys_b = set(k for idx in groups[1] for k in (items[idx].get('attributes') or {}).keys())
    if not keys_a or not keys_b:
        return items
    if keys_a & keys_b:  # any overlap → not a merge candidate
        return items

    # Check each base label appears in both groups
    bases_a = set(base_labels[i] for i in groups[0])
    bases_b = set(base_labels[i] for i in groups[1])
    if bases_a != bases_b:
        return items

    # Build merged items: one per base label, combining attributes
    # Use group[0] order for stable output
    merged = []
    base_to_indices: Dict[str, List[int]] = {}
    for i, base in enumerate(base_labels):
        base_to_indices.setdefault(base, []).append(i)

    seen_bases: List[str] = []
    for idx in groups[0]:
        base = base_labels[idx]
        if base in seen_bases:
            continue
        seen_bases.append(base)
        combined_attrs = {}
        for j in base_to_indices[base]:
            for k, v in (items[j].get('attributes') or {}).items():
                if v:  # only non-empty
                    combined_attrs[k] = v
        merged.append({'label': base, 'attributes': combined_attrs})

    return merged


def _upgrade_visual_from_placeholder(sd: Dict) -> None:
    """If a slide's visual contains placeholder chart data, prefer a native chart type.

    Mutates `sd['visual']['type']` when placeholder_data is present with an inferred chart type.
    """
    vis = sd.get('visual', {})
    if not isinstance(vis, dict):
        return
    vtype = vis.get('type', 'none')
    placeholder = vis.get('placeholder_data') or {}
    chart_cfg = placeholder.get('chart_config') or placeholder.get('chart_config', {})

    # Heuristic: if chart_config exists or there is 'series' data, infer chart type
    if chart_cfg or placeholder.get('series') or placeholder.get('categories'):
        # Prefer explicit chart_config.type
        inferred = chart_cfg.get('type') if isinstance(chart_cfg, dict) else None
        if not inferred:
            series = placeholder.get('series', [])
            if series and len(series) > 1:
                inferred = 'composite_charts'
            else:
                # fallback: if x values present and many points -> line, else bar
                cats = placeholder.get('categories') or []
                if cats and len(cats) > 6:
                    inferred = 'line_chart'
                else:
                    inferred = 'bar_chart'
        # Only upgrade when current type is none/png/image or generic placeholder
        if vtype in ('none', 'png', 'image', 'placeholder') or vtype.startswith('placeholder'):
            vis['type'] = inferred
            # persist chart_config if available
            if chart_cfg:
                vis.setdefault('chart_config', chart_cfg)
            sd['visual'] = vis


def _merge_adjacent_single_component_slides(slides: List[Dict]) -> List[Dict]:
    """Merge adjacent slides when both are single-component and combinable.

    Rules:
    - KPI only + Chart only -> KPI top, Chart bottom (three-region-top or two-region)
    - Chart only + Bullets only -> Chart left, Bullets right (two-region-split)
    - Bullets only + Bullets only -> concatenate bullets

    Preserves order and section labels. Returns a new slide list.
    """
    out: List[Dict] = []
    i = 0
    n = len(slides)
    while i < n:
        cur = slides[i]
        # skip titles/section dividers
        if cur.get('slide_type') in ('title', 'section_divider'):
            out.append(cur)
            i += 1
            continue
        # look ahead one slide
        if i + 1 < n:
            nxt = slides[i + 1]
            # same section required
            if cur.get('_section_label') == nxt.get('_section_label') and nxt.get('slide_type') not in ('title', 'section_divider'):
                comps_cur = [k for k, v in cur.get('components', {}).items() if isinstance(v, list) and v]
                comps_nxt = [k for k, v in nxt.get('components', {}).items() if isinstance(v, list) and v]
                vis_cur = cur.get('visual', {}).get('type', 'none') != 'none'
                vis_nxt = nxt.get('visual', {}).get('type', 'none') != 'none'

                # KPI only + chart only
                if comps_cur == ['kpis'] and vis_nxt and len(comps_nxt) == 0:
                    merged = dict(cur)
                    merged_components = merged.setdefault('components', {})
                    merged_components['kpis'] = cur.get('components', {}).get('kpis', [])
                    merged['visual'] = nxt.get('visual')
                    merged['slide_type'] = 'data-heavy'
                    out.append(merged)
                    i += 2
                    continue

                # chart only + bullets only
                if vis_cur and comps_nxt == ['bullets'] and len(comps_cur) == 0:
                    merged = dict(cur)
                    merged_components = merged.setdefault('components', {})
                    merged_components['bullets'] = nxt.get('components', {}).get('bullets', [])
                    merged['slide_type'] = 'data-heavy'
                    out.append(merged)
                    i += 2
                    continue

                # bullets + bullets -> concat
                if comps_cur == ['bullets'] and comps_nxt == ['bullets']:
                    merged = dict(cur)
                    merged_components = merged.setdefault('components', {})
                    merged_components['bullets'] = (cur.get('components', {}).get('bullets', []) or []) + (nxt.get('components', {}).get('bullets', []) or [])
                    out.append(merged)
                    i += 2
                    continue
        # default: push current
        out.append(cur)
        i += 1
    return out


def _split_comparison_by_attrs(items: List[Dict], overlap_threshold: float = 0.30) -> Optional[tuple]:
    """Detect if comparison items naturally split into two groups with low attribute overlap.

    Returns (group_a, group_b) if a clean split exists, else None.
    A clean split means the attribute-key overlap between the two groups is < overlap_threshold.

    Heuristic: try splitting by label suffix patterns (e.g. 架构 vs 试点) or by
    attribute-set clustering.
    """
    if len(items) < 4:
        return None

    # Strategy 1: detect label-suffix grouping (common pattern: "XXX 架构" / "XXX 试点")
    import re
    suffix_groups: Dict[str, List[int]] = {}
    for i, it in enumerate(items):
        label = it.get('label', '')
        # Extract suffix after last space/underscore
        m = re.search(r'[\s_]([^\s_]+)$', label.strip())
        suffix = m.group(1) if m else label.strip()
        suffix_groups.setdefault(suffix, []).append(i)

    # Need exactly 2 groups for a clean split
    if len(suffix_groups) == 2:
        groups = list(suffix_groups.values())
        g_a = [items[i] for i in groups[0]]
        g_b = [items[i] for i in groups[1]]
        keys_a = set(k for it in g_a for k in (it.get('attributes') or {}).keys()
                     if (it.get('attributes') or {}).get(k))
        keys_b = set(k for it in g_b for k in (it.get('attributes') or {}).keys()
                     if (it.get('attributes') or {}).get(k))
        if keys_a and keys_b:
            overlap = len(keys_a & keys_b) / len(keys_a | keys_b)
            if overlap < overlap_threshold:
                return (g_a, g_b)

    # Strategy 2: cluster by non-empty attribute keys
    attr_sets = []
    for it in items:
        keys = set(k for k, v in (it.get('attributes') or {}).items() if v)
        attr_sets.append(keys)

    # Try all 2-way splits preserving order
    n = len(items)
    best_split = None
    best_overlap = 1.0
    for split_at in range(2, n - 1):
        keys_a = set(k for s in attr_sets[:split_at] for k in s)
        keys_b = set(k for s in attr_sets[split_at:] for k in s)
        if keys_a and keys_b:
            overlap = len(keys_a & keys_b) / len(keys_a | keys_b)
            if overlap < best_overlap:
                best_overlap = overlap
                best_split = split_at

    if best_split is not None and best_overlap < overlap_threshold:
        return (items[:best_split], items[best_split:])

    return None


def _extract_assertion_from_comparison(items: List[Dict], notes: str = '') -> str:
    """Synthesize an assertion from comparison items and notes.

    Priority:
    1. Speaker notes that describe the comparison (contain 对比/差异/vs keywords)
    2. Deduplicated entity names from labels + attribute highlights
    3. Fallback to count-based template
    """
    if not items:
        return ''

    # 1. Notes-first: if notes describe the comparison, use that
    if notes:
        first = re.split(r'[。；;.\n]', notes)[0].strip()
        if any(kw in first for kw in ('对比', '對比', '差异', '差異', '优势',
                                       '優勢', '选型', '選型', '比较', '比較',
                                       'vs', 'VS', '策略')):
            if len(first) <= 80:
                return first

    # 2. Deduplicate entity names from labels
    #    e.g. ["AWS 架构", "AWS 试点", "Dell 架构", "Dell 试点"] → ["AWS", "Dell"]
    labels = [it.get('label', '') for it in items]
    unique_entities: List[str] = []
    seen: set = set()
    for lbl in labels:
        entity = lbl.split()[0] if lbl else ''
        if entity and entity not in seen:
            unique_entities.append(entity)
            seen.add(entity)

    # If dedup was too aggressive (< 2 entities from 2+ labels), use full labels
    if len(unique_entities) < 2 and len(labels) >= 2:
        unique_entities = labels

    # Try to find a key differentiator from attributes
    differentiator = ''
    for it in items:
        attrs = it.get('attributes', {})
        if isinstance(attrs, dict):
            for key in ('亮点', '亮點', '特点', '特點', 'highlight', '场景', '場景'):
                val = attrs.get(key, '')
                if val and len(val) <= 40 and not differentiator:
                    differentiator = val

    if len(unique_entities) == 2:
        base = f"{unique_entities[0]} vs {unique_entities[1]}"
        return f"{base}：关键差异与选型建议"
    elif len(unique_entities) == 3:
        names = '、'.join(unique_entities[:3])
        return f"{names}：多方案对比与适用场景"
    elif len(unique_entities) >= 4:
        names = '、'.join(unique_entities[:3])
        return f"{names} 等 {len(unique_entities)} 方案对比与选型分析"
    else:
        return f"{len(labels)} 项方案对比与选型分析"


def _extract_assertion_from_notes(notes: str) -> str:
    """Extract a concise assertion from speaker notes."""
    if not notes:
        return ''

    def first_sentence(text: str, max_len: int = 80) -> str:
        if not text:
            return ''
        s = re.split(r'[。；;\.\n]', text)[0].strip()
        if len(s) > max_len:
            s = s[: max_len - 3] + '…'
        return s

    def extract_field(notes_text: str, field: str) -> str:
        """Extract a structured field from notes like 'Summary: ... Action: ...'."""
        if not notes_text:
            return ''
        text = ' '.join(str(notes_text).replace('\r', '\n').split())
        # Common markers in our pipeline
        markers = {
            'summary': ['Summary:', 'Summary：', '总结:', '总结：'],
            'rationale': ['Rationale:', 'Rationale：', '原因:', '原因：', '逻辑:', '逻辑：'],
            'evidence': ['Evidence:', 'Evidence：', '证据:', '证据：'],
            'action': ['Action:', 'Action：', '行动:', '行动：', '建议:', '建议：'],
            'risks': ['Risks:', 'Risks：', '风险:', '风险：'],
        }
        key = field.lower().strip()
        keys = markers.get(key, [])
        if not keys:
            return ''

        # Locate the first occurrence of any start marker
        starts = []
        lower = text.lower()
        for mk in keys:
            idx = lower.find(mk.lower())
            if idx >= 0:
                starts.append((idx, mk))
        if not starts:
            return ''
        start_idx, start_mk = sorted(starts, key=lambda x: x[0])[0]
        value_start = start_idx + len(start_mk)

        # Find the nearest next marker (any field) after value_start
        next_positions = []
        for mk_list in markers.values():
            for mk in mk_list:
                j = lower.find(mk.lower(), value_start)
                if j >= 0:
                    next_positions.append(j)
        value_end = min(next_positions) if next_positions else len(text)
        val = text[value_start:value_end].strip(" -：:；;。\n\t")
        return val

    # Prefer the explicit Summary section; fall back to first sentence.
    summary = extract_field(notes, 'summary')
    if summary:
        return first_sentence(summary)
    return first_sentence(notes)


def _extract_insight_from_notes(notes: str) -> str:
    """Extract a compact, actionable insight from structured speaker notes."""
    if not notes:
        return ''

    def first_sentence(text: str, max_len: int = 80) -> str:
        if not text:
            return ''
        s = re.split(r'[。；;\.\n]', text)[0].strip()
        if len(s) > max_len:
            s = s[: max_len - 3] + '…'
        return s

    # Reuse the assertion helper's field parser via a small inline copy (kept local for safety)
    text = ' '.join(str(notes).replace('\r', '\n').split())
    lower = text.lower()
    markers = {
        'summary': ['Summary:', 'Summary：', '总结:', '总结：'],
        'rationale': ['Rationale:', 'Rationale：', '原因:', '原因：', '逻辑:', '逻辑：'],
        'evidence': ['Evidence:', 'Evidence：', '证据:', '证据：'],
        'action': ['Action:', 'Action：', '行动:', '行动：', '建议:', '建议：'],
        'risks': ['Risks:', 'Risks：', '风险:', '风险：'],
    }

    def extract_field(field: str) -> str:
        keys = markers.get(field, [])
        if not keys:
            return ''
        starts = []
        for mk in keys:
            idx = lower.find(mk.lower())
            if idx >= 0:
                starts.append((idx, mk))
        if not starts:
            return ''
        start_idx, start_mk = sorted(starts, key=lambda x: x[0])[0]
        value_start = start_idx + len(start_mk)
        next_positions = []
        for mk_list in markers.values():
            for mk in mk_list:
                j = lower.find(mk.lower(), value_start)
                if j >= 0:
                    next_positions.append(j)
        value_end = min(next_positions) if next_positions else len(text)
        return text[value_start:value_end].strip(" -：:；;。\n\t")

    # Prefer Action (what to do), then Rationale (so-what), then Risks.
    for fld in ('action', 'rationale', 'risks'):
        v = extract_field(fld)
        if v:
            return first_sentence(v)
    # Fall back to 2nd sentence of Summary if available
    summary = extract_field('summary')
    if summary:
        sents = [s.strip() for s in re.split(r'[。；;\.\n]', summary) if s.strip()]
        if len(sents) >= 2:
            return first_sentence(sents[1])
        return first_sentence(sents[0])
    return ''


def extract_assertion(sd: Dict) -> str:
    """Extract or generate an assertion title for a slide."""
    # Already has assertion? Keep it.
    if sd.get('assertion'):
        return sd['assertion']

    stype = sd.get('slide_type', '')
    comps = sd.get('components', {})
    bullets = comps.get('bullets', [])
    comparison = comps.get('comparison_items', [])
    notes = sd.get('speaker_notes', '')

    # Skip title and section_divider — they don't need assertions
    if stype in ('title', 'section_divider'):
        return ''

    # Try bullets first
    if bullets:
        result = _extract_assertion_from_bullets(bullets)
        if result:
            return result

    # Try comparison items
    if comparison:
        result = _extract_assertion_from_comparison(comparison, notes)
        if result:
            return result

    # Fallback to speaker notes
    if notes:
        result = _extract_assertion_from_notes(notes)
        if result:
            return result

    return ''


# ---------------------------------------------------------------------------
# Step 2: Page Merging
# ---------------------------------------------------------------------------

def _content_density(sd: Dict) -> int:
    """Count how many renderable content items a slide has."""
    count = 0
    comps = sd.get('components', {})
    for key in ('bullets', 'kpis', 'comparison_items', 'callouts', 'risks'):
        items = comps.get(key)
        if items and isinstance(items, list):
            count += len(items)
    if sd.get('visual', {}).get('type', 'none') != 'none':
        count += 1
    content = sd.get('content', [])
    if content:
        count += len(content)
    return count


def _can_merge(a: Dict, b: Dict) -> bool:
    """Check if two adjacent slides can be merged."""
    a_type = a.get('slide_type', '')
    b_type = b.get('slide_type', '')

    # Don't merge title slides
    if a_type == 'title' or b_type == 'title':
        return False

    # Keep section divider as a pacing/transition page.
    if a_type == 'section_divider' or b_type == 'section_divider':
        return False

    # Protect key slide types from structural merges.
    if a_type in MERGE_PROTECTED_TYPES or b_type in MERGE_PROTECTED_TYPES:
        return False

    # Don't merge if both have charts (too much data)
    a_has_chart = a.get('visual', {}).get('type', 'none') != 'none'
    b_has_chart = b.get('visual', {}).get('type', 'none') != 'none'
    if a_has_chart and b_has_chart:
        return False

    # Don't merge if combined density would be very high
    combined = _content_density(a) + _content_density(b)
    if combined > 10:  # raised from 8 to allow denser merges
        return False

    # Prefer merging low-density pages, but allow same-type merges
    a_density = _content_density(a)
    b_density = _content_density(b)
    if a_density >= 4 and b_density >= 4:
        # Allow same slide_type merges (homogeneous content works well together)
        if a_type != b_type:
            return False

    return True


def _merge_slides(a: Dict, b: Dict) -> Dict:
    """Merge slide b into slide a, combining components.

    Special case: if a is a section_divider, b becomes the primary slide
    and the section info from a is preserved as context.
    """
    if a.get('slide_type') == 'section_divider':
        # Section divider absorbed: b becomes primary, a provides section context
        merged = copy.deepcopy(b)
        merged['_merged_from'] = [a.get('id'), b.get('id')]
        merged['_section_label'] = a.get('title', '')
        # Absorb any callouts from the divider
        a_callouts = a.get('components', {}).get('callouts', [])
        if a_callouts:
            b_callouts = merged.get('components', {}).get('callouts', [])
            merged.setdefault('components', {})['callouts'] = list(a_callouts) + list(b_callouts or [])
        # Absorb content from divider as additional context
        a_content = a.get('content', [])
        if a_content:
            b_content = merged.get('content', [])
            merged['content'] = list(a_content) + list(b_content or [])
        return merged

    # Normal merge: combine two content slides
    merged = copy.deepcopy(a)
    merged['_merged_from'] = [a.get('id'), b.get('id')]

    # Merge components
    a_comps = merged.get('components', {})
    b_comps = b.get('components', {})

    for key in ('bullets', 'kpis', 'comparison_items', 'callouts', 'risks'):
        a_items = a_comps.get(key, [])
        b_items = b_comps.get(key, [])
        if b_items:
            if not a_items:
                a_comps[key] = list(b_items)
            else:
                a_comps[key] = list(a_items) + list(b_items)
    merged['components'] = a_comps

    # Merge content
    a_content = merged.get('content', [])
    b_content = b.get('content', [])
    if b_content:
        merged['content'] = list(a_content) + list(b_content)

    # If b has visual and a doesn't, take b's visual
    if merged.get('visual', {}).get('type', 'none') == 'none' and b.get('visual', {}).get('type', 'none') != 'none':
        merged['visual'] = copy.deepcopy(b.get('visual', {}))

    # Merge speaker notes
    a_notes = merged.get('speaker_notes', '')
    b_notes = b.get('speaker_notes', '')
    if b_notes:
        merged['speaker_notes'] = f"{a_notes}\n{b_notes}" if a_notes else b_notes

    # Update slide_type to the more complex one
    # NOTE: decision/matrix slides rely on v1 per-type renderers today (Track B
    # region renderers do not yet cover decisions/risks). Preserve these types
    # so we don't lose content when a slide later falls back to v1 rendering.
    priority = {
        'decision': 6,
        'data-heavy': 5,
        'matrix': 5,
        'comparison': 4,
        'timeline': 4,
        'technical': 3,
        'flowchart': 3,
        'bullet-list': 2,
    }
    a_pri = priority.get(a.get('slide_type', ''), 1)
    b_pri = priority.get(b.get('slide_type', ''), 1)
    if b_pri > a_pri:
        merged['slide_type'] = b['slide_type']

    # If the merged slide contains decisions/risks, force a compatible type.
    # This avoids downgrading a decision/risk slide into data-heavy, which is
    # prone to missing the primary component in v1 renderers.
    try:
        comps = merged.get('components', {}) or {}
        if isinstance(comps.get('decisions'), list) and comps.get('decisions'):
            merged['slide_type'] = 'decision'
        elif isinstance(comps.get('risks'), list) and comps.get('risks'):
            # Prefer matrix for risk-heavy slides
            merged['slide_type'] = 'matrix'
    except Exception:
        pass

    # Preserve section labels from both sides
    if b.get('_section_label') and not merged.get('_section_label'):
        merged['_section_label'] = b['_section_label']

    return merged


def _find_section_boundary_ids(sections: List[Dict]) -> set:
    """Return set of slide IDs that start a section."""
    return {sec.get('start_slide', 0) for sec in sections}


def merge_pages(slides: List[Dict], sections: List[Dict], enabled: bool = True) -> List[Dict]:
    """Merge low-density adjacent pages within sections.

    Strategy:
    1. Section dividers are absorbed into their first content slide
       (the section label is preserved as _section_label metadata).
    2. Adjacent low-density content slides within a section are merged.
    3. The first section_divider (opening / key conclusions) is kept.
    4. Multi-pass: repeat until no more merges are possible (max 3 passes).
    """
    if not enabled:
        return slides

    current = list(slides)
    original_count = len(current)
    max_merges = max(0, int(original_count * MAX_MERGE_RATIO))
    merge_count = 0

    for _pass in range(3):  # multi-pass: up to 3 iterations
        result: List[Dict] = []
        i = 0
        changed = False

        while i < len(current):
            curr = current[i]

            # Try merging with next slide
            if i + 1 < len(current):
                nxt = current[i + 1]
                if merge_count < max_merges and _can_merge(curr, nxt):
                    merged = _merge_slides(curr, nxt)
                    result.append(merged)
                    i += 2
                    changed = True
                    merge_count += 1
                    continue

            result.append(curr)
            i += 1

        current = result
        if not changed:
            break

    return current


# ---------------------------------------------------------------------------
# Step 3: Insight Annotation
# ---------------------------------------------------------------------------

def extract_insight(sd: Dict) -> str:
    """Extract an insight for the slide from speaker notes or components."""
    # Already has insight? Keep it.
    if sd.get('insight'):
        return sd['insight']

    stype = sd.get('slide_type', '')
    if stype in ('title', 'section_divider'):
        return ''

    notes = sd.get('speaker_notes', '')
    comps = sd.get('components', {})

    # Prefer structured Action/Rationale/Risks fields in speaker notes
    if notes:
        structured = _extract_insight_from_notes(notes)
        if structured:
            return structured

        # Legacy fallback: look for recommendation/suggestion patterns in raw notes
        patterns = [
            r'建[議议](.{5,60})',
            r'(?:应|應)(.{5,40})',
            r'(?:需要|需)(.{5,40})',
            r'(?:优先|優先)(.{5,40})',
            r'(?:关键|關鍵)(.{5,40})',
        ]
        for pat in patterns:
            m = re.search(pat, notes)
            if m:
                return m.group(0).strip()[:80]

        sentences = re.split(r'[。；;\.\n]', notes)
        sentences = [s.strip() for s in sentences if s.strip()]
        if len(sentences) >= 2:
            return sentences[1][:80]
        elif sentences:
            return sentences[0][:80]

    # Try risks as insight
    risks = comps.get('risks', [])
    if risks:
        r = risks[0]
        return f"风险提示：{r.get('label', '')} — {r.get('mitigation', '')}"[:80]

    return ''


# ---------------------------------------------------------------------------
# Step 3.5: Component Enrichment
# ---------------------------------------------------------------------------

def _synthesize_callouts_from_comparison(sd: Dict) -> List[Dict]:
    """Extract key insights from comparison attributes to create callouts.

    Scans comparison_items' attributes for actionable keys like '建议',
    '风险', '限制', '回退' and converts them into structured callouts.
    Also appends a callout from speaker_notes if available.
    """
    callouts: List[Dict] = []
    items = sd.get('components', {}).get('comparison_items', [])
    notes = sd.get('speaker_notes', '')

    # Extract actionable info from attributes
    action_keys = ('建议', '建議', '风险', '風險', '备注', '備註',
                   '回退', '限制', 'recommendation', 'risk')
    for it in items:
        attrs = it.get('attributes', {})
        if isinstance(attrs, dict):
            for key in action_keys:
                val = attrs.get(key, '')
                if val and len(val) > 5:
                    icon = '⚠' if key in ('风险', '風險', '限制', 'risk') else '💡'
                    callouts.append({
                        'label': f'{icon} {it.get("label", "")}',
                        'text': f'{key}：{val}',
                    })

    # Add a callout from notes if present
    if notes:
        sentences = [s.strip() for s in re.split(r'[。；;.\n]', notes) if s.strip() and len(s.strip()) > 5]
        if sentences:
            callouts.append({
                'label': '💡 关键洞察',
                'text': sentences[0][:80],
            })

    # Limit to 3 most relevant callouts
    return callouts[:3]


def _synthesize_callout_from_notes(sd: Dict) -> List[Dict]:
    """Create callouts from speaker notes for chart-only or sparse slides."""
    notes = sd.get('speaker_notes', '')
    if not notes:
        return []

    callouts: List[Dict] = []
    # Collect assertion / insight to avoid duplicating them as callouts
    assertion = sd.get('assertion', '')
    insight = sd.get('insight', '')
    sentences = [s.strip() for s in re.split(r'[。；;.\n]', notes)
                 if s.strip() and len(s.strip()) > 5]
    for sent in sentences[:4]:  # scan more, filter duplicates
        if assertion and _text_overlap(sent, assertion):
            continue
        if insight and _text_overlap(sent, insight):
            continue
        icon = '⚠' if any(kw in sent for kw in ('风险', '挑战', '限制', '注意')) else '💡'
        callouts.append({
            'label': f'{icon} 关键提示',
            'text': sent[:80],
        })
        if len(callouts) >= 2:
            break
    return callouts


def enrich_components(slides: List[Dict]) -> List[Dict]:
    """Enrich slides that have sparse components with synthesized callouts.

    Targets:
    - Comparison-only slides → add callouts from attributes + notes
    - Chart-only slides → add callouts from notes
    - Any slide with ≤1 component type and no callouts
    """
    for sd in slides:
        stype = sd.get('slide_type', '')
        if stype == 'title':
            continue

        # Special-case: section dividers should not be empty transition pages.
        # If they only have a one-liner callout like "A + B + C", split it into
        # 2-3 bullets so the divider carries "本节要点" without fabricating data.
        if stype == 'section_divider':
            comps = sd.setdefault('components', {}) or {}
            bullets = comps.get('bullets')
            if not (isinstance(bullets, list) and bullets):
                callouts = comps.get('callouts')
                one_liner = ''
                if isinstance(callouts, list) and callouts:
                    one_liner = (callouts[0].get('text') or '').strip()
                if not one_liner and isinstance(sd.get('content'), list) and sd.get('content'):
                    one_liner = str(sd.get('content')[0]).strip()

                # Split by common separators; keep short, distinct phrases.
                parts = [p.strip() for p in re.split(r'[+＋、/|,，；;\n]', one_liner) if p.strip()]
                # Filter out overly long sentences; divider bullets should be scannable.
                parts = [p for p in parts if len(p) <= 24]
                if len(parts) >= 2:
                    comps['bullets'] = parts[:3]
            continue

        # Upgrade visuals from placeholder data to native chart types when possible
        _upgrade_visual_from_placeholder(sd)

        comps = sd.get('components', {})
        has_comparison = isinstance(comps.get('comparison_items'), list) and len(comps.get('comparison_items', [])) > 0
        has_chart = sd.get('visual', {}).get('type', 'none') != 'none'
        has_bullets = isinstance(comps.get('bullets'), list) and len(comps.get('bullets', [])) > 0
        has_callouts = isinstance(comps.get('callouts'), list) and len(comps.get('callouts', [])) > 0
        has_kpis = isinstance(comps.get('kpis'), list) and len(comps.get('kpis', [])) > 0

        # Count component types
        comp_types = sum([has_comparison, has_chart, has_bullets, has_callouts, has_kpis])

        # Merge comparison items that share a base label (e.g. "AWS 架构" + "AWS 试点" → "AWS")
        if has_comparison:
            items_before = comps.get('comparison_items', [])
            merged = _merge_items_by_base_label(items_before)
            if len(merged) < len(items_before):
                comps['comparison_items'] = merged

        # Enrich if only 1 component type and no callouts yet
        if comp_types <= 1 and not has_callouts:
            if has_comparison:
                # When items split into attribute-disjoint groups, all attributes
                # are already fully visible in the sub-tables.  Synthesising
                # callouts from attributes would be 100 % redundant.  Only add
                # a callout from speaker_notes in that case.
                items = comps.get('comparison_items', [])
                split = _split_comparison_by_attrs(items)
                if split:
                    new_callouts = _synthesize_callout_from_notes(sd)
                else:
                    new_callouts = _synthesize_callouts_from_comparison(sd)
                if new_callouts:
                    sd.setdefault('components', {})['callouts'] = new_callouts
            elif has_chart:
                new_callouts = _synthesize_callout_from_notes(sd)
                if new_callouts:
                    sd.setdefault('components', {})['callouts'] = new_callouts

    # After per-slide enrichment, attempt to merge adjacent single-component slides
    slides = _merge_adjacent_single_component_slides(slides)

    return slides


# ---------------------------------------------------------------------------
# Step 4: Layout Intent Design
# ---------------------------------------------------------------------------

def _has_component(sd: Dict, key: str) -> bool:
    items = sd.get('components', {}).get(key, [])
    return isinstance(items, list) and len(items) > 0


def _extract_layout_features(sd: Dict) -> Dict[str, Any]:
    """Extract content/layout features used by strategy selection.

    This keeps layout decisions explainable and reduces brittle branching.
    """
    comps = sd.get('components', {}) or {}
    comparison_items = comps.get('comparison_items') or []
    callouts = comps.get('callouts') or []
    kpis = comps.get('kpis') or []
    bullets = comps.get('bullets') or []
    timeline_items = comps.get('timeline_items') or []
    risks = comps.get('risks') or []

    key_counts: Dict[str, int] = {}
    for it in comparison_items:
        for key in (it.get('attributes') or {}).keys():
            key_counts[key] = key_counts.get(key, 0) + 1

    unique_attr_keys = len(key_counts)
    shared_attr_keys = sum(1 for _, v in key_counts.items() if v >= 2)

    vis = sd.get('visual', {}) or {}
    pd = vis.get('placeholder_data', {}) if isinstance(vis, dict) else {}
    chart_cfg = pd.get('chart_config', {}) if isinstance(pd, dict) else {}
    series = chart_cfg.get('series', []) if isinstance(chart_cfg, dict) else []

    return {
        'slide_type': sd.get('slide_type', ''),
        'comparison_items_count': len(comparison_items),
        'callout_count': len(callouts),
        'kpi_count': len(kpis),
        'bullet_count': len(bullets),
        'timeline_count': len(timeline_items),
        'risk_count': len(risks),
        'unique_attr_keys': unique_attr_keys,
        'shared_attr_keys': shared_attr_keys,
        'is_sparse_comparison': len(comparison_items) <= 3 and unique_attr_keys >= 7 and shared_attr_keys <= 2,
        'chart_series_count': len(series),
    }


def _choose_chart_table_width(features: Dict[str, Any]) -> int:
    """Choose chart width in chart+table composition from features."""
    series_count = int(features.get('chart_series_count', 0) or 0)
    sparse_comp = bool(features.get('is_sparse_comparison'))
    if sparse_comp:
        return 48
    if series_count <= 1:
        return 45
    if series_count >= 3:
        return 55
    return 50


def design_layout_intent(sd: Dict) -> Optional[Dict]:
    """Design a layout_intent based on slide_type and available components."""
    stype = sd.get('slide_type', '')

    # No layout for title / section_divider (handled specially by renderer)
    if stype in ('title', 'section_divider'):
        return None

    has_chart = sd.get('visual', {}).get('type', 'none') != 'none'
    has_kpis = _has_component(sd, 'kpis')
    has_bullets = _has_component(sd, 'bullets')
    has_comparison = _has_component(sd, 'comparison_items')
    # Normalize callout-like keys for layout planning
    comps = sd.get('components', {}) or {}
    if _has_component(sd, 'risk_callouts') and not _has_component(sd, 'callouts'):
        comps['callouts'] = list(comps.get('risk_callouts') or [])
        sd['components'] = comps
    has_callouts = _has_component(sd, 'callouts')
    has_risks = _has_component(sd, 'risks')
    has_flow = sd.get('visual', {}).get('type') == 'flow_diagram'
    content = sd.get('content', [])
    has_content = isinstance(content, list) and len(content) > 0
    features = _extract_layout_features(sd)
    strategy_decision = 'default'

    regions: List[Dict] = []

    # NOTE:
    # Keep Track-B on explicit region layouts whenever possible.
    # Region renderers now support decisions/risks/flow, so avoid forcing
    # v1 fallback for these content types.

    # ------- data-heavy: chart dominant -------
    if stype == 'data-heavy':
        if has_chart and has_kpis:
            if has_bullets:
                # KPI row on top, chart left, bullets right
                regions = [
                    {'id': 'kpis', 'position': 'top-18', 'renderer': 'kpi_row', 'data_source': 'components.kpis'},
                    {'id': 'main_chart', 'position': 'left-65', 'renderer': 'chart', 'data_source': 'visual'},
                    {'id': 'side_bullets', 'position': 'right-35', 'renderer': 'bullet_list', 'data_source': 'components.bullets'},
                ]
                template = 'three-region-top'
            else:
                # KPI row on top, chart filling bottom
                regions = [
                    {'id': 'kpis', 'position': 'top-18', 'renderer': 'kpi_row', 'data_source': 'components.kpis'},
                    {'id': 'main_chart', 'position': 'bottom-80', 'renderer': 'chart', 'data_source': 'visual'},
                ]
                template = 'two-region-split'
        elif has_chart:
            if has_comparison:
                # Chart + comparison table (from merged slides)
                # Check if comparison items split into attribute-disjoint groups
                comp_items = sd['components']['comparison_items']
                split = _split_comparison_by_attrs(comp_items)
                chart_pct = _choose_chart_table_width(features)
                table_pct = 100 - chart_pct
                if split:
                    group_a, group_b = split
                    sd['components']['comparison_split'] = {'groups': [group_a, group_b]}
                    strategy_decision = 'data-heavy/chart+comparison-split'
                    regions = [
                        {'id': 'main_chart', 'position': f'left-{chart_pct}', 'renderer': 'chart', 'data_source': 'visual'},
                        {'id': 'table_split', 'position': f'right-{table_pct}', 'renderer': 'comparison_table_split', 'data_source': 'components.comparison_split'},
                    ]
                    template = 'two-region-split'
                else:
                    strategy_decision = 'data-heavy/chart+comparison'
                    regions = [
                        {'id': 'main_chart', 'position': f'left-{chart_pct}', 'renderer': 'chart', 'data_source': 'visual'},
                        {'id': 'table', 'position': f'right-{table_pct}', 'renderer': 'comparison_table', 'data_source': 'components.comparison_items'},
                    ]
                    template = 'two-region-split'
            elif has_bullets:
                regions = [
                    {'id': 'main_chart', 'position': 'left-65', 'renderer': 'chart', 'data_source': 'visual'},
                    {'id': 'side_bullets', 'position': 'right-35', 'renderer': 'bullet_list', 'data_source': 'components.bullets'},
                ]
                template = 'two-region-split'
            elif has_callouts:
                # Chart + synthesised callouts (from speaker notes)
                strategy_decision = 'data-heavy/chart+callouts'
                regions = [
                    {'id': 'main_chart', 'position': 'left-65', 'renderer': 'chart', 'data_source': 'visual'},
                    {'id': 'callouts', 'position': 'right-35', 'renderer': 'callout_stack', 'data_source': 'components.callouts'},
                ]
                template = 'two-region-split'
            else:
                # If we have an insight but no bullets/callouts, synthesize a
                # minimal callout from the insight to avoid a single-element page.
                # This aligns with the optimization plan's "multi-region composite"
                # target and materially improves information density.
                insight = (sd.get('insight') or '').strip()
                if insight:
                    sd.setdefault('components', {}).setdefault('callouts', [
                        {'label': '要点', 'text': insight},
                    ])
                    strategy_decision = 'data-heavy/chart+insight-callout'
                    regions = [
                        {'id': 'main_chart', 'position': 'left-65', 'renderer': 'chart', 'data_source': 'visual'},
                        {'id': 'callouts', 'position': 'right-35', 'renderer': 'callout_stack', 'data_source': 'components.callouts'},
                    ]
                    template = 'two-region-split'
                else:
                    strategy_decision = 'data-heavy/chart-full'
                    regions = [
                        {'id': 'main_chart', 'position': 'full', 'renderer': 'chart', 'data_source': 'visual'},
                    ]
                    template = 'full-width'
        else:
            # Data-heavy without chart: fallback
            if has_kpis:
                regions.append({'id': 'kpis', 'position': 'top-25', 'renderer': 'kpi_row', 'data_source': 'components.kpis'})
            if has_bullets:
                regions.append({'id': 'bullets', 'position': 'full', 'renderer': 'bullet_list', 'data_source': 'components.bullets'})
            template = 'full-width'

    # ------- comparison -------
    elif stype == 'comparison':
        if has_comparison:
            items_count = len(sd['components']['comparison_items'])
            # Check if items split into two attribute-disjoint groups
            split = _split_comparison_by_attrs(sd['components']['comparison_items'])
            if split:
                group_a, group_b = split
                sd['components']['comparison_split'] = {'groups': [group_a, group_b]}
                strategy_decision = 'comparison/split-by-attrs'
                if has_callouts:
                    regions = [
                        {'id': 'table_split', 'position': 'left-65', 'renderer': 'comparison_table_split', 'data_source': 'components.comparison_split'},
                        {'id': 'callouts', 'position': 'right-35', 'renderer': 'callout_stack', 'data_source': 'components.callouts'},
                    ]
                else:
                    regions = [
                        {'id': 'table_split', 'position': 'full', 'renderer': 'comparison_table_split', 'data_source': 'components.comparison_split'},
                    ]
                template = 'two-region-split' if has_callouts else 'full-width'
            elif features.get('is_sparse_comparison') and has_callouts:
                # Sparse matrix pages benefit from slightly wider table area.
                strategy_decision = 'comparison/sparse-with-callouts'
                regions = [
                    {'id': 'table', 'position': 'left-70', 'renderer': 'comparison_table', 'data_source': 'components.comparison_items'},
                    {'id': 'callouts', 'position': 'right-30', 'renderer': 'callout_stack', 'data_source': 'components.callouts'},
                ]
                template = 'two-region-split'
            elif has_callouts:
                strategy_decision = 'comparison/table+callouts'
                regions = [
                    {'id': 'table', 'position': 'left-65', 'renderer': 'comparison_table', 'data_source': 'components.comparison_items'},
                    {'id': 'callouts', 'position': 'right-35', 'renderer': 'callout_stack', 'data_source': 'components.callouts'},
                ]
                template = 'two-region-split'
            elif has_bullets:
                strategy_decision = 'comparison/table+bullets'
                regions = [
                    {'id': 'table', 'position': 'left-65', 'renderer': 'comparison_table', 'data_source': 'components.comparison_items'},
                    {'id': 'side_bullets', 'position': 'right-35', 'renderer': 'bullet_list', 'data_source': 'components.bullets'},
                ]
                template = 'two-region-split'
            else:
                strategy_decision = 'comparison/full-table'
                regions = [
                    {'id': 'table', 'position': 'full', 'renderer': 'comparison_table', 'data_source': 'components.comparison_items'},
                ]
                template = 'full-width'

    # ------- decision -------
    elif stype == 'decision':
        has_decisions = _has_component(sd, 'decisions')
        if has_decisions and has_kpis:
            if has_callouts:
                dense_callouts = int(features.get('callout_count', 0) or 0) >= 3
                left_pct = 60 if dense_callouts else 65
                right_pct = 40 if dense_callouts else 35
                strategy_decision = 'decision/kpi+decisions+dense-callouts' if dense_callouts else 'decision/kpi+decisions+callouts'
                regions = [
                    {'id': 'kpis', 'position': 'top-18', 'renderer': 'kpi_row', 'data_source': 'components.kpis'},
                    {'id': 'decisions', 'position': f'left-{left_pct}', 'renderer': 'decisions', 'data_source': 'components.decisions'},
                    {'id': 'callouts', 'position': f'right-{right_pct}', 'renderer': 'callout_stack', 'data_source': 'components.callouts'},
                ]
                template = 'three-region-top'
            else:
                strategy_decision = 'decision/kpi+decisions'
                regions = [
                    {'id': 'kpis', 'position': 'top-18', 'renderer': 'kpi_row', 'data_source': 'components.kpis'},
                    {'id': 'decisions', 'position': 'bottom-80', 'renderer': 'decisions', 'data_source': 'components.decisions'},
                ]
                template = 'two-region-split'
        elif has_decisions:
            if has_callouts:
                dense_callouts = int(features.get('callout_count', 0) or 0) >= 3
                strategy_decision = 'decision/decisions+callouts-dense' if dense_callouts else 'decision/decisions+callouts'
                regions = [
                    {'id': 'decisions', 'position': 'left-65' if dense_callouts else 'left-70', 'renderer': 'decisions', 'data_source': 'components.decisions'},
                    {'id': 'callouts', 'position': 'right-35' if dense_callouts else 'right-30', 'renderer': 'callout_stack', 'data_source': 'components.callouts'},
                ]
                template = 'two-region-split'
            else:
                strategy_decision = 'decision/full'
                regions = [
                    {'id': 'decisions', 'position': 'full', 'renderer': 'decisions', 'data_source': 'components.decisions'},
                ]
                template = 'full-width'

    # ------- matrix -------
    elif stype == 'matrix':
        if has_risks and has_flow:
            heavy_risks = int(features.get('risk_count', 0) or 0) >= 4
            strategy_decision = 'matrix/risks+flow-heavy' if heavy_risks else 'matrix/risks+flow'
            regions = [
                {'id': 'risks', 'position': 'top-65' if heavy_risks else 'top-60', 'renderer': 'risks', 'data_source': 'components.risks'},
                {'id': 'flow', 'position': 'bottom-35' if heavy_risks else 'bottom-40', 'renderer': 'flow', 'data_source': 'visual.placeholder_data'},
            ]
            template = 'two-region-split'
        elif has_risks:
            if has_callouts:
                strategy_decision = 'matrix/risks+callouts'
                regions = [
                    {'id': 'risks', 'position': 'left-70', 'renderer': 'risks', 'data_source': 'components.risks'},
                    {'id': 'callouts', 'position': 'right-30', 'renderer': 'callout_stack', 'data_source': 'components.callouts'},
                ]
                template = 'two-region-split'
            else:
                strategy_decision = 'matrix/full-risks'
                regions = [
                    {'id': 'risks', 'position': 'full', 'renderer': 'risks', 'data_source': 'components.risks'},
                ]
                template = 'full-width'

    # ------- timeline -------
    elif stype == 'timeline':
        has_timeline = _has_component(sd, 'timeline_items')
        if has_timeline and has_flow:
            dense_timeline = int(features.get('timeline_count', 0) or 0) >= 5
            strategy_decision = 'timeline/stacked-dense' if dense_timeline else 'timeline/stacked'
            regions = [
                {'id': 'timeline', 'position': 'top-50' if dense_timeline else 'top-55', 'renderer': 'progression', 'data_source': 'components.timeline_items'},
                {'id': 'flow', 'position': 'bottom-50' if dense_timeline else 'bottom-45', 'renderer': 'flow', 'data_source': 'visual.placeholder_data.flow_data'},
            ]
            template = 'two-region-split'
        elif has_timeline:
            strategy_decision = 'timeline/full'
            regions = [
                {'id': 'timeline', 'position': 'full', 'renderer': 'progression', 'data_source': 'components.timeline_items'},
            ]
            template = 'full-width'

    # ------- bullet-list -------
    elif stype == 'bullet-list':
        if has_kpis and has_bullets and has_risks:
            regions = [
                {'id': 'kpis', 'position': 'top-22', 'renderer': 'kpi_row', 'data_source': 'components.kpis'},
                {'id': 'bullets', 'position': 'left-55', 'renderer': 'bullet_list', 'data_source': 'components.bullets'},
                {'id': 'risks', 'position': 'right-45', 'renderer': 'callout_stack', 'data_source': 'components.callouts'},
            ]
            template = 'three-region-top'
        elif has_kpis and has_bullets:
            regions = [
                {'id': 'kpis', 'position': 'top-22', 'renderer': 'kpi_row', 'data_source': 'components.kpis'},
                {'id': 'bullets', 'position': 'bottom-78', 'renderer': 'bullet_list', 'data_source': 'components.bullets'},
            ]
            template = 'two-region-split'
        elif has_bullets and has_risks:
            regions = [
                {'id': 'bullets', 'position': 'left-60', 'renderer': 'bullet_list', 'data_source': 'components.bullets'},
                {'id': 'risks', 'position': 'right-40', 'renderer': 'callout_stack', 'data_source': 'components.callouts'},
            ]
            template = 'two-region-split'
        elif has_bullets:
            regions = [
                {'id': 'bullets', 'position': 'full', 'renderer': 'bullet_list', 'data_source': 'components.bullets'},
            ]
            template = 'full-width'
        else:
            # No bullets, fall back to content
            regions = [
                {'id': 'content', 'position': 'full', 'renderer': 'bullet_list', 'data_source': 'content'},
            ]
            template = 'full-width'

    # ------- flowchart -------
    elif stype == 'flowchart':
        if has_flow:
            regions = [
                {'id': 'flow', 'position': 'full', 'renderer': 'flow', 'data_source': 'visual.placeholder_data'},
            ]
            if has_bullets:
                regions = [
                    {'id': 'flow', 'position': 'left-60', 'renderer': 'flow', 'data_source': 'visual.placeholder_data'},
                    {'id': 'side_bullets', 'position': 'right-40', 'renderer': 'bullet_list', 'data_source': 'components.bullets'},
                ]
                template = 'two-region-split'
            else:
                template = 'full-width'
        else:
            regions = [
                {'id': 'content', 'position': 'full', 'renderer': 'bullet_list', 'data_source': 'content'},
            ]
            template = 'full-width'

    # ------- fallback: any other type -------
    else:
        vtype = sd.get('visual', {}).get('type', 'none')
        if has_chart:
            # Keep renderer aligned with visual semantics
            if vtype == 'architecture_diagram':
                regions.append({'id': 'architecture', 'position': 'full', 'renderer': 'architecture', 'data_source': 'visual.placeholder_data'})
            elif vtype in ('flow_diagram', 'decision_tree'):
                regions.append({'id': 'flow', 'position': 'full', 'renderer': 'flow', 'data_source': 'visual.placeholder_data'})
            else:
                regions.append({'id': 'chart', 'position': 'full', 'renderer': 'chart', 'data_source': 'visual'})
        if has_bullets:
            pos = 'full' if not has_chart else 'right-40'
            regions.append({'id': 'bullets', 'position': pos, 'renderer': 'bullet_list', 'data_source': 'components.bullets'})
        if has_comparison:
            regions.append({'id': 'table', 'position': 'full', 'renderer': 'comparison_table', 'data_source': 'components.comparison_items'})
        if has_kpis:
            regions.append({'id': 'kpis', 'position': 'top-22', 'renderer': 'kpi_row', 'data_source': 'components.kpis'})
        if not regions:
            regions.append({'id': 'content', 'position': 'full', 'renderer': 'bullet_list', 'data_source': 'content'})
        template = 'full-width'

    if not regions:
        return None

    return {
        'template': template,
        'regions': regions,
        'strategy_meta': {
            'version': 'feature-v1',
            'decision': strategy_decision,
            'features': features,
        },
    }


# ---------------------------------------------------------------------------
# Step 5: Composite Transform (orchestrate steps 1-4)
# ---------------------------------------------------------------------------

def transform_v1_to_v2(
    v1_data: Dict,
    *,
    enable_merge: bool = True,
    enable_assertion: bool = True,
    enable_insight: bool = True,
    enable_layout: bool = True,
) -> Dict:
    """Transform a v1 slides_semantic.json into v2.

    Returns a new dict (deep copy) — the original is not modified.
    """
    v2 = copy.deepcopy(v1_data)
    slides: List[Dict] = v2.get('slides', [])
    sections: List[Dict] = v2.get('sections', [])

    def _normalize_component_keys(sd: Dict) -> None:
        """Normalize component aliases to schema-friendly keys in-place."""
        comps = sd.setdefault('components', {}) or {}
        # risk_callouts -> callouts
        rc = comps.get('risk_callouts')
        if isinstance(rc, list) and rc:
            existing = comps.get('callouts') if isinstance(comps.get('callouts'), list) else []
            comps['callouts'] = list(existing) + list(rc)
            comps.pop('risk_callouts', None)
        sd['components'] = comps

    def _sanitize_section_divider(sd: Dict) -> None:
        """Keep section dividers lightweight and schema-consistent."""
        if sd.get('slide_type') != 'section_divider':
            return
        comps = sd.get('components', {}) or {}
        callouts = comps.get('callouts') if isinstance(comps.get('callouts'), list) else []
        # Divider should carry at most one callout and no heavy components.
        sd['components'] = {'callouts': callouts[:1]} if callouts else {}

    def _recompute_sections(slides_list: List[Dict], old_sections: List[Dict]) -> List[Dict]:
        """Recompute section start_slide after merge/compression."""
        title_to_accent = {s.get('title', ''): s.get('accent', 'primary') for s in (old_sections or [])}
        ordered_old = [s.get('title', '') for s in (old_sections or []) if s.get('title')]
        rebuilt: List[Dict] = []
        seen: set = set()
        current_title = ordered_old[0] if ordered_old else ''

        for idx, sd in enumerate(slides_list, 1):
            if sd.get('slide_type') == 'section_divider':
                current_title = sd.get('title', '') or current_title
            elif sd.get('_section_label'):
                current_title = sd.get('_section_label')
            elif idx == 1 and not current_title and ordered_old:
                current_title = ordered_old[0]

            if not current_title:
                continue
            if current_title in seen:
                continue

            sec_id = f"S{len(rebuilt) + 1}"
            accent = title_to_accent.get(current_title)
            if not accent and old_sections:
                accent = old_sections[min(len(rebuilt), len(old_sections) - 1)].get('accent', 'primary')
            rebuilt.append({
                'id': sec_id,
                'title': current_title,
                'start_slide': idx,
                'accent': accent or 'primary',
            })
            seen.add(current_title)

        return rebuilt

    # Step 1: Assertion extraction
    if enable_assertion:
        for sd in slides:
            assertion = extract_assertion(sd)
            if assertion and assertion != sd.get('title', ''):
                sd['assertion'] = assertion
                # Deduplicate: remove bullets that became the assertion.
                # Allow the list to become empty — the layout engine will
                # simply not render a bullet region for 0 bullets.
                bullets = sd.get('components', {}).get('bullets', [])
                if bullets:
                    deduped = [b for b in bullets if not _text_overlap(b, assertion)]
                    if len(deduped) < len(bullets):
                        sd['components']['bullets'] = deduped

    # Step 2: Page merging
    if enable_merge:
        slides = merge_pages(slides, sections, enabled=True)
        v2['slides'] = slides
        # Re-number slide IDs
        for i, sd in enumerate(slides):
            sd['id'] = i + 1
            sd['slide_id'] = i + 1

    # Step 3: Insight annotation
    if enable_insight:
        for sd in slides:
            insight = extract_insight(sd)
            if insight:
                sd['insight'] = insight

    # Step 3.5: Component enrichment (add callouts to sparse slides)
    slides = enrich_components(slides)

    # Normalize components and keep section_divider lightweight
    for sd in slides:
        _normalize_component_keys(sd)
        _sanitize_section_divider(sd)

    # Step 4: Layout intent design
    if enable_layout:
        for sd in slides:
            # Pre-process: convert risks into callout-compatible format
            risks = sd.get('components', {}).get('risks', [])
            if risks and isinstance(risks, list):
                risk_callouts = []
                for r in risks:
                    label = r.get('label', '')
                    desc = r.get('description', '')
                    mitigation = r.get('mitigation', '')
                    prob = r.get('probability', '')
                    impact = r.get('impact', '')
                    text = f"⚠ {label}"
                    if desc:
                        text += f"：{desc}"
                    if mitigation:
                        text += f"\n→ {mitigation}"
                    risk_callouts.append({
                        'label': f"⚠ {label} ({prob}/{impact})" if prob else f"⚠ {label}",
                        'text': text,
                    })
                comps = sd.setdefault('components', {})
                existing = comps.get('callouts') if isinstance(comps.get('callouts'), list) else []
                comps['callouts'] = existing + risk_callouts
                comps.pop('risk_callouts', None)

            layout = design_layout_intent(sd)
            if layout:
                sd['layout_intent'] = layout

    # Recompute section start slides after merge/reorder so labels/progress stay accurate
    v2['sections'] = _recompute_sections(slides, sections)

    # Add v2 metadata
    v2['schema_version'] = 2
    v2['ea_transform'] = {
        'version': '1.1.0',
        'merging_enabled': enable_merge,
        'original_slide_count': len(v1_data.get('slides', [])),
        'output_slide_count': len(slides),
    }

    return v2


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='EA Transform: v1 → v2 slides_semantic.json')
    parser.add_argument('--input', '-i',
                        default='docs/presentations/storage-frontier-20260211/slides_semantic.json',
                        help='Path to v1 slides_semantic.json')
    parser.add_argument('--output', '-o',
                        default=None,
                        help='Output path for v2 JSON (default: <input_dir>/slides_semantic_v2.json)')
    parser.add_argument('--no-merge', action='store_true',
                        help='Disable page merging')
    parser.add_argument('--stats', action='store_true',
                        help='Print transformation statistics')
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    with open(input_path, 'r', encoding='utf-8') as f:
        v1_data = json.load(f)

    v2_data = transform_v1_to_v2(v1_data, enable_merge=not args.no_merge)

    output_path = Path(args.output) if args.output else input_path.parent / 'slides_semantic_v2.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(v2_data, f, ensure_ascii=False, indent=2)

    slides = v2_data['slides']
    meta = v2_data.get('ea_transform', {})
    original_count = meta.get('original_slide_count', '?')
    output_count = meta.get('output_slide_count', '?')

    print(f"EA Transform complete: {original_count} → {output_count} slides")
    print(f"Output: {output_path}")

    if args.stats:
        assertion_count = sum(1 for s in slides if s.get('assertion'))
        insight_count = sum(1 for s in slides if s.get('insight'))
        layout_count = sum(1 for s in slides if s.get('layout_intent'))
        multi_region = sum(1 for s in slides if s.get('layout_intent') and len(s['layout_intent'].get('regions', [])) >= 2)
        print(f"\nStatistics:")
        print(f"  Assertion titles: {assertion_count}/{len(slides)} ({assertion_count/len(slides)*100:.0f}%)")
        print(f"  Insight bars:     {insight_count}/{len(slides)} ({insight_count/len(slides)*100:.0f}%)")
        print(f"  Layout intents:   {layout_count}/{len(slides)} ({layout_count/len(slides)*100:.0f}%)")
        print(f"  Multi-region:     {multi_region}/{len(slides)} ({multi_region/len(slides)*100:.0f}%)")
        compression = output_count / original_count if original_count else 0
        print(f"  Compression:      {compression:.2f} ({original_count}→{output_count})")


if __name__ == '__main__':
    main()
