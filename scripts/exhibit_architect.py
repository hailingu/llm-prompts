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
    # Take first sentence
    first = re.split(r'[。；;.\n]', notes)[0].strip()
    if len(first) > 80:
        first = first[:77] + '…'
    return first


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
    # Don't merge title slides
    if a.get('slide_type') == 'title' or b.get('slide_type') == 'title':
        return False

    # Section dividers can be absorbed into the next content slide
    if a.get('slide_type') == 'section_divider':
        return True  # absorb divider into next slide

    # Don't merge a content slide into a section_divider
    if b.get('slide_type') == 'section_divider':
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
        a_type = a.get('slide_type', '')
        b_type = b.get('slide_type', '')
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
    priority = {'data-heavy': 5, 'comparison': 4, 'flowchart': 3, 'bullet-list': 2}
    a_pri = priority.get(a.get('slide_type', ''), 1)
    b_pri = priority.get(b.get('slide_type', ''), 1)
    if b_pri > a_pri:
        merged['slide_type'] = b['slide_type']

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

    for _pass in range(3):  # multi-pass: up to 3 iterations
        result: List[Dict] = []
        i = 0
        changed = False

        while i < len(current):
            curr = current[i]

            # Keep the first section divider (slide 2 — opening & conclusions)
            # but absorb subsequent ones
            if curr.get('slide_type') == 'section_divider' and curr.get('id', 0) <= 2:
                result.append(curr)
                i += 1
                continue

            # Try merging with next slide
            if i + 1 < len(current):
                nxt = current[i + 1]
                if _can_merge(curr, nxt):
                    merged = _merge_slides(curr, nxt)
                    # Try to also merge a third slide (for section_divider chains)
                    if i + 2 < len(current):
                        nxt2 = current[i + 2]
                        if curr.get('slide_type') == 'section_divider' and _can_merge(merged, nxt2):
                            merged = _merge_slides(merged, nxt2)
                            result.append(merged)
                            i += 3
                            changed = True
                            continue
                    result.append(merged)
                    i += 2
                    changed = True
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

    # Try to find actionable parts in speaker notes
    if notes:
        # Look for recommendation/suggestion patterns
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

        # Second sentence often has insight
        sentences = re.split(r'[。；;.\n]', notes)
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
        if stype in ('title', 'section_divider'):
            continue

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

    return slides


# ---------------------------------------------------------------------------
# Step 4: Layout Intent Design
# ---------------------------------------------------------------------------

def _has_component(sd: Dict, key: str) -> bool:
    items = sd.get('components', {}).get(key, [])
    return isinstance(items, list) and len(items) > 0


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
    has_callouts = _has_component(sd, 'callouts')
    has_risks = _has_component(sd, 'risks')
    has_flow = sd.get('visual', {}).get('type') == 'flow_diagram'
    content = sd.get('content', [])
    has_content = isinstance(content, list) and len(content) > 0

    regions: List[Dict] = []

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
                # Determine chart complexity for width allocation
                chart_series = sd.get('visual', {}).get('placeholder_data', {}).get('chart_config', {}).get('series', [])
                chart_simple = len(chart_series) <= 1
                if split:
                    group_a, group_b = split
                    sd['components']['comparison_split'] = {'groups': [group_a, group_b]}
                    chart_pct = 30 if chart_simple else 40
                    table_pct = 100 - chart_pct
                    regions = [
                        {'id': 'main_chart', 'position': f'left-{chart_pct}', 'renderer': 'chart', 'data_source': 'visual'},
                        {'id': 'table_split', 'position': f'right-{table_pct}', 'renderer': 'comparison_table_split', 'data_source': 'components.comparison_split'},
                    ]
                    template = 'two-region-split'
                else:
                    chart_pct = 35 if chart_simple else 50
                    table_pct = 100 - chart_pct
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
                regions = [
                    {'id': 'main_chart', 'position': 'left-65', 'renderer': 'chart', 'data_source': 'visual'},
                    {'id': 'callouts', 'position': 'right-35', 'renderer': 'callout_stack', 'data_source': 'components.callouts'},
                ]
                template = 'two-region-split'
            else:
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
            elif has_callouts:
                regions = [
                    {'id': 'table', 'position': 'left-65', 'renderer': 'comparison_table', 'data_source': 'components.comparison_items'},
                    {'id': 'callouts', 'position': 'right-35', 'renderer': 'callout_stack', 'data_source': 'components.callouts'},
                ]
                template = 'two-region-split'
            elif has_bullets:
                regions = [
                    {'id': 'table', 'position': 'left-65', 'renderer': 'comparison_table', 'data_source': 'components.comparison_items'},
                    {'id': 'side_bullets', 'position': 'right-35', 'renderer': 'bullet_list', 'data_source': 'components.bullets'},
                ]
                template = 'two-region-split'
            else:
                regions = [
                    {'id': 'table', 'position': 'full', 'renderer': 'comparison_table', 'data_source': 'components.comparison_items'},
                ]
                template = 'full-width'

    # ------- bullet-list -------
    elif stype == 'bullet-list':
        if has_kpis and has_bullets and has_risks:
            regions = [
                {'id': 'kpis', 'position': 'top-22', 'renderer': 'kpi_row', 'data_source': 'components.kpis'},
                {'id': 'bullets', 'position': 'left-55', 'renderer': 'bullet_list', 'data_source': 'components.bullets'},
                {'id': 'risks', 'position': 'right-45', 'renderer': 'callout_stack', 'data_source': 'components.risk_callouts'},
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
                {'id': 'risks', 'position': 'right-40', 'renderer': 'callout_stack', 'data_source': 'components.risk_callouts'},
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
        if has_chart:
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

    # Step 3: Insight annotation
    if enable_insight:
        for sd in slides:
            insight = extract_insight(sd)
            if insight:
                sd['insight'] = insight

    # Step 3.5: Component enrichment (add callouts to sparse slides)
    slides = enrich_components(slides)

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
                sd.setdefault('components', {})['risk_callouts'] = risk_callouts

            layout = design_layout_intent(sd)
            if layout:
                sd['layout_intent'] = layout

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
