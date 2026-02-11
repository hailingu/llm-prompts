"""Tests for the Exhibit Architect (EA) v1→v2 transform."""
import json
import sys
import os

sys.path.insert(0, os.path.abspath('scripts'))
from exhibit_architect import (
    extract_assertion,
    extract_insight,
    design_layout_intent,
    transform_v1_to_v2,
    merge_pages,
)


def _load_v1():
    with open('docs/presentations/storage-frontier-20260211/slides_semantic.json', 'r', encoding='utf-8') as f:
        return json.load(f)


def test_assertion_extraction_from_bullets():
    sd = {
        'slide_type': 'bullet-list',
        'components': {
            'bullets': [
                '全球市场从 ~85 亿美元增长到 ~110 亿美元（2023→2026）',
                'NVMe 采纳率超过 65%',
            ]
        },
        'speaker_notes': '',
    }
    assertion = extract_assertion(sd)
    assert assertion, 'Should extract an assertion from bullets'
    assert len(assertion) <= 80, 'Assertion should be concise'


def test_assertion_extraction_from_comparison():
    sd = {
        'slide_type': 'comparison',
        'components': {
            'comparison_items': [
                {'label': 'Plan A', 'attributes': {}},
                {'label': 'Plan B', 'attributes': {}},
            ]
        },
        'speaker_notes': '',
    }
    assertion = extract_assertion(sd)
    assert 'Plan A' in assertion and 'Plan B' in assertion


def test_assertion_skips_title_slides():
    sd = {'slide_type': 'title', 'components': {}, 'speaker_notes': ''}
    assert extract_assertion(sd) == ''


def test_assertion_preserves_existing():
    sd = {'slide_type': 'data-heavy', 'assertion': 'Existing assertion', 'components': {}, 'speaker_notes': ''}
    assert extract_assertion(sd) == 'Existing assertion'


def test_insight_extraction_from_notes():
    sd = {
        'slide_type': 'data-heavy',
        'speaker_notes': '指出数据来源。建议优先布局 NVMe 市场。',
        'components': {},
    }
    insight = extract_insight(sd)
    assert insight, 'Should extract insight'
    assert '建议' in insight or '优先' in insight or '布局' in insight


def test_insight_from_risks():
    sd = {
        'slide_type': 'bullet-list',
        'speaker_notes': '',
        'components': {
            'risks': [{'label': '成本', 'description': '过高', 'mitigation': '分层存储'}]
        },
    }
    insight = extract_insight(sd)
    assert '成本' in insight or '分层' in insight


def test_layout_intent_data_heavy_with_chart_and_kpis():
    sd = {
        'slide_type': 'data-heavy',
        'visual': {'type': 'line_chart', 'placeholder_data': {}},
        'components': {'kpis': [{'label': 'A', 'value': '100'}]},
    }
    layout = design_layout_intent(sd)
    assert layout is not None
    assert len(layout['regions']) >= 2
    renderers = {r['renderer'] for r in layout['regions']}
    assert 'chart' in renderers
    assert 'kpi_row' in renderers


def test_layout_intent_comparison_with_callouts():
    sd = {
        'slide_type': 'comparison',
        'visual': {'type': 'none'},
        'components': {
            'comparison_items': [{'label': 'A', 'attributes': {}}, {'label': 'B', 'attributes': {}}],
            'callouts': [{'label': 'Note', 'text': 'Important'}],
        },
    }
    layout = design_layout_intent(sd)
    assert layout is not None
    assert len(layout['regions']) == 2
    renderers = {r['renderer'] for r in layout['regions']}
    assert 'comparison_table' in renderers
    assert 'callout_stack' in renderers


def test_layout_intent_skips_title():
    sd = {'slide_type': 'title', 'visual': {'type': 'none'}, 'components': {}}
    assert design_layout_intent(sd) is None


def test_full_transform_produces_v2():
    v1 = _load_v1()
    v2 = transform_v1_to_v2(v1, enable_merge=True)
    slides = v2['slides']

    # Basic structure
    assert v2.get('schema_version') == 2
    assert v2.get('ea_transform', {}).get('original_slide_count') == 23
    assert len(slides) <= 15, f'Expected ≤15 slides after merge, got {len(slides)}'

    # Assertion coverage ≥70%
    content_slides = [s for s in slides if s['slide_type'] not in ('title', 'section_divider')]
    assertion_count = sum(1 for s in content_slides if s.get('assertion'))
    assert assertion_count / len(content_slides) >= 0.7, \
        f'Assertion rate {assertion_count}/{len(content_slides)} < 70%'

    # Layout intent coverage
    layout_count = sum(1 for s in content_slides if s.get('layout_intent'))
    assert layout_count / len(content_slides) >= 0.7, \
        f'Layout intent rate {layout_count}/{len(content_slides)} < 70%'

    # Multi-region rate ≥40%
    multi_region = sum(1 for s in content_slides
                       if s.get('layout_intent') and len(s['layout_intent'].get('regions', [])) >= 2)
    assert multi_region / len(content_slides) >= 0.4, \
        f'Multi-region rate {multi_region}/{len(content_slides)} < 40%'


def test_transform_no_merge():
    v1 = _load_v1()
    v2 = transform_v1_to_v2(v1, enable_merge=False)
    assert len(v2['slides']) == 23, 'Without merge, slide count should be unchanged'


def test_merge_absorbs_section_dividers():
    v1 = _load_v1()
    v2 = transform_v1_to_v2(v1, enable_merge=True)
    slides = v2['slides']
    divider_count = sum(1 for s in slides if s['slide_type'] == 'section_divider')
    assert divider_count <= 2, f'Expected ≤2 section dividers after merge, got {divider_count}'
    # First divider should be kept (opening)
    assert slides[1]['slide_type'] == 'section_divider'


def test_merged_slides_have_section_labels():
    v1 = _load_v1()
    v2 = transform_v1_to_v2(v1, enable_merge=True)
    slides = v2['slides']
    labeled = [s for s in slides if s.get('_section_label')]
    assert len(labeled) >= 3, f'Expected ≥3 slides with section labels, got {len(labeled)}'
