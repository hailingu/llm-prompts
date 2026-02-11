import os
import sys
ROOT = os.getcwd()
PKG_PATH = os.path.join(ROOT, 'skills', 'ppt-generator')
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if PKG_PATH not in sys.path:
    sys.path.insert(0, PKG_PATH)

import math
from ppt_generator.metrics import compute_deck_metrics


def test_empty_deck():
    metrics = compute_deck_metrics({'slides': []})
    assert metrics['assertion_title_rate'] == 0.0
    assert metrics['native_visual_rate'] == 0.0
    assert metrics['compression_ratio'] == 0.0
    assert metrics['placeholder_rate'] == 0.0
    assert metrics['multi_region_rate'] == 0.0
    assert metrics['avg_components_per_slide'] == 0.0


def test_metrics_simple_synthetic():
    slides = [
        # slide 1: has assertion, native chart placeholder (chart_config, no rendered)
        {
            'id': 1,
            'content': ['p1', 'p2'],
            'visual': {'type': 'line_chart', 'placeholder_data': {'chart_config': {'x': [1], 'series': []}}},
            'assertion': 'Yes'
        },
        # slide 2: placeholder visual (no chart_config, no rendered)
        {
            'id': 2,
            'content': ['p1', 'p2', 'p3'],
            'visual': {'type': 'bar_chart', 'placeholder_data': {}},
        },
        # slide 3: v2 with multiple regions
        {
            'id': 3,
            'content': [],
            'layout_intent': {'regions': [{'id': 'r1'}, {'id': 'r2'}]},
        },
        # slide 4: component with architecture-like payload (native shape)
        {
            'id': 4,
            'content': ['p1'],
            'components': {'architecture': [{'nodes': [{'id': 'n1', 'label': 'A'}], 'edges': []}]}
        }
    ]

    semantic = {'slides': slides}
    m = compute_deck_metrics(semantic)

    assert math.isclose(m['assertion_title_rate'], 1 / 4)
    assert math.isclose(m['native_visual_rate'], 2 / 4)
    # compression ratio = slides (4) / input paragraphs (2 + 3 + 0 + 1 = 6)
    assert math.isclose(m['compression_ratio'], 4 / 6)
    assert math.isclose(m['placeholder_rate'], 1 / 4)
    assert math.isclose(m['multi_region_rate'], 1 / 4)
    # components per slide = 1 component in slide 4 -> total_components=1 => avg 1/4
    assert math.isclose(m['avg_components_per_slide'], 1 / 4)


def test_metrics_counts_bounded():
    # Run on a real-ish v2 sample (should run without exceptions)
    import json
    base = 'docs/presentations/storage-frontier-20260211'
    sem = json.load(open(base + '/slides_semantic_v2.json', encoding='utf-8'))
    m = compute_deck_metrics(sem)
    # All rates are bounded 0..1 except compression_ratio which can be >0
    for k in ('assertion_title_rate', 'native_visual_rate', 'placeholder_rate', 'multi_region_rate'):
        assert 0.0 <= m[k] <= 1.0, f"{k} out of range: {m[k]}"
    assert m['avg_components_per_slide'] >= 0.0
    assert m['compression_ratio'] >= 0.0
