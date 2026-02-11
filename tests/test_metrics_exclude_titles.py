import os, sys
ROOT = os.getcwd()
PKG_PATH = os.path.join(ROOT, 'skills', 'ppt-generator')
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if PKG_PATH not in sys.path:
    sys.path.insert(0, PKG_PATH)

from ppt_generator.metrics import compute_deck_metrics


def test_compute_excludes_title_and_section_divider():
    semantic = {
        'slides': [
            {'slide_type': 'title'},
            {'slide_type': 'section_divider'},
            {'slide_type': 'data-heavy', 'assertion': 'A', 'visual': {'type': 'line_chart', 'placeholder_data': {'chart_config': {'series': [{'name':'s','values':[1]}]}}}},
            {'slide_type': 'comparison', 'assertion': None, 'visual': {'type': 'none'}}
        ]
    }
    metrics = compute_deck_metrics(semantic)
    # total slides considered should be 2 (excluding title and section_divider)
    assert metrics['total_slides'] == 2
    # assertion rate should be 1/2 = 0.5
    assert metrics['assertion_title_rate'] == 0.5
    # native visual rate should count the line_chart -> 1/2
    assert metrics['native_visual_rate'] == 0.5
