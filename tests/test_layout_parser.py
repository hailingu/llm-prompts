import importlib.util
import os
import sys

# Make repository paths importable so package relative imports succeed
ROOT = os.getcwd()
PKG_PATH = os.path.join(ROOT, 'skills', 'ppt-generator')
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if PKG_PATH not in sys.path:
    sys.path.insert(0, PKG_PATH)

from ppt_generator.grid import GridSystem
from ppt_generator.renderers import compute_region_bounds, resolve_data_source, detect_schema_version


def test_detect_schema_version():
    assert detect_schema_version({}) == 1
    assert detect_schema_version({'layout_intent': {}}) == 1
    assert detect_schema_version({'layout_intent': {'regions': []}}) == 2


def test_compute_region_bounds_full_and_left_right():
    grid = GridSystem({})
    left, top, width, height = compute_region_bounds('full', grid, bar_h=0.8)
    assert abs(left - grid.margin_h) < 1e-6
    assert abs(width - grid.usable_w) < 1e-6

    l, t, w, h = compute_region_bounds('left-60', grid, bar_h=0.8)
    assert abs(l - grid.margin_h) < 1e-6
    assert abs(w - grid.usable_w * 0.6) < 1e-6

    l2, t2, w2, h2 = compute_region_bounds('right-40', grid, bar_h=0.8)
    assert abs(w2 - grid.usable_w * 0.4) < 1e-6
    assert abs(l2 - (grid.margin_h + grid.usable_w - w2)) < 1e-6


def test_compute_region_bounds_col_and_top():
    grid = GridSystem({})
    l, t, w, h = compute_region_bounds('col-0-6', grid, bar_h=0.8)
    l_exp, w_exp = grid.col_span(6, 0)
    assert abs(l - l_exp) < 1e-6
    assert abs(w - w_exp) < 1e-6

    l2, t2, w2, h2 = compute_region_bounds('top-30', grid, bar_h=0.8)
    # height should be approx 30% of available default height
    avail = max(1.0, grid.slide_h - 0.8 - 0.5)
    assert abs(h2 - (avail * 0.3)) < 1e-6


def test_resolve_data_source_basic():
    slide = {
        'components': {
            'kpis': [{'label': 'a', 'value': 1}]
        },
        'visual': {'type': 'bar_chart'},
        'content': ['a', 'b']
    }
    assert resolve_data_source(slide, 'components.kpis') == slide['components']['kpis']
    assert resolve_data_source(slide, 'visual') == slide['visual']
    assert resolve_data_source(slide, 'content') == slide['content']
    assert resolve_data_source(slide, 'components.missing') is None
