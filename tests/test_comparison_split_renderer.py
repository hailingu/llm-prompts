import os, sys
ROOT = os.getcwd()
PKG_PATH = os.path.join(ROOT, 'skills', 'ppt-generator')
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if PKG_PATH not in sys.path:
    sys.path.insert(0, PKG_PATH)

from pptx import Presentation
from pptx.util import Inches
from ppt_generator.renderers import render_region_comparison_split
from ppt_generator.grid import GridSystem


def make_slide():
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    spec = {}
    grid = GridSystem(spec)
    return prs, slide, spec, grid


def test_comparison_split_two_groups_renders():
    prs, slide, spec, grid = make_slide()
    left, top, width, height = grid.margin_h, 1.0, grid.usable_w, 3.0
    groups = [
        [{'label': 'A 架构', 'attributes': {'x': 1}}, {'label': 'B 架构', 'attributes': {'x': 2}}],
        [{'label': 'C 试点', 'attributes': {'y': 3}}, {'label': 'D 试点', 'attributes': {'y': 4}}],
    ]
    render_region_comparison_split(slide, {'groups': groups}, (left, top, width, height), spec)
    assert len(slide.shapes) > 0
