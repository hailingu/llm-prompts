"""Renderers for slides and components.

This module reuses helpers and grid to provide all rendering functions
exported by the original monolithic script.
"""
from typing import Any, Dict, List
import json
import os
import re
import logging

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.dml.color import RGBColor
from pptx.oxml.ns import qn

from .helpers import (
    get_color, get_color_hex, get_font_size, hex_to_rgb, px_to_inches
)
from .grid import GridSystem


# For brevity we copy selectively the rendering helpers from the original
# script, focusing on title bar, bottom bar, speaker notes, and the main
# per-slide dispatcher. Additional component renderers (kpis, bullets,
# tables, visuals) are also included as smaller functions.


def get_layout_zones(spec: Dict) -> Dict[str, float]:
    lz = spec.get('layout_zones', {})
    return {
        'title_bar_h': lz.get('title_bar_height_default', 0.55),
        'title_bar_h_narrow': lz.get('title_bar_height_narrow', 0.4),
        'bottom_bar_h': max(lz.get('bottom_bar_height', 0.25), 0.25),
        'content_margin_top': lz.get('content_margin_top', 0.12),
        'content_bottom_margin': lz.get('content_bottom_margin', 0.2),
    }


def get_title_bar_mode(spec: Dict, slide_type: str) -> str:
    if slide_type in ('title', 'section_divider'):
        return 'none'
    stl = spec.get('slide_type_layouts', {})
    entry = stl.get(slide_type, stl.get('default', {}))
    return entry.get('title_bar', 'standard')


def apply_font_to_run(run: Any, spec: Dict) -> None:
    try:
        font_family = spec.get('typography_system', {}).get('font_family')
        cjk_font = spec.get('typography_system', {}).get('cjk_font_family')
        if font_family:
            run.font.name = font_family
            r_pr = getattr(run._element, 'rPr', None)
            if r_pr is not None:
                try:
                    r_pr.rFonts.set(qn('a:ascii'), font_family)
                    r_pr.rFonts.set(qn('a:hAnsi'), font_family)
                except Exception:
                    pass
        if cjk_font:
            r_pr = getattr(run._element, 'rPr', None)
            if r_pr is not None:
                try:
                    r_pr.rFonts.set(qn('a:ea'), cjk_font)
                except Exception:
                    pass
    except Exception:
        pass


def create_card_shape(slide: Any, spec: Dict, left_in: float, top_in: float, width_in: float, height_in: float):
    try:
        br = spec.get('component_library', {}).get('card', {}).get('border_radius')
    except Exception:
        br = None
    shape_type = MSO_SHAPE.ROUNDED_RECTANGLE if br is None or br != 0 else MSO_SHAPE.RECTANGLE
    return slide.shapes.add_shape(shape_type, Inches(left_in), Inches(top_in), Inches(width_in), Inches(height_in))


def render_title_bar(slide: Any, spec: Dict, grid: GridSystem, title: str, slide_num: int, _total_slides: int,
                     section_label: str = '', accent_color_token: str = 'primary', mode: str = 'standard') -> float:
    if mode == 'none':
        return 0.0
    lz = get_layout_zones(spec)
    bar_h = lz['title_bar_h_narrow'] if mode == 'narrow' else lz['title_bar_h']
    title_font_pt = get_font_size(spec, 'slide_title')
    title_text_h = title_font_pt / 72.0 + 0.08
    if section_label:
        min_bar_h = 0.28 + title_text_h
    else:
        min_bar_h = 0.10 + title_text_h
    bar_h = max(bar_h, min_bar_h)

    bar = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0), Inches(0),
        Inches(grid.slide_w), Inches(bar_h)
    )
    bar.fill.solid()
    bar.fill.fore_color.rgb = get_color(spec, accent_color_token)
    bar.line.fill.background()

    if section_label:
        tb = slide.shapes.add_textbox(
            Inches(grid.margin_h), Inches(0.06), Inches(4), Inches(0.22)
        )
        tf = tb.text_frame
        tf.word_wrap = False
        p = tf.paragraphs[0]
        run = p.add_run()
        run.text = section_label
        run.font.size = Pt(get_font_size(spec, 'section_label'))
        run.font.bold = True
        run.font.color.rgb = get_color(spec, 'on_primary')
        apply_font_to_run(run, spec)

    title_top = 0.22 if section_label else 0.10
    tb = slide.shapes.add_textbox(
        Inches(grid.margin_h), Inches(title_top),
        Inches(grid.slide_w - 2 * grid.margin_h - 1.0), Inches(bar_h - title_top - 0.05)
    )
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = title
    run.font.size = Pt(get_font_size(spec, 'slide_title'))
    run.font.bold = True
    run.font.color.rgb = get_color(spec, 'on_primary')
    apply_font_to_run(run, spec)

    tb_num = slide.shapes.add_textbox(
        Inches(grid.slide_w - 1.0), Inches(title_top), Inches(0.7), Inches(0.35)
    )
    tf_num = tb_num.text_frame
    p = tf_num.paragraphs[0]
    run = p.add_run()
    run.text = f"{slide_num}"
    run.font.size = Pt(get_font_size(spec, 'page_number'))
    run.font.bold = True
    run.font.color.rgb = get_color(spec, 'on_primary')
    apply_font_to_run(run, spec)
    p.alignment = PP_ALIGN.RIGHT

    return bar_h


def render_bottom_bar(slide: Any, spec: Dict, grid: GridSystem, section_name: str, slide_num: int, total_slides: int,
                      section_index: int = 0, total_sections: int = 6, accent_token: str = 'primary') -> None:
    lz = get_layout_zones(spec)
    bar_h = lz['bottom_bar_h']
    bar_top = grid.slide_h - bar_h

    bar = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0), Inches(bar_top), Inches(grid.slide_w), Inches(bar_h)
    )
    bar.fill.solid()
    bar.fill.fore_color.rgb = get_color(spec, 'surface_variant')
    bar.line.fill.background()

    tb = slide.shapes.add_textbox(Inches(grid.margin_h), Inches(bar_top + 0.03), Inches(3), Inches(bar_h - 0.06))
    tf = tb.text_frame
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = section_name
    run.font.size = Pt(get_font_size(spec, 'label'))
    run.font.bold = True
    run.font.color.rgb = get_color(spec, accent_token)

    prog_left = grid.slide_w / 2 - 1.5
    seg_w = 3.0 / max(total_sections, 1)
    for i in range(total_sections):
        seg = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE,
            Inches(prog_left + i * seg_w + 0.02), Inches(bar_top + bar_h / 2 - 0.03),
            Inches(seg_w - 0.04), Inches(0.06)
        )
        seg.fill.solid()
        if i <= section_index:
            seg.fill.fore_color.rgb = get_color(spec, accent_token)
        else:
            seg.fill.fore_color.rgb = get_color(spec, 'outline')
        seg.line.fill.background()

    tb = slide.shapes.add_textbox(
        Inches(grid.slide_w - grid.margin_h - 1.5), Inches(bar_top + 0.03), Inches(1.5), Inches(bar_h - 0.06)
    )
    tf = tb.text_frame
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.RIGHT
    run = p.add_run()
    run.text = f"{slide_num} / {total_slides}"
    run.font.size = Pt(get_font_size(spec, 'page_number'))
    run.font.color.rgb = get_color(spec, 'on_surface_variant')


def render_speaker_notes(slide: Any, notes_text: Any) -> None:
    if not notes_text:
        return
    text = notes_text if isinstance(notes_text, str) else json.dumps(notes_text, ensure_ascii=False)
    notes_slide = slide.notes_slide
    notes_slide.notes_text_frame.text = text


# Minimal component renderers used by slide renderers

def render_kpis(slide: Any, kpis: List[Dict], spec: Dict, _grid: GridSystem, left: float, top: float, width: float) -> float:
    if not kpis:
        return 0
    n = len(kpis)
    card_gap = 0.15
    card_w = (width - card_gap * (n - 1)) / max(n, 1)
    card_h = 0.85
    for i, kpi in enumerate(kpis):
        cx = left + i * (card_w + card_gap)
        card = create_card_shape(slide, spec, cx, top, card_w, card_h)
        card.fill.solid()
        card.fill.fore_color.rgb = get_color(spec, 'primary_container')
        card.line.fill.background()
        # Value
        tb = slide.shapes.add_textbox(Inches(cx + 0.12), Inches(top + 0.10), Inches(card_w - 0.24), Inches(0.40))
        tf = tb.text_frame
        p = tf.paragraphs[0]
        run = p.add_run()
        val = kpi.get('value', '')
        trend = kpi.get('trend', '')
        trend_arrow = {'up': ' ↑', 'down': ' ↓', 'stable': ' →'}.get(trend, '')
        run.text = f"{val}{trend_arrow}"
        run.font.size = Pt(get_font_size(spec, 'kpi_value'))
        run.font.bold = True
        run.font.color.rgb = get_color(spec, 'on_primary_container')
        apply_font_to_run(run, spec)
        # Label
        tb2 = slide.shapes.add_textbox(Inches(cx + 0.12), Inches(top + 0.52), Inches(card_w - 0.24), Inches(0.25))
        tf2 = tb2.text_frame
        p2 = tf2.paragraphs[0]
        run2 = p2.add_run()
        run2.text = kpi.get('label', '')
        run2.font.size = Pt(get_font_size(spec, 'kpi_label'))
        run2.font.color.rgb = get_color(spec, 'on_primary_container')
    return card_h + 0.15


# Visual/placeholder renderers (simplified)

def render_visual(slide: Any, visual: Dict, spec: Dict, _grid: GridSystem, left: float, top: float, width: float, height: float) -> None:
    if not visual or visual.get('type') in (None, 'none'):
        return
    pd = visual.get('placeholder_data', {})
    if pd.get('chart_config'):
        render_chart_table(slide, visual, spec, left, top, width, height)
    elif pd.get('mermaid_code'):
        render_mermaid_placeholder(slide, visual, spec, left, top, width, height)
    else:
        render_visual_placeholder(slide, visual, spec, left, top, width, height)


def render_chart_table(slide: Any, visual: Dict, spec: Dict, left: float, top: float, width: float, height: float) -> None:
    config = visual.get('placeholder_data', {}).get('chart_config', {})
    labels = config.get('labels', [])
    series = config.get('series', [])
    if not labels:
        return
    n_cols = len(labels)
    col_w = width / max(n_cols, 1)
    n_rows = len(series) + 1
    title_offset = 0.35 if visual.get('title') else 0
    usable_h = height - title_offset
    row_h = max(0.42, min(1.2, usable_h / max(n_rows, 1)))
    if visual.get('title'):
        tb = slide.shapes.add_textbox(Inches(left), Inches(top - 0.30), Inches(width), Inches(0.28))
        p = tb.text_frame.paragraphs[0]
        run = p.add_run()
        run.text = visual['title']
        run.font.size = Pt(get_font_size(spec, 'table_header'))
        run.font.bold = True
        run.font.color.rgb = get_color(spec, 'on_surface')
    for j, label in enumerate(labels):
        tb = slide.shapes.add_textbox(Inches(left + j * col_w), Inches(top), Inches(col_w), Inches(row_h))
        tf = tb.text_frame
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        tf.word_wrap = True
        p = tf.paragraphs[0]
        run = p.add_run()
        run.text = str(label)
        run.font.size = Pt(get_font_size(spec, 'table_header'))
        run.font.bold = True
        run.font.color.rgb = get_color(spec, 'on_surface')
    sep = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(left), Inches(top + row_h - 0.02), Inches(width), Pt(2))
    sep.fill.solid()
    sep.fill.fore_color.rgb = get_color(spec, 'primary')
    sep.line.fill.background()
    for r, s in enumerate(series):
        ry = top + (r + 1) * row_h
        if r % 2 == 1:
            stripe = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(left), Inches(ry), Inches(width), Inches(row_h))
            stripe.fill.solid()
            stripe.fill.fore_color.rgb = get_color(spec, 'surface_variant')
            stripe.line.fill.background()
        data = s.get('data', [])
        for j, val in enumerate(data):
            if j >= n_cols:
                break
            tb = slide.shapes.add_textbox(Inches(left + j * col_w), Inches(ry), Inches(col_w), Inches(row_h))
            tf = tb.text_frame
            tf.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf.word_wrap = True
            p = tf.paragraphs[0]
            run = p.add_run()
            run.text = str(val)
            run.font.size = Pt(get_font_size(spec, 'table_cell'))
            run.font.color.rgb = get_color(spec, 'on_surface')
            if isinstance(val, (int, float)):
                p.alignment = PP_ALIGN.RIGHT


def render_mermaid_placeholder(slide: Any, visual: Dict, spec: Dict, left: float, top: float, width: float, height: float) -> None:
    mermaid = visual.get('placeholder_data', {}).get('mermaid_code', '')
    card = create_card_shape(slide, spec, left, top, width, min(height, 3.5))
    card.fill.solid()
    card.fill.fore_color.rgb = get_color(spec, 'surface_variant')
    card.line.color.rgb = get_color(spec, 'outline')
    card.line.width = Pt(1)
    if visual.get('title'):
        tb = slide.shapes.add_textbox(Inches(left + 0.2), Inches(top + 0.12), Inches(width - 0.4), Inches(0.28))
        p = tb.text_frame.paragraphs[0]
        run = p.add_run()
        run.text = f"📊 {visual['title']}"
        run.font.size = Pt(get_font_size(spec, 'table_header'))
        run.font.bold = True
        run.font.color.rgb = get_color(spec, 'on_surface')
        apply_font_to_run(run, spec)
    preview = '\n'.join(mermaid.strip().split('\n')[:8])
    if len(mermaid.strip().split('\n')) > 8:
        preview += '\n  ...'
    tb = slide.shapes.add_textbox(Inches(left + 0.2), Inches(top + 0.50), Inches(width - 0.4), Inches(min(height - 0.7, 2.5)))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    run = p.add_run()
    run.font.size = Pt(11)
    run.font.color.rgb = get_color(spec, 'outline')
    apply_font_to_run(run, spec)
    run.text = preview


def render_visual_placeholder(slide: Any, visual: Dict, spec: Dict, left: float, top: float, width: float, height: float) -> None:
    card = create_card_shape(slide, spec, left, top, width, min(height, 2.5))
    card.fill.solid()
    card.fill.fore_color.rgb = get_color(spec, 'surface_variant')
    card.line.color.rgb = get_color(spec, 'outline')
    card.line.width = Pt(1)
    label = visual.get('title', visual.get('type', 'Visual'))
    reqs = visual.get('content_requirements', [])
    text = f"[{label}]"
    if reqs:
        text += '\n' + '\n'.join(f"  • {r}" for r in reqs[:3])
    tb = slide.shapes.add_textbox(Inches(left + 0.2), Inches(top + 0.3), Inches(width - 0.4), Inches(min(height - 0.6, 2.0)))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    run = p.add_run()
    run.text = text
    run.font.size = Pt(get_font_size(spec, 'table_header'))
    run.font.color.rgb = get_color(spec, 'outline')
    apply_font_to_run(run, spec)


# Dispatcher of slide types (partial set)

RENDERERS = {
    'title': None,  # implemented below
    'section_divider': None,
    'bullet-list': None,
    'two-column': None,
    'comparison': None,
    'decision': None,
    'data-heavy': None,
}


def _resolve_bg_image(spec: Dict, slide_type: str, sd: Dict) -> str:
    img = sd.get('background_image', '')
    if img:
        return img
    stl = spec.get('slide_type_layouts', {})
    entry = stl.get(slide_type, stl.get('default', {}))
    return entry.get('background_image', '')


def find_section_for_slide(slide_id: int, sections: List[Dict]) -> Dict:
    result = {}
    for sec in sorted(sections, key=lambda s: s.get('start_slide', 0)):
        if slide_id >= sec.get('start_slide', 999):
            result = sec
    return result


def _render_background(slide: Any, sd: Dict, spec: Dict, grid: GridSystem, stype: str) -> None:
    """Render the slide background image or solid fill."""
    bg_token = spec.get('slide_type_layouts', {}).get(stype, spec.get('slide_type_layouts', {}).get('default', {})).get('background', 'surface')
    bg_image_path = _resolve_bg_image(spec, stype, sd)
    if bg_image_path and os.path.isfile(bg_image_path):
        pic = slide.shapes.add_picture(bg_image_path, Emu(0), Emu(0), Inches(grid.slide_w), Inches(grid.slide_h))
        sp_tree = slide.shapes._spTree
        sp_tree.remove(pic._element)
        sp_tree.insert(2, pic._element)
        overlay = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Emu(0), Emu(0), Inches(grid.slide_w), Inches(grid.slide_h))
        fill = overlay.fill
        fill.solid()
        fill.fore_color.rgb = get_color(spec, bg_token)
        from pptx.oxml.ns import qn as _qn
        solid_fill_elem = fill._fill.find(_qn('a:solidFill'))
        if solid_fill_elem is not None:
            srgb = solid_fill_elem.find(_qn('a:srgbClr'))
            if srgb is None:
                srgb = solid_fill_elem[0] if len(solid_fill_elem) else None
            if srgb is not None:
                try:
                    from lxml import etree
                    alpha = etree.SubElement(srgb, _qn('a:alpha'))
                    alpha.set('val', '40000')
                except Exception:
                    logging.getLogger(__name__).warning(
                        'lxml not available or failed to set overlay alpha; skipping alpha modification'
                    )
        overlay.line.fill.background()
        sp_tree = slide.shapes._spTree
        sp_tree.remove(overlay._element)
        sp_tree.insert(3, overlay._element)
    else:
        bg = slide.background.fill
        bg.solid()
        bg.fore_color.rgb = get_color(spec, bg_token)


def _render_bullets_fallback(slide: Any, sd: Dict, spec: Dict, grid: GridSystem, bar_h: float) -> None:
    """Render simple bullet-list fallback content."""
    top = bar_h + 0.12
    content_bullets = sd.get('content', [])
    bullet_y = top + 0.15
    for i, bullet in enumerate(content_bullets[:8]):
        tb = slide.shapes.add_textbox(Inches(grid.margin_h + 0.1), Inches(bullet_y + i * 0.48), Inches(grid.usable_w - 0.2), Inches(0.45))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        run = p.add_run()
        run.text = f"• {bullet}"
        run.font.size = Pt(get_font_size(spec, 'bullet_text'))
        run.font.color.rgb = get_color(spec, 'on_surface')
        p.line_spacing = 1.5


def _render_footer_and_notes(slide: Any, sd: Dict, spec: Dict, grid: GridSystem, stype: str, sec_title: str, sec_accent_token: str, sec_index: int, sections_len: int, slide_num: int, total_slides: int) -> None:
    """Render bottom bar and speaker notes."""
    if stype not in ('title', 'section_divider'):
        render_bottom_bar(
            slide,
            spec,
            grid,
            sec_title,
            slide_num,
            total_slides,
            section_index=sec_index,
            total_sections=sections_len,
            accent_token=sec_accent_token,
        )
    render_speaker_notes(slide, sd.get('speaker_notes'))


def render_slide(prs: Presentation, sd: Dict, spec: Dict, grid: GridSystem, sections: List[Dict], slide_num: int, total_slides: int) -> None:
    """Render a single slide with full styling (delegates to helpers)."""
    stype = sd.get('slide_type', 'bullet-list')
    slide = prs.slides.add_slide(prs.slide_layouts[6])

    # Section context
    slide_id = sd.get('slide_id', slide_num)
    section = find_section_for_slide(slide_id, sections)
    sec_id = section.get('id', 'A')
    sec_title = section.get('title', '')
    sec_accent_token = spec.get('section_accents', {}).get(sec_id, 'primary')
    sec_index = next((i for i, s in enumerate(sections) if s.get('id') == sec_id), 0)

    # Background
    _render_background(slide, sd, spec, grid, stype)

    # Title bar
    tb_mode = get_title_bar_mode(spec, stype)
    bar_h = 0.0
    if tb_mode != 'none':
        section_label = f"Section {sec_id} · {sec_title}" if sec_title else ''
        bar_h = render_title_bar(slide, spec, grid, sd.get('title', ''), slide_num, total_slides, section_label=section_label, accent_color_token=sec_accent_token, mode=tb_mode)

    # Content (fallback)
    _render_bullets_fallback(slide, sd, spec, grid, bar_h)

    # Footer & notes
    _render_footer_and_notes(slide, sd, spec, grid, stype, sec_title, sec_accent_token, sec_index, len(sections), slide_num, total_slides)
