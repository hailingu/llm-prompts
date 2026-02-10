"""Helper utilities for PPT generation.

Contains color/token helpers and typography helpers.
"""
from typing import Dict

from pptx.dml.color import RGBColor

_FONT_SIZE_DEFAULTS = {
    'display_large': 40, 'headline_large': 28, 'title': 22,
    'slide_title': 22, 'slide_subtitle': 16, 'section_label': 10,
    'page_number': 10, 'label': 10, 'label_large': 12,
    'body': 11, 'body_text': 14, 'bullet_text': 14,
    'kpi_value': 20, 'kpi_label': 11,
    'table_header': 12, 'table_cell': 11,
    'callout_text': 13,
}


def hex_to_rgb(hex_str: str) -> RGBColor:
    """Convert a hex color string to PPTX RGBColor."""
    h = hex_str.lstrip('#')
    return RGBColor(int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def get_color(spec: Dict, token_name: str) -> RGBColor:
    """Resolve color token to RGBColor with fallbacks."""
    cs = spec.get('color_system') or spec.get('design_system', {}).get('color_system', {})
    val = cs.get(token_name)
    if val:
        return hex_to_rgb(val)
    val = spec.get('tokens', {}).get('colors', {}).get(token_name)
    if val and isinstance(val, str):
        return hex_to_rgb(val)
    palette = spec.get('theme_tokens', {}).get('palette', {})
    val = palette.get(token_name)
    if val:
        return hex_to_rgb(val)
    _DEFAULTS = {
        'primary': '#2563EB', 'secondary': '#10B981',
        'primary_container': '#E6F0FF', 'on_primary_container': '#0F172A',
        'on_primary': '#FFFFFF', 'surface': '#FFFFFF',
        'surface_variant': '#F3F4F6', 'surface_dim': '#F8FAFC',
        'on_surface': '#0F172A', 'on_surface_variant': '#6B7280',
        'outline': '#D1D5DB', 'muted': '#6B7280',
        'error': '#DC2626', 'warning': '#F59E0B', 'success': '#10B981',
        'accent_1': '#2563EB', 'accent_2': '#10B981',
        'accent_3': '#F59E0B', 'accent_4': '#A78BFA',
    }
    return hex_to_rgb(_DEFAULTS.get(token_name, '#1A1C1E'))


def get_color_hex(spec: Dict, token_name: str) -> str:
    """Return the hex string for a token (fallbacks included)."""
    cs = spec.get('color_system') or spec.get('design_system', {}).get('color_system', {})
    val = cs.get(token_name)
    if val:
        return val
    val = spec.get('tokens', {}).get('colors', {}).get(token_name)
    if val and isinstance(val, str):
        return val
    palette = spec.get('theme_tokens', {}).get('palette', {})
    val = palette.get(token_name)
    if val:
        return val
    _DEFAULTS = {
        'primary': '#2563EB', 'secondary': '#10B981',
        'primary_container': '#E6F0FF', 'on_primary_container': '#0F172A',
        'on_primary': '#FFFFFF', 'surface': '#FFFFFF',
        'surface_variant': '#F3F4F6', 'surface_dim': '#F8FAFC',
        'on_surface': '#0F172A', 'on_surface_variant': '#6B7280',
        'outline': '#D1D5DB', 'muted': '#6B7280',
        'error': '#DC2626', 'warning': '#F59E0B', 'success': '#10B981',
        'accent_1': '#2563EB', 'accent_2': '#10B981',
        'accent_3': '#F59E0B', 'accent_4': '#A78BFA',
    }
    return _DEFAULTS.get(token_name, '#1A1C1E')


def get_font_size(spec: Dict, role: str) -> int:
    """Resolve a font size for a role with legacy and token-based fallbacks."""
    ts = spec.get('typography_system') or spec.get('design_system', {}).get('typography_system', {})
    explicit = ts.get('explicit_sizes', {})
    if role in explicit:
        entry = explicit[role]
        if isinstance(entry, dict):
            return entry.get('size_pt', _FONT_SIZE_DEFAULTS.get(role, 16))
        return entry
    token_ts = spec.get('tokens', {}).get('typography_system', {})
    if role in token_ts:
        entry = token_ts[role]
        if isinstance(entry, dict):
            return entry.get('size_pt', _FONT_SIZE_DEFAULTS.get(role, 16))
        return entry
    scale = ts.get('type_scale', {})
    if role in scale:
        entry = scale[role]
        if isinstance(entry, dict):
            return entry.get('size_pt', 16)
        return entry
    return _FONT_SIZE_DEFAULTS.get(role, 14)


def apply_font_to_run(run, spec: Dict):
    """Apply font styling to a text run based on design spec.
    
    Args:
        run: python-pptx text run object
        spec: Design specification dictionary
    """
    # Get font family from spec
    font_family = None
    if 'fonts' in spec and spec['fonts']:
        for font_entry in spec['fonts']:
            if font_entry.get('role') == 'body':
                font_family = font_entry.get('family', 'Arial')
                break
    
    if not font_family:
        font_family = 'Arial'
    
    # Apply font
    run.font.name = font_family


def px_to_inches(px: float) -> float:
    """Convert pixels (assumed 96dpi) to inches."""
    return px / 96.0
