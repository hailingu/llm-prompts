from typing import Any, Dict, List
import os
import sys


def _count_components(components: Dict) -> int:
    if not components or not isinstance(components, dict):
        return 0
    c = 0
    for v in components.values():
        if isinstance(v, list):
            c += len(v)
        elif isinstance(v, dict):
            # treat dict as one component container
            c += 1
        else:
            c += 1
    return c


def _has_shape_component(components: Dict) -> bool:
    if not components or not isinstance(components, dict):
        return False
    for v in components.values():
        if isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    if 'nodes' in item or 'steps' in item:
                        return True
        elif isinstance(v, dict):
            if 'nodes' in v or 'steps' in v:
                return True
    return False


def compute_deck_metrics(semantic: Dict[str, Any], rendered_info: Dict[str, Any] = None) -> Dict[str, float]:
    """Compute deck-level metrics described in Task 6.1.

    Parameters
    ----------
    semantic: dict
        The slides semantic JSON (top-level dict with 'slides' list).
    rendered_info: dict, optional
        Optional runtime rendering info (not required for the basic metrics).

    Returns
    -------
    dict: {
      'assertion_title_rate': float,  # 0..1
      'native_visual_rate': float,    # 0..1
      'compression_ratio': float,     # slides / input_paragraphs (>=0)
      'placeholder_rate': float,      # 0..1
      'multi_region_rate': float,     # 0..1
      'avg_components_per_slide': float,
    }

    Notes
    -----
    This function is best-effort and intentionally conservative: it makes
    reasonable inferences from the semantic JSON alone and does not require
    rendered_info to be present. Empty decks return zeros for all metrics.
    """
    slides: List[Dict] = semantic.get('slides', []) if isinstance(semantic, dict) else []
    total_slides = len(slides)

    if total_slides == 0:
        return {
            'assertion_title_rate': 0.0,
            'native_visual_rate': 0.0,
            'compression_ratio': 0.0,
            'placeholder_rate': 0.0,
            'multi_region_rate': 0.0,
            'avg_components_per_slide': 0.0,
        }

    assertion_count = 0
    native_visual_count = 0
    placeholder_count = 0
    multi_region_count = 0
    total_components = 0
    total_input_paragraphs = 0

    for s in slides:
        if s.get('assertion'):
            assertion_count += 1

        # Visual classification
        visual = s.get('visual') or {}
        pd = visual.get('placeholder_data', {}) if isinstance(visual, dict) else {}
        has_rendered = bool(visual.get('rendered_png_path') or visual.get('rendered_svg_path')) if isinstance(visual, dict) else False
        has_chart_config = bool(pd.get('chart_config'))
        has_mermaid = bool(pd.get('mermaid_code'))

        if has_chart_config and not has_rendered:
            # chart placeholder with config and no pre-rendered asset -> native chart candidate
            native_visual_count += 1
        elif _has_shape_component(s.get('components', {})):
            # slides containing architecture/flow definitions
            native_visual_count += 1

        # Placeholder detection: visual type none or missing placeholder data and no rendered asset
        v_type = visual.get('type') if isinstance(visual, dict) else None
        if v_type == 'none' or (not has_rendered and not has_chart_config and not has_mermaid and v_type is not None):
            placeholder_count += 1

        # multi-region detection
        layout = s.get('layout_intent') or {}
        regions = layout.get('regions') if isinstance(layout, dict) else None
        if isinstance(regions, list) and len(regions) >= 2:
            multi_region_count += 1

        # components counting
        comps = s.get('components') or {}
        comp_count = _count_components(comps)
        total_components += comp_count

        # input paragraph heuristic: count content paragraphs + components list lengths
        content_count = len(s.get('content', []) or [])
        comp_text_count = 0
        if isinstance(comps, dict):
            for v in comps.values():
                if isinstance(v, list):
                    # Count only textual list items (e.g., bullets, callouts, kpi entries)
                    for item in v:
                        if isinstance(item, dict):
                            # Skip complex shape-like objects (architecture/flow)
                            if 'nodes' in item or 'steps' in item:
                                continue
                            # Treat other dicts as one textual unit
                            comp_text_count += 1
                        else:
                            comp_text_count += 1
                elif isinstance(v, dict):
                    # If dict contains shape-like keys, skip
                    if 'nodes' in v or 'steps' in v:
                        comp_text_count += 0
                    else:
                        comp_text_count += 1
                else:
                    comp_text_count += 1
        total_input_paragraphs += content_count + comp_text_count

    assertion_title_rate = assertion_count / total_slides
    native_visual_rate = native_visual_count / total_slides
    placeholder_rate = placeholder_count / total_slides
    multi_region_rate = multi_region_count / total_slides
    avg_components_per_slide = total_components / total_slides
    # Guard denominator for compression ratio
    denom = total_input_paragraphs if total_input_paragraphs > 0 else 1
    compression_ratio = total_slides / denom

    result = {
        'assertion_title_rate': assertion_title_rate,
        'native_visual_rate': native_visual_rate,
        'compression_ratio': compression_ratio,
        'placeholder_rate': placeholder_rate,
        'multi_region_rate': multi_region_rate,
        'avg_components_per_slide': avg_components_per_slide,
    }
    # Include some light metadata
    result_meta = {
        'total_slides': total_slides,
        'total_input_paragraphs': total_input_paragraphs,
    }
    return result | result_meta


def write_metrics(metrics_dict: Dict, output_dir: str, deck_id: str = None) -> str:
    """Append metrics JSON line to `metrics.jsonl` in output_dir.

    Returns the path to the metrics file written.
    """
    import json
    from datetime import datetime
    if not output_dir:
        raise ValueError('output_dir must be provided')
    os.makedirs(output_dir, exist_ok=True)
    metrics_file = os.path.join(output_dir, 'metrics.jsonl')

    record = {
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'deck_id': deck_id or metrics_dict.get('deck_id'),
        'metrics': metrics_dict,
    }
    with open(metrics_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')
    return metrics_file


def audit_metrics(metrics: Dict[str, Any]) -> List[str]:
    """Run audit checks on metrics and return list of warning strings.

    Warnings are printed to stderr and should be attached to the metrics record
    under the `warnings` key when persisted.
    """
    warnings: List[str] = []
    if not isinstance(metrics, dict):
        return warnings

    def _fmt_val(v):
        try:
            return f"{float(v):.2f}"
        except Exception:
            return str(v)

    # Rules config: (metric_key, yellow_check, red_check, yellow_hint, red_hint)
    rules = [
        ('assertion_title_rate', lambda v: v < 0.70, lambda v: v < 0.50, 'yellow line: <0.70', 'red line: <0.50'),
        ('native_visual_rate', lambda v: v < 0.60, lambda v: v < 0.40, 'yellow line: <0.60', 'red line: <0.40'),
        ('compression_ratio', lambda v: v > 0.50, lambda v: v > 0.70, 'yellow line: >0.50', 'red line: >0.70'),
        ('placeholder_rate', lambda v: v > 0.10, lambda v: v > 0.20, 'yellow line: >0.10', 'red line: >0.20'),
    ]

    for key, yellow_fn, red_fn, yellow_hint, red_hint in rules:
        val = metrics.get(key)
        if val is None:
            continue
        try:
            v = float(val)
        except Exception:
            continue
        # Check red first
        if red_fn(v):
            msg = f"{key}={_fmt_val(v)} ({red_hint})"
            full = f"⚠️ AUDIT WARNING: {msg}"
            warnings.append(full)
            print(full, file=sys.stderr)
        elif yellow_fn(v):
            msg = f"{key}={_fmt_val(v)} ({yellow_hint})"
            full = f"⚠️ AUDIT WARNING: {msg}"
            warnings.append(full)
            print(full, file=sys.stderr)

    return warnings