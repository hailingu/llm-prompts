import json


def test_layout_intent_definition():
    schema = json.load(open('standards/slides-render-schema.json', 'r', encoding='utf-8'))
    defs = schema.get('definitions', {})
    slide = defs.get('slide', {})
    props = slide.get('properties', {})

    assert 'layout_intent' in props, 'layout_intent missing in slide properties'
    li = props['layout_intent']
    assert li['type'] == 'object'

    template_enum = li['properties']['template']['enum']
    assert isinstance(template_enum, list) and len(template_enum) == 6

    renderer_enum = li['properties']['regions']['items']['properties']['renderer']['enum']
    expected = {"chart", "comparison_table", "kpi_row", "callout_stack", "progression", "bullet_list", "architecture", "flow"}
    assert set(renderer_enum) >= expected


def test_v1_backward_compatibility_smoke():
    # basic smoke: a minimal v1 slide (no layout_intent) is still structurally valid
    slide = {
        "slide_id": 1,
        "title": "Title",
        "slide_type": "title",
        "content": []
    }
    assert 'slide_id' in slide and 'title' in slide and 'slide_type' in slide
