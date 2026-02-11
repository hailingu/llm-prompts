import json


def test_schema_has_assertion_and_insight():
    schema = json.load(open('standards/slides-render-schema.json', 'r', encoding='utf-8'))
    defs = schema.get('definitions', {})
    slide = defs.get('slide', {})
    props = slide.get('properties', {})
    assert 'assertion' in props, 'assertion field missing in slide properties'
    assert props['assertion']['type'] == 'string'
    assert 'insight' in props, 'insight field missing in slide properties'
    assert props['insight']['type'] == 'string'


def test_schema_version_updated():
    schema = json.load(open('standards/slides-render-schema.json', 'r', encoding='utf-8'))
    assert schema.get('version') == '2.0.0'
    assert schema.get('$id') == 'slides-render-schema-v2'