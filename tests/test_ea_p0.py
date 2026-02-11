import copy
from scripts.exhibit_architect import _upgrade_visual_from_placeholder, _merge_adjacent_single_component_slides


def test_upgrade_visual_from_placeholder_bar():
    sd = {'visual': {'type': 'png', 'placeholder_data': {'categories': ['a','b'], 'series': [{'name':'s','values':[1,2]}]}}}
    _upgrade_visual_from_placeholder(sd)
    assert sd['visual']['type'] in ('bar_chart','line_chart','composite_charts')


def test_merge_adjacent_kpis_chart():
    s1 = {'slide_type':'data-heavy','_section_label':'sec1','components':{'kpis':[{'label':'k1'}]},'visual':{'type':'none'}}
    s2 = {'slide_type':'data-heavy','_section_label':'sec1','components':{},'visual':{'type':'bar_chart','placeholder_data':{}}}
    out = _merge_adjacent_single_component_slides([s1,s2])
    assert len(out)==1
    assert out[0]['visual']['type']=='bar_chart'
    assert 'kpis' in out[0]['components']


def test_merge_adjacent_chart_bullets():
    s1 = {'slide_type':'data-heavy','_section_label':'sec1','components':{},'visual':{'type':'line_chart'}}
    s2 = {'slide_type':'data-heavy','_section_label':'sec1','components':{'bullets':[{'text':'a'}]},'visual':{'type':'none'}}
    out = _merge_adjacent_single_component_slides([s1,s2])
    assert len(out)==1
    assert out[0]['visual']['type']=='line_chart'
    assert 'bullets' in out[0]['components']


def test_merge_adjacent_bullets_concat():
    s1 = {'slide_type':'bullet-list','_section_label':'sec1','components':{'bullets':[{'text':'a'}]}}
    s2 = {'slide_type':'bullet-list','_section_label':'sec1','components':{'bullets':[{'text':'b'}]}}
    out = _merge_adjacent_single_component_slides([s1,s2])
    assert len(out)==1
    assert out[0]['components']['bullets'][0]['text']=='a'
    assert out[0]['components']['bullets'][1]['text']=='b'
