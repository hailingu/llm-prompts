import json
import os

ROOT = os.getcwd()
SCHEMA_PATH = os.path.join(ROOT, 'standards', 'slides-render-schema.json')

def test_schema_has_shape_component_definitions():
    d = json.load(open(SCHEMA_PATH))
    comps = d['definitions']['components']['properties']
    assert 'architecture_data' in comps
    assert 'flow_data' in comps

    arch = comps['architecture_data']
    # architecture_data should define nodes and edges
    assert 'nodes' in arch['properties']
    assert 'edges' in arch['properties']

    flow = comps['flow_data']
    assert 'steps' in flow['properties']
    assert 'transitions' in flow['properties']


if __name__ == '__main__':
    print('Run: python3 -m pytest -q tests/test_schema_shapes.py')
