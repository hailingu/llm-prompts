import os
import sys
ROOT = os.getcwd()
PKG_PATH = os.path.join(ROOT, 'skills', 'ppt-generator')
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if PKG_PATH not in sys.path:
    sys.path.insert(0, PKG_PATH)

import json
import tempfile
from ppt_generator.metrics import write_metrics


def test_write_metrics_appends_and_creates_file(tmp_path):
    md = {'assertion_title_rate': 0.5}
    outdir = str(tmp_path / 'out')
    mf = write_metrics(md, outdir, deck_id='test-deck')
    assert os.path.exists(mf)
    with open(mf, encoding='utf-8') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec['deck_id'] == 'test-deck'
    assert 'timestamp' in rec
    assert rec['metrics']['assertion_title_rate'] == 0.5


def test_write_metrics_appends_multiple(tmp_path):
    md1 = {'a': 1}
    md2 = {'b': 2}
    outdir = str(tmp_path / 'out')
    mf = write_metrics(md1, outdir, deck_id='d')
    mf2 = write_metrics(md2, outdir, deck_id='d')
    assert mf == mf2
    with open(mf, encoding='utf-8') as f:
        lines = [l for l in f.readlines() if l.strip()]
    assert len(lines) == 2


def test_integration_cli_generates_metrics(tmp_path):
    # integration: run generate_pptx on the storage-frontier v2 sample and check metrics.jsonl
    import json
    from ppt_generator.cli import generate_pptx

    base = 'docs/presentations/storage-frontier-20260211'
    semantic = os.path.join(base, 'slides_semantic_v2.json')
    design = os.path.join(base, 'design_spec.json')
    out = str(tmp_path / 'storage-frontier-test.pptx')
    generate_pptx(semantic, design, out)
    metrics_file = os.path.join(os.path.dirname(out), 'metrics.jsonl')
    assert os.path.exists(metrics_file)
    with open(metrics_file, encoding='utf-8') as f:
        content = [l.strip() for l in f if l.strip()]
    # since generate_pptx appends a line, assert at least one line and basic fields
    assert len(content) >= 1
    rec = json.loads(content[-1])
    assert 'metrics' in rec
    assert 'schema_version' in rec['metrics']
    assert 'total_slides' in rec['metrics']
