import os
import sys
ROOT = os.getcwd()
PKG_PATH = os.path.join(ROOT, 'skills', 'ppt-generator')
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if PKG_PATH not in sys.path:
    sys.path.insert(0, PKG_PATH)

from ppt_generator.metrics import audit_metrics, write_metrics
import json


def test_no_warnings():
    m = {
        'assertion_title_rate': 0.85,
        'native_visual_rate': 0.70,
        'compression_ratio': 0.35,
        'placeholder_rate': 0.05
    }
    w = audit_metrics(m)
    assert isinstance(w, list)
    assert len(w) == 0


def test_yellow_and_red_warnings_and_persist(tmp_path, capsys):
    m = {
        'assertion_title_rate': 0.45,  # red
        'native_visual_rate': 0.55,     # yellow
        'compression_ratio': 0.8,       # red
        'placeholder_rate': 0.15        # yellow
    }
    w = audit_metrics(m)
    # should have at least 4 warnings (or fewer if combined) and contain red/yellow hints
    assert any('assertion_title_rate' in ww and 'red line' in ww for ww in w)
    assert any('native_visual_rate' in ww and 'yellow line' in ww for ww in w)

    # ensure warnings are printed to stderr
    captured = capsys.readouterr()
    assert 'AUDIT WARNING' in captured.err

    # persist with write_metrics: warnings should be written into metrics.jsonl
    outdir = str(tmp_path / 'o')
    m['deck_id'] = 'd'
    m['schema_version'] = 2
    m['warnings'] = w
    mf = write_metrics(m, outdir, deck_id='d')
    with open(mf, encoding='utf-8') as f:
        rec = json.loads(f.readline())
    assert 'warnings' in rec['metrics']
    assert isinstance(rec['metrics']['warnings'], list)
