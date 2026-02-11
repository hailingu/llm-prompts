import json
import subprocess
from pathlib import Path

BASE = Path('docs/presentations/storage-frontier-20260211')
SEM_IN = BASE / 'slides_semantic.json'
SEM_V2 = BASE / 'slides_semantic_v2.json'
AUDIT = BASE / 'ea_audit.json'
PPTX = BASE / 'storage-frontier-v2-ea.pptx'
CLI = Path('skills/ppt-generator/bin/generate_pptx.py')


def test_ea_smoke_and_render(tmp_path):
    # Run EA smoke
    subprocess.check_call(['python3', 'scripts/ea_smoke.py'])
    assert SEM_V2.exists(), 'v2 semantic not created'
    assert AUDIT.exists(), 'ea_audit not created'

    audit = json.load(open(AUDIT, encoding='utf-8'))
    summary = audit.get('summary', {})
    assert 'compression_ratio' in summary, 'audit summary missing compression_ratio'
    # Assert compression ratio is <= 0.85 (lenient) and track for CI
    assert summary['compression_ratio'] <= 0.85, f"Compression ratio too high: {summary['compression_ratio']}"
    # Assert assertion coverage >= 0.7
    assert summary['assertion_coverage'] >= 0.7, f"Low assertion coverage: {summary['assertion_coverage']}"

    # Generate PPTX from v2
    subprocess.check_call(['python3', str(CLI), '--semantic', str(SEM_V2), '--design', str(BASE / 'design_spec.json'), '--output', str(PPTX)])
    assert PPTX.exists(), 'PPTX not generated from v2 semantic'
