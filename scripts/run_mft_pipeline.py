#!/usr/bin/env python3
"""Run full MFT generation pipeline: repair -> MO checks -> render PPTX -> validate PPTX -> run QA
Usage:
  python3 scripts/run_mft_pipeline.py --session docs/presentations/mft-20260203
"""
import argparse
import subprocess
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--session', required=True, help='Session folder under docs/presentations/')
args = parser.parse_args()

session = Path(args.session)
if not session.exists():
    print('Session folder not found:', session)
    sys.exit(1)

repair_script = session / 'scripts' / 'repair_and_run_mo_checks.py'
if not repair_script.exists():
    print('Repair script not found:', repair_script)
    sys.exit(1)

semantic_repaired = session / 'slides_semantic.repaired.json'
design_spec = session / 'design_spec.json'
output_pptx = session / 'MFT.pptx'
qa_report = session / 'qa_report.json'

# Step 1: repair + MO checks
print('Running repair and MO checks...')
res = subprocess.run(['python3', str(repair_script)], capture_output=True, text=True)
print(res.stdout)
if res.returncode != 0:
    print('Repair script failed:', res.stderr)
    sys.exit(res.returncode)

# Step 2: render PPTX
print('Rendering PPTX...')
cmd = ['python3', 'skills/ppt-generator/bin/generate_pptx.py', '--semantic', str(semantic_repaired), '--design', str(design_spec), '--output', str(output_pptx)]
res = subprocess.run(cmd, capture_output=True, text=True)
print(res.stdout)
if res.returncode != 0:
    print('Renderer failed:', res.stderr)
    sys.exit(res.returncode)

# Step 3: MR validator (pptx validation)
print('Validating PPTX with MR checks...')
validator = Path('scripts/validate_pptx_and_write_report.py')
val_out = session / 'validate_report.json'
res = subprocess.run(['python3', str(validator), str(output_pptx), str(semantic_repaired), str(design_spec), str(val_out)], capture_output=True, text=True)
print(res.stdout)

# Step 4: run QA script
print('Running high-level QA...')
res = subprocess.run(['python3', 'scripts/run_pptx_qa.py', '--out-dir', str(session)], capture_output=True, text=True)
print(res.stdout)
if res.returncode not in (0,2):
    print('QA runner error:', res.stderr)
    sys.exit(res.returncode)

print('Pipeline complete. Reports:')
print(' - semantic repaired:', semantic_repaired)
print(' - PPTX:', output_pptx)
print(' - validate report:', val_out)
print(' - qa report:', qa_report)

sys.exit(0)
