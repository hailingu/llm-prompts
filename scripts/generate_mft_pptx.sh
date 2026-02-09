#!/usr/bin/env bash
set -euo pipefail

# Prepared generator script for MFT presentation
# NOTE: DO NOT RUN until the PPT Creative Director approves the design and visuals.

SEMANTIC="output/slides_semantic.json"
DESIGN="output/design_spec.json"
OUTPUT="docs/presentations/mft-20260206/MFT.pptx"
QA_OUT="docs/presentations/mft-20260206/qa_report.json"
VISUAL_REPORT="output/visual_report.json"

mkdir -p "$(dirname "$OUTPUT")"

echo "Running renderer with:\n  semantic = $SEMANTIC\n  design   = $DESIGN\n  output   = $OUTPUT"

python3 skills/ppt-generator/bin/generate_pptx.py \
  --semantic "$SEMANTIC" \
  --design "$DESIGN" \
  --output "$OUTPUT"

echo "Renderer exit code: $?"

# Post-generation QA (run after renderer completes)
if [ -f "scripts/run_pptx_qa.py" ]; then
  echo "Running QA script to produce $QA_OUT"
  python3 scripts/run_pptx_qa.py "$OUTPUT" "$SEMANTIC" "$DESIGN" --out "$QA_OUT"
  echo "QA results: $QA_OUT"
else
  echo "QA script not found: scripts/run_pptx_qa.py — skip QA step"
fi

echo "Generation script completed. Do not distribute PPTX until Creative Director approves final QA report and visual artifacts."