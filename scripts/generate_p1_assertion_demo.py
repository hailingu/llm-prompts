"""Generate a demo PPTX using the packaged renderers to validate Task 2.4 (assertion + insight rendering)."""
import json
import os
from pptx import Presentation
from pptx.util import Inches

# Import the packaged renderers module (ensure package import context)
import sys
sys.path.insert(0, os.path.abspath('skills/ppt-generator'))
from ppt_generator import renderers

SEM = 'docs/presentations/storage-frontier-20260211/slides_semantic.json'
DES = 'docs/presentations/storage-frontier-20260211/design_spec.json'
OUT = 'docs/presentations/storage-frontier-20260211/storage-frontier-v10-assertion-packaged.pptx'

semantic = json.load(open(SEM, encoding='utf-8'))
design = json.load(open(DES, encoding='utf-8'))

grid = renderers.GridSystem(design)

prs = Presentation()
prs.slide_width = Inches(grid.slide_w)
prs.slide_height = Inches(grid.slide_h)

sections = semantic.get('sections', [])
slides = semantic.get('slides', [])

for i, sd in enumerate(slides, 1):
    # Inject deck title for title slide
    if sd.get('slide_type') == 'title':
        sd['_deck_title'] = semantic.get('title', '')
    renderers.render_slide(prs, sd, design, grid, sections, i, len(slides))

os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
prs.save(OUT)
print('Saved packaged PPTX:', OUT)
