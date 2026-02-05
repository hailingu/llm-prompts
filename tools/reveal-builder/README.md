Reveal Builder (POC)

Quickstart:

1. Install dependencies:
   npm install

2. Build the demo:
   node tools/reveal-builder/build.js

3. Preview locally:
   ./tools/reveal-builder/start.sh

Notes:
- build.js uses `docs/specs/design-spec.reveal.json` to populate CSS variables into `tools/dist/index.html` for the POC.
- This is a minimal scaffold; full feature development (markdown parser, chart rendering) is in subsequent tasks.
