const assert = require('assert');
const path = require('path');
const fs = require('fs');
const diagrams = require('../src/diagrams');

async function testRenderMermaid() {
  const code = `graph LR\n  A --> B\n  B --> C`;
  const outDir = path.resolve(__dirname, '../dist');
  if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });
  const rel = diagrams.renderMermaidToSvg(code, outDir);
  if (!rel) {
    throw new Error('mermaid-cli (mmdc) not available; install @mermaid-js/mermaid-cli in tools/reveal-builder to enable server-side rendering');
  }
  const svgPath = path.resolve(outDir, rel);
  assert(fs.existsSync(svgPath), 'SVG file should exist after rendering');
  const content = fs.readFileSync(svgPath, 'utf8');
  assert(content.trim().startsWith('<svg'), 'Rendered file should be SVG');
  console.log('✔ mermaid server-side render test passed');
}

try {
  testRenderMermaid();
  console.log('ALL DIAGRAM TESTS PASSED');
  process.exit(0);
} catch (e) {
  console.error('DIAGRAM TEST FAILURE:', e && e.message);
  process.exit(1);
}