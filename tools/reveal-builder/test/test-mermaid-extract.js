const assert = require('assert');
const path = require('path');
const fs = require('fs');
const render = require('../src/render');

function testMermaidExtraction() {
  const yaml = `reveal_hint:\n  layout: "full-bleed"\n  chart_type: "diagram"\n  mermaid_code: |\n    graph LR\n      A[Start] --> B{Is it OK?}\n      B -- Yes --> C[Proceed]\n      B -- No --> D[Abort]\n  alt_text: "Process flow diagram"`;

  const parsed = render.parseVisualRaw(yaml);
  assert(parsed, 'parseVisualRaw should return an object');
  assert(parsed.mermaid_code, 'mermaid_code should be extracted');
  assert(parsed.mermaid_code.includes('graph LR'), 'mermaid_code should contain the diagram start');
  assert(!parsed.mermaid_code.includes('alt_text'), 'mermaid_code should not include alt_text line');
  console.log('✔ mermaid extraction test passed');
}

function testMermaidWithDifferentIndent() {
  const yaml = `reveal_hint:\n    mermaid_code: |\n      sequenceDiagram\n        participant A\n        participant B\n\n    alt_text: "seq"`;

  const parsed = render.parseVisualRaw(yaml);
  assert(parsed && parsed.mermaid_code, 'should extract mermaid_code with deeper indent');
  assert(parsed.mermaid_code.includes('sequenceDiagram'));
  console.log('✔ mermaid different indent test passed');
}

try {
  testMermaidExtraction();
  testMermaidWithDifferentIndent();
  // Ensure buildSlides uses YAML parser path and emits mermaid block properly
  const render = require('../src/render');
  render.buildSlides();
  const out = fs.readFileSync(path.resolve(__dirname, '../../dist/index.html'), 'utf8');
  assert(out.includes('<div class="mermaid">'), 'index.html should include mermaid div');
  assert(!out.includes('Process flow diagram"'), 'index.html should not include alt_text inline');
  console.log('✔ buildSlides render test passed');

  console.log('ALL TESTS PASSED');
  process.exit(0);
} catch (e) {
  console.error('TEST FAILURE:', e && e.message);
  process.exit(1);
}