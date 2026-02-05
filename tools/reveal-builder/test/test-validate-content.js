const assert = require('assert');
const path = require('path');
const fs = require('fs');
const parser = require('../src/parser');
const validator = require('../scripts/validate_content');

function testValidateContent() {
  const inFile = path.resolve(__dirname, '../../../docs/example-chart-raw.md');
  const outDir = path.resolve(__dirname, '../dist');
  if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });
  const parsed = parser.parseSlidesMd(inFile);
  fs.writeFileSync(path.join(outDir, 'slides.json'), JSON.stringify(parsed, null, 2));

  const out = path.join(outDir, 'qa_report.json');
  const report = validator.runContentChecks(path.join(outDir, 'slides.json'), out);
  assert(report && Array.isArray(report.issues), 'report should contain issues array');
  // Expect speaker notes missing for example-chart-raw.md
  assert(report.issues.some(i => i.rule_id === 'speaker-notes-missing'), 'should flag missing speaker notes');
  console.log('✔ content validator basic test passed');
}

try {
  testValidateContent();
  console.log('ALL VALIDATE CONTENT TESTS PASSED');
  process.exit(0);
} catch (e) {
  console.error('VALIDATE CONTENT TEST FAILURE:', e && e.message);
  process.exit(1);
}