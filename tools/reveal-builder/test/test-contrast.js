const assert = require('assert');
const path = require('path');
const fs = require('fs');
const checker = require('../scripts/check_contrast');

function testContrastBasics() {
  const specPath = path.resolve(__dirname, '../../..', 'docs', 'specs', 'design-spec.reveal.json');
  const outPath = path.resolve(__dirname, '../dist/contrast_report.json');
  if (!fs.existsSync(path.dirname(outPath))) fs.mkdirSync(path.dirname(outPath), { recursive: true });
  const report = checker.runChecks(specPath, outPath);
  assert(report && Array.isArray(report.results));
  // basic assertions: important pairs exist
  assert(report.results.some(r => r.pair === 'primary/on_primary'), 'primary/on_primary must be checked');
  assert(report.results.some(r => r.pair.startsWith('chart_palette.')), 'chart_palette entries checked');
  // confirm output file exists
  assert(fs.existsSync(outPath), 'contrast_report.json should be written');
  console.log('✔ contrast report test passed');
}

try {
  testContrastBasics();
  console.log('ALL CONTRAST TESTS PASSED');
  process.exit(0);
} catch (e) {
  console.error('CONTRAST TEST FAILURE:', e && e.message);
  process.exit(1);
}
