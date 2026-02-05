const assert = require('assert');
const path = require('path');
const fs = require('fs');
const validator = require('../scripts/validate_reveal');

async function testA11y() {
  const target = path.resolve(__dirname, '../dist/index.html');
  // ensure index exists (build step may be required externally)
  if (!fs.existsSync(target)) {
    throw new Error('index.html missing in tools/dist; run build first');
  }
  // run checker
  await validator.runA11y(target);
  const report = path.resolve(__dirname, '../dist/a11y_report.json');
  if (!fs.existsSync(report)) throw new Error('a11y_report.json not created');
  const data = JSON.parse(fs.readFileSync(report, 'utf8'));
  assert(Array.isArray(data.violations), 'violations array must be present');
  console.log('✔ a11y script ran and produced report');
}

(async () => {
  try {
    await testA11y();
    console.log('ALL A11Y TESTS PASSED');
    process.exit(0);
  } catch (e) {
    console.error('A11Y TEST FAILURE:', e && e.message);
    process.exit(1);
  }
})();