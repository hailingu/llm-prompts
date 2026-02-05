const assert = require('assert');
const path = require('path');
const fs = require('fs');
const parser = require('../src/parser');
const render = require('../src/render');

function testChartA11y() {
  const inFile = path.resolve(__dirname, '../../../docs/example-chart-raw.md');
  const parsed = parser.parseSlidesMd(inFile);
  // renderer reads slides.json from tools/dist
  const outDir = path.resolve(__dirname, '../../dist');
  if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });
  fs.writeFileSync(path.join(outDir, 'slides.json'), JSON.stringify(parsed, null, 2), 'utf8');

  // build slides
  render.buildSlides();
  const html = fs.readFileSync(path.resolve(__dirname, '../../dist/index.html'), 'utf8');

  // check for canvas with aria-describedby linking to table id
  const canvasMatch = html.match(/<canvas[^>]*id="(chart-\d+)"[^>]*aria-describedby="(chart-\d+-data)"/);
  assert(canvasMatch, 'Canvas with aria-describedby to data table should exist');
  assert(canvasMatch[1] === canvasMatch[2].replace(/-data$/, ''), 'Canvas id and table id should correspond');

  // check table exists and has id
  const tableMatch = html.match(/<table[^>]*id="(chart-\d+-data)"[^>]*>/);
  assert(tableMatch, 'Data table element with expected id should exist');

  console.log('✔ chart a11y render test passed');
}

try {
  testChartA11y();
  console.log('ALL CHART A11Y TESTS PASSED');
  process.exit(0);
} catch (e) {
  console.error('CHART A11Y TEST FAILURE:', e && e.message);
  process.exit(1);
}