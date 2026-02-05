const fs = require('fs');
const path = require('path');
let puppeteer = null;
let AxePuppeteer = null;
try {
  puppeteer = require('puppeteer-core');
  ({ AxePuppeteer } = require('axe-puppeteer'));
} catch (e) {
  // Puppeteer not available; we'll fall back to jsdom + axe-core
}
const { execSync } = require('child_process');

function findChromeExecutable() {
  if (process.env.CHROME_PATH) return process.env.CHROME_PATH;
  const candidates = [
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    '/Applications/Chromium.app/Contents/MacOS/Chromium',
    '/usr/bin/google-chrome',
    '/usr/bin/chromium-browser',
    '/usr/bin/chrome',
  ];
  for (const c of candidates) {
    try { if (fs.existsSync(c)) return c; } catch (e) {}
  }
  try {
    const which = execSync('which google-chrome || which google-chrome-stable || which chrome || which chromium-browser || which chromium', { encoding: 'utf8' }).trim();
    if (which) return which;
  } catch (e) {}
  return null;
}

async function runA11y(targetPath) {
  const abs = path.isAbsolute(targetPath) ? targetPath : path.resolve(process.cwd(), targetPath);
  if (!fs.existsSync(abs)) {
    throw new Error(`Target HTML not found: ${abs}`);
  }
  const outDir = path.dirname(abs);
  const reportPath = path.join(outDir, 'a11y_report.json');
  const url = abs.startsWith('http') ? abs : 'file://' + abs;

  // prefer existing system Chrome/Chromium to avoid downloading Chromium during install
  const executablePath = findChromeExecutable();
  const launchOpts = { args: ['--no-sandbox', '--disable-setuid-sandbox'] };
  if (executablePath) launchOpts.executablePath = executablePath;
  if (puppeteer && AxePuppeteer) {
    const browser = await puppeteer.launch(launchOpts);
    const page = await browser.newPage();
    await page.goto(url, { waitUntil: 'networkidle0' });

    const results = await new AxePuppeteer(page).include('body').analyze();
    await browser.close();
    return results;
  }

  // Fallback: use jsdom + axe-core (no browser required)
  const { JSDOM } = require('jsdom');
  const axe = require('axe-core');
  const html = fs.readFileSync(abs, 'utf8');
  const dom = new JSDOM(html);
  const { window } = dom;
  const { document } = window;

  const results = await new Promise((resolve, reject) => {
    axe.run(document, {}, (err, res) => {
      if (err) return reject(err);
      resolve(res);
    });
  });
  return results;

  const counts = results.violations.reduce((acc, v) => {
    acc.total = (acc.total || 0) + 1;
    acc[v.impact] = (acc[v.impact] || 0) + 1;
    return acc;
  }, {});

  const summary = {
    url,
    timestamp: new Date().toISOString(),
    counts,
    violations: results.violations,
    passes: results.passes,
    incomplete: results.incomplete,
    inapplicable: results.inapplicable
  };

  fs.writeFileSync(reportPath, JSON.stringify(summary, null, 2), 'utf8');
  console.log(`Wrote ${reportPath} with ${results.violations.length} violations`);

  const hasCritical = results.violations.some(v => (v.impact === 'critical' || v.impact === 'serious'));
  if (hasCritical) {
    console.error('Critical/Serious accessibility violations found');
    process.exit(2);
  }
  process.exit(0);
}

if (require.main === module) {
  const target = process.argv[2] || path.resolve(__dirname, '../dist/index.html');
  runA11y(target).catch(err => {
    console.error('A11Y check failed:', err && err.message || err);
    process.exit(1);
  });
}

module.exports = { runA11y };
