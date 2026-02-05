const fs = require('fs');
const path = require('path');
const { spawnSync, execSync } = require('child_process');

function findChromeExecutable() {
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

function copyRecursive(src, dest) {
  if (!fs.existsSync(src)) return;
  const stat = fs.statSync(src);
  if (stat.isDirectory()) {
    if (!fs.existsSync(dest)) fs.mkdirSync(dest, { recursive: true });
    for (const f of fs.readdirSync(src)) copyRecursive(path.join(src, f), path.join(dest, f));
  } else {
    fs.copyFileSync(src, dest);
  }
}

function usage() {
  console.log('Usage: node scripts/generate_previews.js --deck <path/to/index.html> --out <outDir> [--slides N]');
}

async function main() {
  const args = process.argv.slice(2);
  let deck = null; let out = null; let slides = null;
  for (let i=0;i<args.length;i++){
    if (args[i] === '--deck') deck = args[++i];
    if (args[i] === '--out') out = args[++i];
    if (args[i] === '--slides') slides = parseInt(args[++i],10);
  }
  if (!deck || !out) { usage(); process.exit(2); }
  const absDeck = path.resolve(deck);
  if (!fs.existsSync(absDeck)) { console.error('Deck not found', absDeck); process.exit(2); }
  const outDir = path.resolve(out);
  if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });

  // copy deck files (index.html + theme.css + diagrams if present)
  const deckDir = path.dirname(absDeck);
  copyRecursive(deckDir, outDir);

  // copy reveal-builder QA artifacts if present
  const rbDist = path.resolve(__dirname, '../dist');
  const qaFiles = ['qa_report.json','a11y_report.json','contrast_report.json'];
  for (const f of qaFiles) {
    const p = path.join(rbDist, f);
    if (fs.existsSync(p)) fs.copyFileSync(p, path.join(outDir, f));
  }

  // count slides from slides.json if present
  let nslides = slides;
  const slidesJsonPath = path.join(deckDir, 'slides.json');
  if (!nslides && fs.existsSync(slidesJsonPath)) {
    const s = JSON.parse(fs.readFileSync(slidesJsonPath,'utf8'));
    nslides = s.slides ? s.slides.length : 1;
  }
  if (!nslides) nslides = 1;
  // clamp
  nslides = Math.min(nslides, 10);

  const chrome = findChromeExecutable();
  if (!chrome) {
    console.warn('No system Chrome/Chromium found; cannot generate PNG previews.');
    process.exit(0);
  }

  console.log('Using chrome', chrome);
  // generate screenshots for first nslides
  for (let i=0;i<nslides;i++) {
    const slideUrl = 'file://' + path.join(outDir, 'index.html') + '#/' + i;
    const outPng = path.join(outDir, `preview-slide-${i+1}.png`);
    console.log('Capturing slide', i+1, 'to', outPng);
    const res = spawnSync(chrome, ['--headless', '--disable-gpu', `--screenshot=${outPng}`, '--window-size=1280,720', slideUrl], { timeout: 20000 });
    if (res.error) console.warn('Screenshot failed for slide', i+1, res.error.message);
    if (res.status !== 0) console.warn('Chrome exited with status', res.status, res.stderr && res.stderr.toString().slice(0,300));
  }

  console.log('Generated previews in', outDir);
}

if (require.main === module) main().catch(e=>{ console.error(e); process.exit(1); });

module.exports = { findChromeExecutable, copyRecursive };
