const fs = require('fs');
const path = require('path');
const child_process = require('child_process');

const specPath = path.resolve(__dirname, '../../docs/specs/design-spec.reveal.json');
const deckTemplatePath = path.resolve(__dirname, 'templates/reveal/deck.hbs');
const themePath = path.resolve(__dirname, 'templates/reveal/theme.css');
const renderScript = path.resolve(__dirname, 'src/render.js');
let outDir = path.resolve(__dirname, '../dist');

// Basic CLI parsing
const args = process.argv.slice(2);
let inputFile = null;
let exportPdf = false;
let pdfOut = null;

for (let i = 0; i < args.length; i++) {
  const a = args[i];
  if (a === '--input' || a === '-i') { inputFile = args[++i]; }
  else if (a === '--out' || a === '-o') { outDir = path.resolve(args[++i]); }
  else if (a === '--export-pdf') { exportPdf = true; }
  else if (a === '--pdf-out') { pdfOut = args[++i]; }
}

if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });

// Parse slides if input is provided
if (inputFile) {
  console.log('Parsing slides from', inputFile);
  const parser = require('./src/parser.js');
  const parsed = parser.parseSlidesMd(inputFile);
  fs.writeFileSync(path.join(outDir, 'slides.json'), JSON.stringify(parsed, null, 2), 'utf8');
  console.log('Wrote slides.json to', path.join(outDir, 'slides.json'));
}

const spec = JSON.parse(fs.readFileSync(specPath, 'utf8'));

// Build theme.css by prepending :root vars from tokens and typography
let theme = fs.readFileSync(themePath, 'utf8');
const cssVars = [];
for (const [k, v] of Object.entries(spec.tokens)) {
  cssVars.push(`--color-${k}: ${v};`);
}
for (const [k, v] of Object.entries(spec.typography.type_scale)) {
  cssVars.push(`--type-${k}: ${v.size}px;`);
}
const varsBlock = `:root {\n  ${cssVars.join('\n  ')}\n}\n`;
fs.writeFileSync(path.join(outDir, 'theme.css'), varsBlock + theme, 'utf8');

// If slides.json exists, render the deck HTML using the render script
const slidesJsonPath = path.resolve(outDir, 'slides.json');
if (fs.existsSync(slidesJsonPath)) {
  require(renderScript);
  console.log('Rendered ' + path.join(outDir, 'index.html') + ' from slides.json');
} else {
  // fallback: copy deck.hbs as static HTML for now (simple placeholder)
  const placeholder = fs.readFileSync(deckTemplatePath, 'utf8');
  fs.writeFileSync(path.join(outDir, 'index.html'), placeholder, 'utf8');
  console.log('Wrote placeholder ' + path.join(outDir,'index.html') + ' (slides.json missing)');
}

console.log('Built ' + outDir + ' (index.html + theme.css)');

async function exportPdfFunc() {
  try {
    const puppeteer = require('puppeteer');
    const browser = await puppeteer.launch({args: ['--no-sandbox', '--disable-setuid-sandbox']});
    const page = await browser.newPage();
    const indexPath = 'file://' + path.resolve(outDir, 'index.html');
    await page.goto(indexPath, { waitUntil: 'networkidle0' });
    const pdfPath = pdfOut ? path.resolve(pdfOut) : path.resolve(outDir, 'presentation.pdf');
    await page.pdf({ path: pdfPath, format: 'A4', printBackground: true });
    await browser.close();
    console.log('Exported PDF to', pdfPath);
  } catch (e) {
    console.error('Failed to export PDF. Please run `npm install puppeteer` and try again. Error:', e.message);
    process.exit(1);
  }
}

if (exportPdf) {
  exportPdfFunc().catch(e => { console.error(e); process.exit(1); });
}

