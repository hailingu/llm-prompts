const fs = require('fs');
const path = require('path');

function hexToRgb(hex) {
  if (!hex) return null;
  const h = hex.replace('#','').trim();
  if (h.length === 3) {
    return {
      r: parseInt(h[0]+h[0],16),
      g: parseInt(h[1]+h[1],16),
      b: parseInt(h[2]+h[2],16)
    };
  }
  return { r: parseInt(h.slice(0,2),16), g: parseInt(h.slice(2,4),16), b: parseInt(h.slice(4,6),16) };
}

function srgb2linear(c) {
  const s = c / 255;
  return s <= 0.04045 ? s / 12.92 : Math.pow((s + 0.055) / 1.055, 2.4);
}

function luminance(hex) {
  const rgb = hexToRgb(hex);
  const r = srgb2linear(rgb.r);
  const g = srgb2linear(rgb.g);
  const b = srgb2linear(rgb.b);
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

function contrastRatio(hex1, hex2) {
  const l1 = luminance(hex1);
  const l2 = luminance(hex2);
  const lighter = Math.max(l1,l2);
  const darker = Math.min(l1,l2);
  return +( (lighter + 0.05) / (darker + 0.05) ).toFixed(2);
}

// Color-blind simulation matrices (Brettel-esque approximations)
const SIM_MATRICES = {
  protanopia: [
    [0.56667, 0.43333, 0.0],
    [0.55833, 0.44167, 0.0],
    [0.0, 0.24167, 0.75833]
  ],
  deuteranopia: [
    [0.625, 0.375, 0.0],
    [0.7, 0.3, 0.0],
    [0.0, 0.3, 0.7]
  ],
  tritanopia: [
    [0.95, 0.05, 0.0],
    [0.0, 0.43333, 0.56667],
    [0.0, 0.475, 0.525]
  ]
};

function applyMatrix(rgb, mat) {
  const r = (rgb.r*mat[0][0] + rgb.g*mat[0][1] + rgb.b*mat[0][2]);
  const g = (rgb.r*mat[1][0] + rgb.g*mat[1][1] + rgb.b*mat[1][2]);
  const b = (rgb.r*mat[2][0] + rgb.g*mat[2][1] + rgb.b*mat[2][2]);
  // clamp to [0,255]
  return { r: Math.min(255, Math.max(0, Math.round(r))), g: Math.min(255, Math.max(0, Math.round(g))), b: Math.min(255, Math.max(0, Math.round(b))) };
}

function rgbToHex(rgb) {
  const toHex = (n) => ('0'+n.toString(16)).slice(-2);
  return `#${toHex(rgb.r)}${toHex(rgb.g)}${toHex(rgb.b)}`;
}

function simulate(hex, kind) {
  const rgb = hexToRgb(hex);
  // matrices assume 0-255 input
  const m = SIM_MATRICES[kind];
  const out = applyMatrix(rgb, m);
  return rgbToHex(out);
}

function runChecks(specPath, outPath) {
  const spec = JSON.parse(fs.readFileSync(specPath, 'utf8'));
  const report = { specPath, timestamp: new Date().toISOString(), results: [] };
  const minContrast = (spec.accessibility && spec.accessibility.minContrast) || 4.5;
  const diagContrast = (spec.accessibility && spec.accessibility.diagramContrast) || 7.0;

  // pairs to check: tokens primary/on_primary, surface/on_surface, primary/on_surface, text on primary, chart_palette entries vs surface
  const pairs = [];
  if (spec.tokens && spec.tokens.primary && spec.tokens.on_primary) pairs.push({ name:'primary/on_primary', fg: spec.tokens.on_primary, bg: spec.tokens.primary, threshold: minContrast });
  if (spec.tokens && spec.tokens.surface && spec.tokens.on_surface) pairs.push({ name:'surface/on_surface', fg: spec.tokens.on_surface, bg: spec.tokens.surface, threshold: minContrast });
  if (spec.tokens && spec.tokens.primary && spec.tokens.on_surface) pairs.push({ name:'primary/on_surface', fg: spec.tokens.primary, bg: spec.tokens.surface, threshold: minContrast });

  const palette = spec.chart_palette || {};
  for (const k of Object.keys(palette)) {
    pairs.push({ name:`chart_palette.${k} / surface`, fg: palette[k], bg: spec.tokens && spec.tokens.surface || '#FFFFFF', threshold: diagContrast });
    pairs.push({ name:`chart_palette.${k} / on_surface`, fg: palette[k], bg: spec.tokens && spec.tokens.on_surface || '#000000', threshold: minContrast });
  }

  for (const p of pairs) {
    const ratio = contrastRatio(p.fg, p.bg);
    const ok = ratio >= (p.threshold || minContrast);
    const item = { pair: p.name, fg: p.fg, bg: p.bg, ratio, threshold: p.threshold, pass: ok };
    // simulate color blindness
    item.simulations = {};
    for (const sim of Object.keys(SIM_MATRICES)) {
      const fgSim = simulate(p.fg, sim);
      const bgSim = simulate(p.bg, sim);
      const rSim = contrastRatio(fgSim, bgSim);
      item.simulations[sim] = { fgSim, bgSim, ratio: rSim, pass: rSim >= (p.threshold || minContrast) };
    }
    report.results.push(item);
  }

  fs.writeFileSync(outPath, JSON.stringify(report, null, 2), 'utf8');
  return report;
}

if (require.main === module) {
  // resolve spec from repository root (scripts/ is under tools/reveal-builder/scripts)
  const specPath = path.resolve(__dirname, '../../..', 'docs', 'specs', 'design-spec.reveal.json');
  const outPath = path.resolve(__dirname, '..', 'dist', 'contrast_report.json');
  try {
    const r = runChecks(specPath, outPath);
    console.log('Wrote contrast report to', outPath);
    const fails = r.results.filter(x => !x.pass || Object.values(x.simulations).some(s => !s.pass));
    if (fails.length) {
      console.warn('Found contrast issues:', fails.map(f => f.pair));
      process.exit(2);
    }
    console.log('All contrast checks passed');
    process.exit(0);
  } catch (e) {
    console.error('Contrast check failed:', e && e.message || e);
    process.exit(1);
  }
}

module.exports = { runChecks, contrastRatio, simulate };
