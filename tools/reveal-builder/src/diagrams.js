const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { spawnSync } = require('child_process');

function hashCode(s) {
  return crypto.createHash('md5').update(s).digest('hex').slice(0, 10);
}

/**
 * Try to render mermaid code to SVG using mmdc (mermaid-cli).
 * Returns relative output path (relative to outDir) on success, or null on failure.
 */
function renderMermaidToSvg(mermaidCode, outDir) {
  if (!mermaidCode) return null;
  const id = 'mermaid-' + hashCode(mermaidCode) + '.svg';
  const diagramsDir = path.resolve(outDir, 'diagrams');
  if (!fs.existsSync(diagramsDir)) fs.mkdirSync(diagramsDir, { recursive: true });
  const outPath = path.join(diagramsDir, id);
  if (fs.existsSync(outPath)) return path.join('diagrams', id);

  // write temp mmd
  const tmpMmd = path.join(diagramsDir, id + '.mmd');
  fs.writeFileSync(tmpMmd, mermaidCode, 'utf8');

  // Prefer a global mmdc if available
  let mmdcPath = null;
  try {
    const which = spawnSync('which', ['mmdc']);
    if (which.status === 0) mmdcPath = which.stdout.toString().trim();
  } catch (e) {}

  if (!mmdcPath) {
    // prefer local package binary if installed in node_modules/.bin
    const localBins = [
      path.resolve(__dirname, '..', 'node_modules', '.bin', 'mmdc'),
      path.resolve(__dirname, '..', '..', 'node_modules', '.bin', 'mmdc')
    ];
    for (const p of localBins) {
      try { if (fs.existsSync(p)) { mmdcPath = p; break; } } catch (e) {}
    }
  }

  if (!mmdcPath) {
    // try npx (will fetch if needed)
    try {
      const r = spawnSync('npx', ['-y', '@mermaid-js/mermaid-cli', '-v'], { stdio: 'ignore' });
      if (r.status === 0) mmdcPath = 'npx';
    } catch (e) {}
  }

  if (!mmdcPath) {
    // not available
    try { fs.unlinkSync(tmpMmd); } catch (e) {}
    return null;
  }

  // run mmdc
  let res;
  if (mmdcPath === 'npx') {
    res = spawnSync('npx', ['-y', '@mermaid-js/mermaid-cli', '-i', tmpMmd, '-o', outPath, '--backgroundColor', 'transparent'], { timeout: 20000 });
  } else {
    res = spawnSync(mmdcPath, ['-i', tmpMmd, '-o', outPath, '--backgroundColor', 'transparent'], { timeout: 20000 });
  }

  if (res && res.status !== 0) {
    // surface some debug if available but don't throw
    const err = (res.stderr && res.stderr.toString()) || (res.stdout && res.stdout.toString()) || '';
    console.debug('mmdc failed:', err.trim().slice(0, 300));
  }

  if (res && res.status === 0 && fs.existsSync(outPath)) {
    try { fs.unlinkSync(tmpMmd); } catch (e) {}
    return path.join('diagrams', id);
  }

  // failed
  try { fs.unlinkSync(tmpMmd); } catch (e) {}
  return null;
}

module.exports = { renderMermaidToSvg };
