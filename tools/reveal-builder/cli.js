#!/usr/bin/env node
const { spawnSync } = require('child_process');
const path = require('path');
const fs = require('fs');

const cmd = process.argv[2] || 'help';

function runNodeScript(scriptPath, args=[]) {
  const nodeArgs = [path.resolve(__dirname, scriptPath), ...args];
  const r = spawnSync('node', nodeArgs, { stdio: 'inherit' });
  if (r.error) throw r.error;
  return r.status;
}

function runShell(scriptPath, args=[]) {
  const r = spawnSync('/bin/bash', [path.resolve(__dirname, scriptPath), ...args], { stdio: 'inherit' });
  if (r.error) throw r.error;
  return r.status;
}

switch (cmd) {
  case 'build': {
    // Example: node cli.js build --input docs/MFT_slides.md --export-pdf
    const extra = process.argv.slice(3);
    const status = runNodeScript('build.js', extra);
    process.exit(status);
  }
  case 'start': {
    // forward flags (e.g., --bg)
    const extra = process.argv.slice(3);
    const status = runShell('start.sh', extra);
    process.exit(status);
  }
  case 'clean': {
    const dist = path.resolve(__dirname, '../dist');
    if (fs.existsSync(dist)) {
      console.log('Removing', dist);
      const rimraf = require('child_process').spawnSync('rm', ['-rf', dist], { stdio: 'inherit' });
      if (rimraf.error) throw rimraf.error;
      console.log('Removed', dist);
    } else console.log('Nothing to clean');
    process.exit(0);
  }
  case 'export-pdf': {
    const extra = process.argv.slice(3);
    // reuse build's export-pdf path
    const status = runNodeScript('build.js', ['--export-pdf', ...extra]);
    process.exit(status);
  }
  default:
    console.log('Usage: cli.js <command> [args]\n');
    console.log('Commands:');
    console.log('  build [--input <slides.md>] [--out <dir>] [--export-pdf]  Build the deck (optionally export PDF)');
    console.log('  start [--bg]                                             Start local server (foreground by default)');
    console.log('  clean                                                    Remove generated dist');
    console.log('  export-pdf [--pdf-out <file>]                            Export PDF from built deck');
    process.exit(0);
}
