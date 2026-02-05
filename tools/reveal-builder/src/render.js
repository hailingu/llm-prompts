const fs = require('fs');
const path = require('path');

const slidesJsonPath = path.resolve(__dirname, '../../dist/slides.json');
const templatePath = path.resolve(__dirname, '../templates/reveal/deck.hbs');
const outDir = path.resolve(__dirname, '../../dist');

if (!fs.existsSync(slidesJsonPath)) {
  if (require.main === module) {
    console.error('slides.json not found. Run parser first (src/parser.js)');
    process.exit(2);
  }
  // When required as a module (e.g., tests), do not abort; functions are exported for testing.
}

let templateSrc = fs.readFileSync(templatePath, 'utf8');

// Try to parse the VISUAL YAML into structured data using js-yaml when available.
let yamlParser = null;
try {
  yamlParser = require('js-yaml');
} catch (e) {
  try {
    // try resolve relative to package
    const p = require.resolve('js-yaml', { paths: [path.resolve(__dirname, '..')] });
    yamlParser = require(p);
  } catch (e2) {
    yamlParser = null; // optional dependency; we have a fallback
  }
}

function extractChartConfigFromYaml(yamlText) {
  if (!yamlText) return null;
  const cfg = {};
  // labels as JSON array
  const labelsMatch = yamlText.match(/chart_config:[\s\S]*?labels:\s*(\[[^\]]*\])/i);
  if (labelsMatch) {
    try { cfg.labels = JSON.parse(labelsMatch[1]); } catch (e) {
      // try to sanitize quotes and retry
      try { cfg.labels = JSON.parse(labelsMatch[1].replace(/\"|\“|\”/g,'"')); } catch (e2) {}
    }
  }
  // series: looks for '- name: "rows"' followed by data: [[..],[..]]
  const seriesMatch = yamlText.match(/series:[\s\S]*?-\s*name:\s*"?([^\n"]+)"?[^\n]*[\s\S]*?data:\s*(\[[\s\S]*?\])(?:\n\s*[a-zA-Z_0-9-]+:|\n$)/i);
  if (seriesMatch) {
    try {
      const data = JSON.parse(seriesMatch[2]);
      cfg.series = [{ name: seriesMatch[1], data }];
    } catch (e) {
      // ignore
    }
  }
  return Object.keys(cfg).length ? cfg : null;
}

function parseVisualRaw(yamlText) {
  if (!yamlText) return null;
  // Prefer full YAML parsing
  if (yamlParser) {
    try {
      const parsed = yamlParser.load(yamlText);
      if (parsed && parsed.reveal_hint) return parsed.reveal_hint;
      if (parsed) return parsed;
    } catch (e) {
      // fall back to heuristics below
    }
  }
  // Lightweight heuristic parsing for minimal fields
  const hint = {};
  const layoutMatch = yamlText.match(/layout:\s*"?([\w\-]+)"?/i);
  if (layoutMatch) hint.layout = layoutMatch[1];
  const chartMatch = yamlText.match(/chart_type:\s*"?([\w\-]+)"?/i);
  if (chartMatch) hint.chart_type = chartMatch[1];
  const altMatch = yamlText.match(/alt_text:\s*"([\s\S]*?)"\s*(?:\n|$)/i);
  if (altMatch) hint.alt_text = altMatch[1];
  // try to extract a simple chart_config block if present
  const chartCfg = extractChartConfigFromYaml(yamlText);
  if (chartCfg) hint.chart_config = chartCfg;
  // heuristic: extract mermaid_code block if yaml parser not available
  const lines = yamlText.split('\n');
  for (let i = 0; i < lines.length; i++) {
    const m = lines[i].match(/^(\s*)mermaid_code:\s*\|?/i);
    if (m) {
      const baseIndent = m[1] ? m[1].length : 0;
      const block = [];
      for (let j = i + 1; j < lines.length; j++) {
        const line = lines[j];
        // stop when we see a sibling key (indent <= baseIndent)
        const lead = line.match(/^(\s*)/);
        const leadLen = lead ? lead[1].length : 0;
        if (line.trim() === '') { block.push(''); continue; }
        if (leadLen <= baseIndent) break;
        // remove leading baseIndent+ at least 1 spaces
        block.push(line.replace(new RegExp('^\\s{1,' + (baseIndent+1) + '}'), ''));
      }
      const raw = block.join('\n').trim();
      if (raw) hint.mermaid_code = raw;
      break;
    }
  }
  return Object.keys(hint).length ? hint : null;
}

function renderSlide(s, isFirst) {


  // Determine layout
  let layout = 'bullets';
  if (s.metadata && s.metadata.slide_type === 'title') layout = 'title-slide';
  else if (s.visual && s.visual.layout) layout = s.visual.layout;
  else if (isFirst) layout = 'title-slide';

  // Title slide
  if (layout === 'title-slide') {
    return `<section class="title-slide"><div><h1 class="hero-title">${s.title}</h1>${s.speaker_notes ? `<p class="h-sub">${s.speaker_notes.split('\n')[0]}</p>` : ''}</div></section>`;
  }

  // start section
  let html = `<section class="bullets card"><h3 class="h-title">${s.title}</h3>`;
  if (s.content && s.content.length) {
    html += '<ul class="body-large">';
    for (const c of s.content) html += `<li>${c}</li>`;
    html += '</ul>';
  }

  // helper: table renderer
  function renderTableFromConfig(cfg, alt) {
    if (!cfg || !Array.isArray(cfg.series) || !Array.isArray(cfg.labels)) return '';
    const rows = [];
    if (Array.isArray(cfg.series[0].data) && Array.isArray(cfg.series[0].data[0])) {
      for (const r of cfg.series[0].data) rows.push(r);
    } else {
      // zip datasets
      const n = cfg.labels.length;
      for (let i = 0; i < n; i++) {
        const row = [];
        for (const ds of cfg.series) row.push(Array.isArray(ds.data) ? ds.data[i] : '');
        rows.push(row);
      }
    }
    let t = `<div class="chart-wrapper"><figure role="group" aria-label="${alt || ''}"><table class="chart-data"><thead><tr>`;
    for (const h of cfg.labels) t += `<th>${h}</th>`;
    t += `</tr></thead><tbody>`;
    for (const r of rows) {
      t += '<tr>';
      for (const c of r) t += `<td>${c}</td>`;
      t += '</tr>';
    }
    t += '</tbody></table></figure></div>';
    return t;
  }

  // helper: canvas renderer
  function renderCanvasFromConfig(cfg, type, id, alt) {
    if (!cfg) return '';
    const labels = cfg.labels || [];
    const series = cfg.series || [];
    const spec = JSON.parse(fs.readFileSync(path.resolve(__dirname, '../../../docs/specs/design-spec.reveal.json'), 'utf8'));
    const palette = Object.values(spec.chart_palette || {});
    const datasets = [];
    for (let i = 0; i < series.length; i++) {
      const s = series[i];
      if (!Array.isArray(s.data)) continue;
      const color = palette[i % palette.length] || '#888';
      datasets.push({ label: s.name || `Series ${i+1}`, data: s.data, backgroundColor: color, borderColor: color });
    }
    if (!datasets.length) return renderTableFromConfig(cfg, alt);
    const cfgObj = { type, data: { labels, datasets }, options: { plugins: { legend: { display: true } } } };
    const jsonCfg = encodeURIComponent(JSON.stringify(cfgObj));
    const headerCols = ['Item'].concat(datasets.map(ds => ds.label));
    const headerHtml = `<thead><tr>${headerCols.map(h=>`<th>${h}</th>`).join('')}</tr></thead>`;
    let bodyHtml = '<tbody>';
    for (let i = 0; i < labels.length; i++) {
      bodyHtml += '<tr>';
      bodyHtml += `<td>${labels[i]}</td>`;
      for (const ds of datasets) bodyHtml += `<td>${ds.data[i] !== undefined ? ds.data[i] : ''}</td>`;
      bodyHtml += '</tr>';
    }
    bodyHtml += '</tbody>';

    const tableId = `${id}-data`;
    // show_table can be enabled via chart_config.show_table === true
    const tableVisible = cfg && cfg.show_table;
    const tableWrapperClass = tableVisible ? 'table-visible' : 'visually-hidden';
    const captionId = `${id}-caption`;

    // canvas gets role and aria references to the data table and caption for screen readers
    const canvasAttrs = `id="${id}" class="chart-canvas" data-chart-config="${jsonCfg}" role="img" aria-label="${alt || ''}" aria-describedby="${tableId}" aria-labelledby="${captionId}"`;

    const tableHtml = `<table id="${tableId}" class="chart-data" aria-label="${alt || ''}">${headerHtml}${bodyHtml}</table>`;
    const captionHtml = `<figcaption id="${captionId}" class="visually-hidden">Data table for ${alt || s.title || 'chart'}</figcaption>`;

    return `<div class="chart-wrapper"><figure role="group" aria-labelledby="${captionId}"><div class="canvas-container"><canvas ${canvasAttrs}></canvas></div><div class="${tableWrapperClass}">${tableHtml}</div>${captionHtml}</figure></div>`;
  }

  // charts / diagrams
  if (s.visual && s.visual.chart_type) {
    const ct = s.visual.chart_type;
    const alt = s.visual.alt_text || `${s.title} chart`;
    const id = `chart-${s.slide_number || Math.random().toString(36).slice(2,8)}`;
    const cfg = s.visual.chart_config || (s._visual_parsed && s._visual_parsed.reveal_hint && s._visual_parsed.reveal_hint.chart_config) || null;
    if (ct === 'table') {
      html += renderTableFromConfig(cfg, alt);
    } else if (['bar','line','scatter'].includes(ct)) {
      html += renderCanvasFromConfig(cfg, ct, id, alt);
    } else if (ct === 'diagram' || s.visual.mermaid_code || (s._visual_parsed && (s._visual_parsed.mermaid_code || (s._visual_parsed.reveal_hint && s._visual_parsed.reveal_hint.mermaid_code)))) {
      try { console.log('DIAGRAM keys', s._visual_parsed && s._visual_parsed.reveal_hint ? Object.keys(s._visual_parsed.reveal_hint) : 'none'); } catch(e){}
      let code = s.visual.mermaid_code || (s._visual_parsed && (s._visual_parsed.mermaid_code || (s._visual_parsed.reveal_hint && s._visual_parsed.reveal_hint.mermaid_code)));
      code = code ? String(code).trim() : '';
      if (!code) {
        html += `<div class="chart-wrapper"><p class="small-muted">[diagram data missing]</p></div>`;
      } else {
        try {
          const diagrams = require('./diagrams');
          const svgRel = diagrams.renderMermaidToSvg(code, path.resolve(__dirname, '../../dist'));
          if (svgRel) html += `<div class="chart-wrapper"><img src="${svgRel}" role="img" aria-label="${alt || ''}"/></div>`;
          else html += `<div class="chart-wrapper"><div class="mermaid">${code.replace(/</g,'&lt;')}</div></div>`;
        } catch (e) {
          html += `<div class="chart-wrapper"><div class="mermaid">${code.replace(/</g,'&lt;')}</div></div>`;
        }
      }
    }
  }

  if (s.speaker_notes) html += `<aside class="notes">${s.speaker_notes}</aside>`;
  html += '</section>';
  return html;
}



if (require.main === module) process.exit(0);

function buildSlides() {
  if (!fs.existsSync(slidesJsonPath)) {
    throw new Error('slides.json not found. Run parser first (src/parser.js)');
  }
  const slidesData = JSON.parse(fs.readFileSync(slidesJsonPath, 'utf8'));
  // Build slides HTML
  const slidesHtml = slidesData.slides.map((s, i) => {
    if (s.visual_raw) {
      // parse and preserve parsed object if possible
      if (yamlParser) {
        try {
          s._visual_parsed = yamlParser.load(s.visual_raw);
          s.visual = s._visual_parsed && s._visual_parsed.reveal_hint ? s._visual_parsed.reveal_hint : (s._visual_parsed || null);
        } catch (e) {
          s.visual = parseVisualRaw(s.visual_raw);
        }
      } else {
        s.visual = parseVisualRaw(s.visual_raw);
      }
    }
    return renderSlide(s, i === 0);
  }).join('\n\n');

  // Inject slidesHtml into deck template (replace the each block)
  const eachBlockRegex = /\{\{#each slides\}\}[\s\S]*?\{\{\/each\}\}/;
  if (!eachBlockRegex.test(templateSrc)) {
    throw new Error('Deck template does not contain "{{#each slides}}...{{/each}}" block');
  }
  let rendered = templateSrc.replace(eachBlockRegex, slidesHtml);

  // Replace meta.title if present
  const title = (slidesData.front_matter && slidesData.front_matter.title) ? slidesData.front_matter.title : 'Reveal POC';
  rendered = rendered.replace(/\{\{\s*meta\.title\s*\}\}/g, title);

  // Write final output
  if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });
  fs.writeFileSync(path.join(outDir, 'index.html'), rendered, 'utf8');
  console.log('Rendered dist/index.html with', slidesData.slides.length, 'slides');
  return rendered;
}

module.exports = {
  render: buildSlides,
  parseVisualRaw,
  buildSlides
};

if (require.main === module) {
  try {
    buildSlides();
    process.exit(0);
  } catch (e) {
    console.error(e.message || String(e));
    process.exit(2);
  }
}
