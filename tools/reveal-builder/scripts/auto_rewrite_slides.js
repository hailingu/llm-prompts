const fs = require('fs');
const path = require('path');
const parser = require('../src/parser');

function makeAssertive(title, content) {
  if (!title) return content && content[0] ? content[0].slice(0,80) : '关键结论';
  // if title already contains punctuation or a verb/metric, keep
  if (/[：:，,.\d%]/.test(title)) return title;
  // else, try to use first bullet as assertion
  if (content && content.length) return `${content[0]}`;
  return title;
}

function splitBullets(slide, max=5) {
  const bullets = slide.content || [];
  if (!bullets.length) return [slide];
  if (bullets.length <= max) return [slide];
  const parts = [];
  for (let i = 0; i < bullets.length; i += max) {
    const chunk = bullets.slice(i, i+max);
    const s = JSON.parse(JSON.stringify(slide));
    s.content = chunk;
    if (i > 0) s.title = (slide.title || '详细说明') + `（续 ${Math.floor(i/max)+1}）`;
    parts.push(s);
  }
  return parts;
}

function mapVisual(visualRaw) {
  if (!visualRaw) return null;
  // try parse YAML-ish using naive heuristics
  const t = visualRaw;
  const typeMatch = t.match(/type:\s*"?([\w\-]+)"?/i);
  const titleMatch = t.match(/title:\s*"?([^"]+)"?/i);
  const obj = { reveal_hint: {} };
  if (typeMatch) {
    const ty = typeMatch[1].toLowerCase();
    if (ty === 'matrix') obj.reveal_hint.layout = 'full-bleed';
    else if (ty === 'comparison') obj.reveal_hint.layout = 'two-column-6040';
    else if (ty === 'sequence') obj.reveal_hint.layout = 'chart-focused';
    else obj.reveal_hint.layout = 'chart-focused';
    obj.reveal_hint.chart_type = ty === 'matrix' ? 'table' : (ty === 'comparison' ? 'chart' : 'diagram');
  }
  if (titleMatch) obj.reveal_hint.alt_text = titleMatch[1];
  return obj.reveal_hint;
}

function renderSlideAsMd(s) {
  let md = `## Slide ${s.slide_number}: ${s.slide_topic || s.title || ''}\n`;
  md += `**Title**: ${s.title || ''}\n\n`;
  md += `**Content**:\n`;
  if (s.content && s.content.length) {
    for (const b of s.content) md += `- ${b}\n`;
  } else {
    md += `- _（请补充要点）_\n`;
  }
  md += `\n`;
  md += `**SPEAKER_NOTES**:\n`;
  md += `${s.speaker_notes && s.speaker_notes.trim() ? s.speaker_notes.trim() : '_请补充 speaker notes，包含 1-2 句 summary，1-2 句 rationale 与 action_'}\n\n`;
  if (s.visual_raw) {
    const vh = mapVisual(s.visual_raw);
    if (vh) {
      md += `**VISUAL**:\n`;
      md += '```yaml\n';
      md += 'reveal_hint:\n';
      for (const k of Object.keys(vh)) {
        md += `  ${k}: "${vh[k]}"\n`;
      }
      md += '```\n\n';
    }
  }
  return md;
}

if (require.main === module) {
  const inPath = path.resolve(__dirname, '../../../docs/MFT_slides.md');
  const outPath = path.resolve(__dirname, '../../../docs/MFT_slides.revealed.md');
  const parsed = parser.parseSlidesMd(inPath);
  const newSlides = [];
  for (const sl of parsed.slides) {
    // make title assertive if not
    const newTitle = makeAssertive(sl.title, sl.content);
    sl.title = newTitle;
    // ensure speaker notes
    if (!sl.speaker_notes || !sl.speaker_notes.trim()) {
      sl.speaker_notes = `Summary: ${newTitle} — 请补充要点与行动项。`;
    }
    // split long bullets
    const pieces = splitBullets(sl, 5);
    // renumber slide_number sequentially (we'll assign after)
    for (const p of pieces) newSlides.push(p);
  }
  // assign slide numbers sequentially and render md
  let i = 1;
  const outParts = [];
  for (const ns of newSlides) {
    ns.slide_number = i++;
    outParts.push(renderSlideAsMd(ns));
  }
  const header = `---\ntitle: "${parsed.front_matter && parsed.front_matter.title || 'Reveal Deck'}"\n---\n\n`;
  fs.writeFileSync(outPath, header + outParts.join('\n\n'), 'utf8');
  console.log('Wrote', outPath, 'with', newSlides.length, 'slides');
}

module.exports = { makeAssertive, splitBullets, mapVisual };