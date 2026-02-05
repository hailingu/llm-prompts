const fs = require('fs');
const path = require('path');

function parseFrontMatter(raw) {
  const fmMatch = raw.match(/^---\s*\n([\s\S]*?)\n---\s*\n/);
  if (!fmMatch) return { front: {}, body: raw };
  const fmRaw = fmMatch[1];
  const front = {};
  for (const line of fmRaw.split(/\n+/)) {
    const m = line.match(/^([^:]+):\s*(.+)$/);
    if (m) {
      const key = m[1].trim();
      let val = m[2].trim();
      if (/^\d{4}-\d{2}-\d{2}/.test(val)) { /* leave as string */ }
      front[key] = val;
    }
  }
  const body = raw.slice(fmMatch[0].length);
  return { front, body };
}

function extractYamlBlock(text) {
  const m = text.match(/```yaml\n([\s\S]*?)\n```/i);
  return m ? m[1].trim() : null;
}

function parseSlidesMd(filePath) {
  const raw = fs.readFileSync(filePath, 'utf8');
  const { front, body } = parseFrontMatter(raw);

  // Split into slide sections by '## Slide' headings
  const headerRegex = /##\s+Slide\s+(\d+):\s*(.*)/g;
  const headers = [...body.matchAll(headerRegex)];
  // Use a lookahead split so headings at start-of-body are captured too
  const rawParts = body.split(/(?=^##\s+Slide\s+\d+:\s*)/m);
  const parts = rawParts.filter(p => p.trim().startsWith('## Slide'));

  const slides = parts.map((part, idx) => {
    const header = headers[idx];
    const slideNumber = header ? parseInt(header[1], 10) : idx + 1;
    const slideTopic = header ? header[2].trim() : '';

    const titleMatch = part.match(/\*\*Title\*\*:\s*(.+)/);
    const title = titleMatch ? titleMatch[1].trim() : '';

    const contentMatch = part.match(/\*\*Content\*\*:\s*\n([\s\S]*?)(?=\n\*\*SPEAKER_NOTES|\n\*\*VISUAL|\n\*\*METADATA|\n##|$)/i);
    let content = [];
    if (contentMatch) {
      const lines = contentMatch[1].split(/\n/).map(l => l.trim()).filter(Boolean);
      for (const line of lines) {
        const m = line.match(/^-\s*(.+)/);
        if (m) content.push(m[1].trim());
      }
    }

    const notesMatch = part.match(/\*\*SPEAKER_NOTES\*\*:\s*\n([\s\S]*?)(?=\n\*\*VISUAL|\n\*\*METADATA|\n##|$)/i);
    const speaker_notes = notesMatch ? notesMatch[1].trim() : '';

    const visualYaml = extractYamlBlock(part);

    const metadataMatch = part.match(/\*\*METADATA\*\*:\s*\n```json\n([\s\S]*?)\n```/i);
    let metadata = null;
    if (metadataMatch) {
      try { metadata = JSON.parse(metadataMatch[1]); } catch (e) { metadata = { _parse_error: e.message }; }
    }

    return {
      slide_number: slideNumber,
      slide_topic: slideTopic,
      title,
      content,
      speaker_notes,
      visual_raw: visualYaml,
      metadata
    };
  });

  return { front_matter: front, slides };
}

if (require.main === module) {
  const file = process.argv[2] || path.resolve(__dirname, '../../../docs/MFT_slides.md');
  const outDir = path.resolve(__dirname, '../../dist');
  if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });
  const parsed = parseSlidesMd(file);
  fs.writeFileSync(path.join(outDir, 'slides.json'), JSON.stringify(parsed, null, 2));
  console.log(`Parsed ${parsed.slides.length} slides from ${file} and wrote to ${path.join(outDir,'slides.json')}`);
}

module.exports = { parseSlidesMd };

