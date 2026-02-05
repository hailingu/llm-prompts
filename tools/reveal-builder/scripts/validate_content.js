const fs = require('fs');
const path = require('path');

function isAssertiveTitle(title) {
  if (!title) return false;
  const verbs = ['is','are','reduce','increase','improve','saves','saves','reduced','reduction','recommends','achieve','enabled','enables','reduced','lowers','raises','improves','cuts','increases'];
  const t = title.toLowerCase();
  if (t.match(/\b\d+%|percent|%\b/)) return true;
  for (const v of verbs) if (t.includes('\b'+v+'\b')) return true;
  // heuristic: contains verb-like words
  return /(is|are|has|have|will|reduce|increases|improve|enable|leads)/i.test(title);
}

function runContentChecks(slidesJsonPath, outPath, options = {}) {
  if (!fs.existsSync(slidesJsonPath)) throw new Error('slides.json not found: ' + slidesJsonPath);
  const slidesData = JSON.parse(fs.readFileSync(slidesJsonPath, 'utf8'));
  const issues = [];
  let total_score = 100;
  let contentScore = 100;

  slidesData.slides.forEach(slide => {
    const numBullets = Array.isArray(slide.content) ? slide.content.length : 0;
    if (numBullets > 6) {
      issues.push({ slide_number: slide.slide_number, rule_id: 'bullet-6x6', severity: 'warning', detail: `Bullets total ${numBullets} (limit 6)`, location: 'content' });
      contentScore -= 10;
    }
    const sn = (slide.speaker_notes || '').trim();
    if (!sn) {
      issues.push({ slide_number: slide.slide_number, rule_id: 'speaker-notes-missing', severity: 'warning', detail: 'No speaker notes present', location: 'speaker_notes' });
      contentScore -= 5;
    }
    const title = slide.title || '';
    if (title && title.split(/\s+/).length <= 3 && !isAssertiveTitle(title)) {
      issues.push({ slide_number: slide.slide_number, rule_id: 'title-assertion', severity: 'suggestion', detail: `Title may be topic-style; consider using an assertion or metric`, location: 'title' });
      contentScore -= 2;
    }
    if ((!slide.content || slide.content.length === 0) && !sn) {
      issues.push({ slide_number: slide.slide_number, rule_id: 'content-empty', severity: 'warning', detail: 'Slide has no content and no speaker notes', location: 'content' });
      contentScore -= 10;
    }
  });

  total_score = Math.max(0, Math.round(contentScore));
  const grade = total_score >= 85 ? 'good' : total_score >= 70 ? 'fair' : 'needs_work';

  const qa_report = {
    total_score,
    grade,
    breakdown: { content: contentScore, visual: null, structure: null, aesthetic: null },
    issues,
    suggestions: [],
    timestamp: new Date().toISOString()
  };

  // simple suggestions: map issues to human-friendly suggestions
  for (const it of issues) {
    if (it.rule_id === 'bullet-6x6') qa_report.suggestions.push(`Slide ${it.slide_number}: reduce bullets or split into multiple slides`);
    if (it.rule_id === 'speaker-notes-missing') qa_report.suggestions.push(`Slide ${it.slide_number}: add speaker notes to capture the talking points`);
    if (it.rule_id === 'title-assertion') qa_report.suggestions.push(`Slide ${it.slide_number}: consider making the title an assertion (e.g., 'Latency reduced by 30%')`);
    if (it.rule_id === 'content-empty') qa_report.suggestions.push(`Slide ${it.slide_number}: add summary or speaker notes`);
  }

  if (!fs.existsSync(path.dirname(outPath))) fs.mkdirSync(path.dirname(outPath), { recursive: true });
  fs.writeFileSync(outPath, JSON.stringify({ qa_report, source: slidesJsonPath }, null, 2), 'utf8');
  return qa_report;
}

if (require.main === module) {
  const slidesJsonPath = process.argv[2] || path.resolve(__dirname, '../dist/slides.json');
  const outPath = process.argv[3] || path.resolve(__dirname, '../dist/qa_report.json');
  try {
    const r = runContentChecks(slidesJsonPath, outPath);
    console.log('Wrote QA report to', outPath);
    process.exit(0);
  } catch (e) {
    console.error('Content QA failed:', e && e.message || e);
    process.exit(1);
  }
}

module.exports = { runContentChecks };