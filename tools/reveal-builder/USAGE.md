Reveal Builder — Expanded Usage & CI Notes

Docs & Examples

- Usage guide: `docs/reveal-builder.md` (quick commands and examples)
- Design spec guidance: `docs/design-spec-reveal.md`
- Examples: `docs/example-chart-raw.md`, `docs/example-mermaid.md`

Validation & QA (recommended in CI)

- `npm run validate-a11y -- /full/path/to/tools/dist/index.html` — accessibility checks (axe)
- `npm run check-contrast` — contrast + color-blind checks for design-spec
- `npm run validate-content` — content QA (bullets, speaker notes, titles)
- `npm test` — run unit/POC tests

CI guidance

- Recommended CI flow on PR: parse & build → run `check-contrast` → run `validate-content` → run `validate-a11y` (if runner provides Chrome) → attach `qa_report.json` and `a11y_report.json` as artifacts.

Notes

- Use `PUPPETEER_SKIP_DOWNLOAD=1 npm ci` in CI to avoid Chromium downloads and provide a Chrome binary via `CHROME_PATH` or install Chrome on runner images.
