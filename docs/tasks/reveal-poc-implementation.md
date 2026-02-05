# Reveal.js POC — 实施任务表

说明：本任务表用于从零开始实现 Reveal.js 演示生成流水线（POC），替换旧的 PPT 工作流。任务以阶段（Phase）组织，每项任务包含目标、产出、负责人（role）、估算工时/天、前置依赖与验收标准。

---

## 里程碑（高层）
- Phase 1 — 设计与规范（Schema & Tokens）
- Phase 2 — 构建 Reveal Builder（转换器 & 模板）
- Phase 3 — 图表与图示集成（Chart.js / mermaid）
- Phase 4 — 可访问性与 QA（axe + contrast）
- Phase 5 — CI/发布与文档
- Phase 6 — 交付与演示（artifact packaging + previews）

---

## Phase 1 — 设计与规范

T1.1 设计 `design-spec.reveal.json` schema
- 目标：定义 Reveal 专用的 design-spec（tokens, layouts, animation, accessibility）
- 产出：`design-spec.reveal.json` 示例文件 + schema 文档（`docs/specs/reveal-schema.md`）
- 负责人：visual-designer
- 估算：0.5 天
- 依赖：无
- 验收标准：schema 包含 color tokens、typography、spacing、layouts、chart_palette、animation_tokens、accessibility.minContrast；示例文件能驱动主题 CSS 变量

T1.2 确定 slides.md VISUAL 扩展字段（用于 Reveal hints）
- 目标：扩展 `slides.md` 的 VISUAL 块，加入 `reveal_hint` (layout, chart_type, chart_config, mermaid_code, alt_text)
- 产出：`docs/specs/slides-viz-hint.md` 与示例 `docs/example-slides.md`
- 负责人：content-planner
- 估算：0.25 天
- 依赖：T1.1
- 验收标准：示例 slides.md 能映射到构建器模板参数

T1.3 设计 QA Schema（`qa_report.json` / `a11y_report.json` 字段定义）
- 目标：定义 POC 需要输出的 QA 报表结构（内容、设计、可访问性、性能）
- 产出：`docs/specs/qa-schema.md`
- 负责人：reveal-specialist + a11y-validator
- 估算：0.25 天
- 验收标准：QA 报表包含整体分数、每页问题列表、关键违规 count 与修复建议

---

## Phase 2 — 构建 Reveal Builder（转换器 & 模板）

T2.1 初始化项目 & 工程脚手架
- 目标：创建 `tools/reveal-builder/`、`package.json`、`start.sh`、`.gitignore`
- 产出：项目骨架、依赖声明（Reveal.js, markdown-it, handlebars/ ejs, chart.js, mermaid-cli, puppeteer）
- 负责人：reveal-specialist
- 估算：0.5 天
- 验收标准：`npm install` 无错误，`start.sh` 能在本地提供静态页面

T2.2 实现 Markdown→Slide parser（包含 front-matter）
- 目标：解析 `slides.md`，提取 slides、speaker notes、VISUAL blocks、metadata
- 产出：`tools/reveal-builder/src/parser.js`（或 .ts）和单元测试
- 负责人：reveal-specialist
- 估算：1 天
- 依赖：T1.2
- 验收标准：parser 能解析 `docs/MFT_slides.md` 并返回 JSON 结构，单元测试覆盖主要路径

T2.3 设计并实现 HTML 模板与 CSS 变量（theme）
- 目标：创建 `templates/reveal/base.html`、`templates/reveal/theme.css`（使用 `design-spec.reveal.json` 变量）
- 产出：主题 CSS、Layouts（title-slide, bullets, two-column-6040, chart-focused, full-bleed）示例
- 负责人：visual-designer + reveal-specialist
- 估算：1 天
- 依赖：T1.1
- 验收标准：模板能渲染所有布局，theme.css 使用 CSS 变量并有注释映射到 design-spec

T2.4 实现 Slide Renderer（template engine）
- 目标：将 parser 输出与模板结合，生成 `index.html` 或 per-deck HTML
- 产出：`tools/reveal-builder/src/render.js`，demo output（`docs/presentations/<session>/index.html`）
- 负责人：reveal-specialist
- 估算：1 天
- 依赖：T2.2, T2.3
- 验收标准：运行 `node build.js --input docs/MFT_slides.md` 能生成可打开的 HTML，speaker notes 在 Reveal notes 插件中可见

T2.5 CLI & build script
- 目标：实现 CLI（build、start、clean、export-pdf）
- 产出：`tools/reveal-builder/cli.js`、`start.sh`、`build.sh`
- 负责人：reveal-specialist
- 估算：0.5 天
- 验收标准：`./start.sh` 启动本地预览，`node build.js --export-pdf` 生成 PDF（Puppeteer）

---

## Phase 3 — 图表与图示集成

T3.1 Chart.js 集成与 chart_config mapping
- 目标：支持 `chart_type` (bar/line/scatter) 与 chart_config（labels, series, colors）自动渲染
- 产出：`templates/reveal/components/chart.js` + renderer hooking
- 负责人：reveal-specialist
- 估算：0.5 天
- 验收标准：示例 chart 按 design-spec 调色板渲染，包含 ARIA 属性和 data table fallback

T3.2 mermaid CLI 集成与 SVG embedding
- 目标：把 mermaid code → SVG（或 PNG）并嵌入 slide，保留 alt_text
- 产出：`tools/reveal-builder/src/diagrams.js`，cache 机制，示例 SVG
- 负责人：reveal-specialist
- 估算：0.5 天
- 验收标准：示例 mermaid code 能生成 SVG，SVG 被嵌入 HTML，并带 `aria-label` 与 `role="img"`

T3.3 图表可访问性：数据表和直接标注
- 目标：在需要时自动生成数据表 fallback、直接标签而非仅 legend
- 产出：helper functions、示例 slides
- 负责人：reveal-specialist
- 估算：0.5 天
- 验收标准：chart rendering 时存在数据表（visible or hidden) and aria-describedby linking

---

## Phase 4 — 可访问性与 QA

T4.1 集成 axe-core（Puppeteer）做自动 a11y 测试
- 目标：实现 `scripts/validate_reveal.js`，运行 axe on generated HTML
- 负责人：a11y-validator
- 估算：0.5 天
- 验收标准：运行 `node scripts/validate_reveal.js docs/presentations/<session>/index.html` 输出 `a11y_report.json`，无 critical violations

T4.2 Contrast checks 与 color-blind 模拟
- 目标：对 `design-spec.reveal.json` 调色板跑对比度检查和色盲模拟
- 负责人：visual-designer + a11y-validator
- 估算：0.5 天
- 验收标准：报告列出 contrast ratios，若低于阈值需在 spec 中建议替换颜色

T4.3 Content QA & speaker notes verification
- 目标：验证 slides.md 的 content QA（前述 content_qa.json）并把问题集成到 `qa_report.json`
- 负责人：content-planner + reveal-specialist
- 估算：0.25 天
- 验收标准：所有关键问题（Key Decisions, bullets limit, speaker notes coverage）通过或列出 fix_suggestions

T4.4 Performance checks（大小/图片/DPI）
- 目标：验证 asset 总大小和图片 DPI；实现压缩策略（pngquant / mozjpeg）
- 负责人：reveal-specialist
- 估算：0.25 天
- 验收标准：总 assets < 10MB（POC target），每图 <5MB，关键 diagrams ≥ 200–300 DPI

---

## Phase 5 — CI/发布与文档

T5.1 GitHub Actions workflow（build + validate + artifacts）
- 目标：实现 `.github/workflows/reveal-build.yml`，自动运行 build + a11y + generate previews on push/PR
- 负责人：reveal-specialist
- 估算：0.5 天
- 验收标准：PR 时自动生成 artifacts 到 `docs/presentations/<session>/` 并 attach `qa_report.json`

T5.2 文档与示例（docs/）
- 目标：创建使用说明（`docs/reveal-builder.md`）、主题说明（`docs/design-spec-reveal.md`）、运行指南（`docs/tasks/*` 已在此）
- 负责人：reveal-specialist + visual-designer + content-planner
- 估算：0.5 天
- 验收标准：README 包含快速开始、示例命令和 QA 说明

---

## Phase 6 — 交付与演示

T6.1 生成演示包并制作预览
- 目标：生成 `docs/presentations/<session>/` 包含 `index.html`, `previews/*.png`, `qa_report.json`, `a11y_report.json`
- 负责人：reveal-specialist
- 估算：0.25 天
- 验收标准：手动打开 `index.html` 可正常播放，预览 PNG 清晰

T6.2 内部演示与收集反馈
- 目标：安排内部 review session（团队 + 创意总监），记录反馈并创建 issues
- 负责人：creative-director
- 估算：0.5 天
- 验收标准：收集反馈并创建 backlog，若发现阻塞性问题必须优先修复

---

## 可选：PPTX 后端（留待未来）
- 说明：如果仍需 PPTX 输出，可单独实现 `ppt-specialist` pipeline（使用 python-pptx + design-spec mapping），但非 POC 必需

---

## 风险识别与缓解
- R1: 自动渲染复杂图（高） → 缓解：优先实现结构化图（Chart.js, mermaid），并要求 visual-designer 提供复杂 SVG
- R2: 可访问性未达标（中） → 缓解：将 axe-core 作为必过的 QA gate；设计时自动保存对比度较低的 token 替代方案
- R3: 导出 PDF 在不同系统渲染差异（中） → 缓解：使用 Puppeteer 固定 Chromium 版本做 Baseline screenshot

---

## 时间估计（粗略总计）
- Phase 1: 1 天
- Phase 2: 4 天
- Phase 3: 1.5 天
- Phase 4: 1.5 天
- Phase 5: 1 天
- Phase 6: 0.75 天

**总计 POC 实现：≈ 9.75 天 (≈ 2 周)**

---

## 验收条件（POC 完成）
- 能用一条命令构建并生成 `docs/presentations/mft-reveal/` 包含 `index.html` 和 `previews/*.png`
- `qa_report.json` 与 `a11y_report.json` 未包含 Critical Violations
- `slides.md` 中的 speaker notes 在 Reveal 演示的 notes 面板可见且完整
- Theme 使用 `design-spec.reveal.json` 的 token（颜色/字体/spacing）被正确注入

---

## 交付件清单
- `design-spec.reveal.json`（示例）
- `tools/reveal-builder/`（代码 & templates）
- `docs/presentations/<session>/`（生成的演示包）
- `docs/reveal-builder.md`（使用说明）
- `docs/specs/`（schema、QA schema）
- `a11y_report.json`, `qa_report.json`, `previews/*.png`

---

如需，我可以现在开始实现 **Phase 1–3（POC 的核心）** 并在 1–2 小时内提交首个可运行 demo（`index.html` + `qa_report.json`）。请确认是否现在开始实施。