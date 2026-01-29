# Changelog

## [Unreleased]

> 建议版本更新：**MINOR**（包含新特性 `feat(...)`）

### Added

- feat(project): 完善短视频替换系统的算法、数据与评估规范；新增 Research Design、算法规格、数据 schema、Great Expectations 示例、评估计划与 MOS 模板，便于 CI 自动化与合规审查。
- feat(docs): 新增量化交易指南 `data-science-standards/quantitative-trading-guide.md`，包含策略、回测、风险管理与生产部署；并将 cheat-sheet 中的量化交易精简为速查并链接至该指南。

### Changed

- docs(prompts): 为 `plan-breakdown-to-task` 与 `task-execute` 提示新增 Execution Parameters（机器可读参数）、参数校验、执行流程與回滚计划，使任务拆分与自动化执行互操作；同时更新示例与错误返回格式。
- docs(prompts): 提交信息修正 — 补充文件名引用与变更说明（见后续提交）
- docs(readme): 更新 `README.md` 为 2026 标准并新增 `changelog-specialist` agent定义。



### 3cd194d

- docs(changelog): 恢复并规范化 `c7f16bd` 段落格式，确保提交 id 小节与条目格式一致。

### e24901f

- docs(agents): 整理 `go-code-reviewer` 与 `java-code-reviewer` 文档，修复 Phase 6/7 报告与迭代模板，去除重复内容并统一结构与风格；同步更新 `agents/go-api-designer.agent.md`、`agents/go-coder-specialist.agent.md`、`agents/go-tech-lead.agent.md`。
- docs(go-standards): 新增 Go 标准文档：`go-standards/static-analysis-setup.md`、`go-standards/api-patterns.md`、`go-standards/agent-collaboration-protocol.md`。

### 6022f20

- docs(go-standards): 修复 MD036/MD001（将斜体用作标题的行替换为真实标题并修正标题等级），并为 `go-standards/effective-go-guidelines.md` 补充简短说明以提高可读性。

### 8cecdec

- docs(go-standards): 微调并补充文档说明，改善示例和格式（MD036 修复系列之一）。

### 200a373

- fix(docs): 将多处示例的强调替换为真实标题以修复 MD036（if-statement 指南）。

### 33ed4cc

- fix(docs): 将包注释中的强调替换为标题（MD036 修复）。

### a3b69a9

- fix(docs): 将缩进与示例中的强调替换为标题（MD036 修复）。

### 2d35d81

- fix(docs): 修复行长示例中使用强调作为标题的问题（MD036 修复）。

### 9ddd9ce

- 新增 `git/git-pull-request.prompt.md` 與 `.github/prompts/git-pull-request.prompt.md`，用于生成 PR Title 與 PR Description（中文、Markdown 格式）

### c7f16bd

- 文档: 让提交提示在生成提交信息后执行提交
- 文档: 按 commit id 重组 CHANGELOG 格式并整理条目
- 将 git 提交提示翻译为英文
- 添加 `.gitmessage` 提交模板（含英文注释与示例，列出提交类型及其中文含义）
- 增加用于生成提交信息的 prompts：`.github/prompts/git-gitmessage-initiate.prompt.md` 与 `git/git-gitmessage-initiate.prompt.md`
- 明确 git 提示行为：要求在适用时更新 README/CHANGELOG，并在请求时提交更改
- 更新 `evaluation/project-evaluate.prompt.md`：重写为英文提示（要求输出为简体中文表格），修复格式损坏并补全内容
- 更新 `README.md`，说明如何启用该仓库的提交模板
- 新增 `analysis/architecture-analysis.prompt.md`：提供全面的架构文档编写指导

---

[Unreleased]: https://github.com/hailingu/llm-prompts/compare/v0.1.0-beta.1...HEAD
[v0.1.0-beta.1]: https://github.com/hailingu/llm-prompts/releases/tag/v0.1.0-beta.1
