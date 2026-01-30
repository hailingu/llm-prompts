# Changelog

## [Unreleased]

> 建议版本更新：**MINOR**（包含新特性 `feat(...)`）

### Added

- feat(agent): 新增 `agents/cortana.agent.md`（Cortana 通用 agent 模板，见 feature/universal-agent 分支）
- feat(project): 完善短视频替换系统的算法、数据与评估规范；新增 Research Design、算法规格、数据 schema、Great Expectations 示例、评估计划与 MOS 模板，便于 CI 自动化与合规审查。
- feat(docs): 新增量化交易指南 `data-science-standards/quantitative-trading-guide.md`，包含策略、回测、风险管理与生产部署；并将 cheat-sheet 中的量化交易精简为速查并链接至该指南。

### Changed

- docs(prompts): 为 `plan-breakdown-to-task` 与 `task-execute` 提示新增 Execution Parameters（机器可读参数）、参数校验、执行流程與回滚计划，使任务拆分与自动化执行互操作；同时更新示例与错误返回格式。
- docs(prompts): 提交信息修正 — 补充文件名引用与变更说明（见后续提交）
- docs(readme): 更新 `README.md` 为 2026 标准并新增 `changelog-specialist` agent定义。
- docs(templates): 移除一组已弃用的 PPT 模板（`bcg-matrix`, `gantt-chart`, `porter-five-forces`, `swot-analysis`, `waterfall-chart`, `basic_template`），以精简模板库并减少维护负担。

---

> 说明：为了提高可维护性，已将旧有的 commit-id 小节整理并从主要变更列表中移除。
完整的历史记录请参阅 Git 提交历史（`git log`）或 release 页面。

---

[Unreleased]: https://github.com/hailingu/llm-prompts/compare/v0.1.0-beta.1...HEAD
[v0.1.0-beta.1]: https://github.com/hailingu/llm-prompts/releases/tag/v0.1.0-beta.1
