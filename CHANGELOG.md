# Changelog

## [Unreleased]

> 建议版本更新：**MINOR**（包含新特性 `feat(...)`）

### Added

- feat(skills): add news-search skill for web search capability
- feat(skills): expand domain-keyword-detection from 6 to 14 domains (energy, data_science, automotive, cloud_infrastructure, telecom, iot, medical_devices, security)
- feat(skills): add memory-manager skill for persistent context across sessions
- feat(skills): add stock-price-tracker skill for real-time stock data via Yahoo Finance
- feat(skills): add Mistakes section to global memory for error tracking

### Changed

- refactor(skills): unify skill docs to SKILL.md (domain-keyword-detection, markdown-formatter, md-table-fixer)
- refactor: move global memory to memory/global.md for consistent organization
- refactor: redesign global memory as persistent context (User Profile, Known Facts, Preferences, Constraints, Project Context, Knowledge Base, Mistakes, Recent Context)
- refactor: convert memory-manager SKILL to English only
- docs: update README with agent system (33+ agents) and skills overview (10 skills)
- docs: update cortana.agent.md to integrate memory-manager skill

### Fixed

- fix(ppt): generate AI development report 2024-2028 McKinsey-style PPT (8 pages)

---

## [v0.3.0] - 2026-02-21

> 发布说明：PPT 生成器增强与流水线优化

### Added

- feat(EA/P1): improve comparison_split layout (side-by-side) + exclude title/section from metrics
- feat(ppt-generator): EA fixes + visual improvements
- feat(ppt): merge patch_v6 into generate_pptx.py
- feat(agent): 新增 `agents/cortana.agent.md`（Cortana 通用 agent 模板，见 feature/universal-agent 分支）
- feat(project): 完善短视频替换系统的算法、数据与评估规范；新增 Research Design、算法规格、数据 schema、Great Expectations 示例、评估计划与 MOS 模板，便于 CI 自动化与合规审查。
- feat(docs): 新增量化交易指南 `data-science-standards/quantitative-trading-guide.md`，包含策略、回测、风险管理与生产部署；并将 cheat-sheet 中的量化交易精简为速查并链接至该指南。

### Fixed

- fix(ppt): empty-chart placeholder (insight strip) + center small bullet lists when visual present; regen v8

### Changed

- chore(ppt): add floating/process/comparison layouts; update slide-theme
  and apply to v2 slides; update ppt-specialist agent
- chore(dag): mark optimization start
- docs(tasks): add PPT pipeline optimization task breakdown — 6 phases, 19 subtasks with execution parameters
- docs(design): add PPT pipeline long-term optimization plan — incremental enhancement, dual-track coexistence strategy
- chore: track generate_pptx.py source (recovered from bytecode cache)
- chore(ppt): v5 layout alternation — data-heavy mirrored split,
  comparison horizontal bars, section divider auto-subtitle, misc fixes
- cleanup: remove outdated requirements file
- cleanup: remove obsolete tests, scripts, tools and standardize agent file
- docs(prompts): 为 `plan-breakdown-to-task` 与 `task-execute` 提示新增 Execution Parameters（机器可读参数）、参数校验、执行流程與回滚计划，使任务拆分与自动化执行互操作；同时更新示例与错误返回格式。
- docs(prompts): 提交信息修正 — 补充文件名引用与变更说明（见后续提交）
- docs(readme): 更新 `README.md` 为 2026 标准并新增 `changelog-specialist` agent定义。
- docs(templates): 移除一组已弃用的 PPT 模板（`bcg-matrix`, `gantt-chart`,
  `porter-five-forces`, `swot-analysis`, `waterfall-chart`, `basic_template`），
  以精简模板库并减少维护负担。
- docs(agent): 修复 `agents/cortana.agent.md` 与 `.github/agents/cortana.agent.md` 中的 Mermaid 流程解析错误，通过预声明节点、简化边标签与转义特殊符号来消除解析异常，并改进可读性与稳定性。

---

> 说明：为了提高可维护性，已将旧有的 commit-id 小节整理并从主要变更列表中移除。
完整的历史记录请参阅 Git 提交历史（`git log`）或 release 页面。

---

[Unreleased]: https://github.com/hailingu/llm-prompts/compare/v0.3.0...HEAD

[v0.3.0]: https://github.com/hailingu/llm-prompts/compare/v30-enrich-20260210...v0.3.0
