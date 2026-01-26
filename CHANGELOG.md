# Changelog

## Unreleased

- docs(go-standards): 修复 MD036/MD001（将斜体用作标题的行替换为真实标题并修正标题等级），并为多处条目补充简短说明以提高可读性（`go-standards/effective-go-guidelines.md`）

- docs(readme): 移除 README 标题中的 emoji 以修复 TOC 锚点（MD051）
- docs(readme): 样式优化：在 README 中加入快速链接（CHANGELOG/Contributing/Security/Code of Conduct）以改善导航

- docs(standards): 新增 `README.md`、`CONTRIBUTING.md`、`CODE_OF_CONDUCT.md` 與 `SECURITY.md`，改进入门与社区指南

- docs(standards): 将 `java-standards/checkstyle-alibaba.xml` 中的注释翻译为英文以便国际化与协作

- docs(standards): 用 `subdomain` 替换不常见术语 `subbusiness` 并添加示例，改进 GAV 约定说明

- Prompt: require generated plan documents from `task/plan-breakdown-to-task.prompt.md` to conform to markdownlint v0.40.0 documentation rules ([markdownlint v0.40.0 文档](https://github.com/DavidAnson/markdownlint/tree/v0.40.0/doc))
- 修复 MD034（裸 URL）：将 `java-standards/alibaba-java-guidelines.md` 中的裸链接替换为 `[阿里巴巴 p3c 仓库](https://github.com/alibaba/p3c)`，并微调文档换行以符合规范
- 修复 MD034（裸 URL）：将 `CHANGELOG.md` 中的 markdownlint 链接改为 Markdown 链接格式
- 修复 MD056（表格列数）：在 `standards/api-patterns.md` 中将 `**2xx Success**` 行补足为 5 列，避免表格缺失数据
- 修复 MD060（表格样式）与 MD034（裸 URL）：调整 `standards/agent-collaboration-protocol.md` 表格样式并替换裸链接；微调 `java-standards/alibaba-java-guidelines.md` 的代码块标记
- 删除过期文档：移除 `standards/agent-improvements-summary.md`（该总结已合并至其他规范文档）
- 修复 `standards/design-review-checklist.md` 排版与示例表格（修复 MD056/MD060）
- 将 `standards/agent-collaboration-protocol.md` 中的协作流程 ASCII 图替换为 Mermaid 图，改进渲染与可维护性，并为不同角色添加节点颜色以便更好区分
- 统一 Mermaid 节点与类名为驼峰命名（camelCase），提高一致性与可维护性
- 格式: 美化 `java-standards/static-analysis-setup.md`（增加空行、规范代码块标记并完善验证步骤）
- 翻译: 将 `java-standards/alibaba-java-guidelines.md` 翻译为英文以便国际化引用
- 修复 MD034（裸 URL）：将 `java-standards/alibaba-java-guidelines.md` 中的裸 GitHub 链接替换为 Markdown 链接格式
- 在 `agents/java-tech-lead.agent.md` 的 Mermaid 图中为不同角色添加颜色并统一类名为 camelCase，改善可视化与可维护性
- 新增 `agents/`, `java-standards/`, `standards/`, `templates/` 文档与模板，补充项目的代理说明、Java 规范、设计文档标准与模板

### 9ddd9ce

- 新增 `git/git-pull-request.prompt.md` 与 `.github/prompts/git-pull-request.prompt.md`，用于生成 PR Title 与 PR Description（中文、Markdown 格式）

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
