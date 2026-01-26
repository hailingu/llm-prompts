# Changelog

## Unreleased

- Prompt: require generated plan documents from `task/plan-breakdown-to-task.prompt.md` to conform to markdownlint v0.40.0 documentation rules ([markdownlint v0.40.0 文档](https://github.com/DavidAnson/markdownlint/tree/v0.40.0/doc))
- 修复 MD034（裸 URL）：将 `java-standards/alibaba-java-guidelines.md` 中的裸链接替换为 `[阿里巴巴 p3c 仓库](https://github.com/alibaba/p3c)`，并微调文档换行以符合规范
- 修复 MD034（裸 URL）：将 `CHANGELOG.md` 中的 markdownlint 链接改为 Markdown 链接格式
- 修复 MD056（表格列数）：在 `standards/api-patterns.md` 中将 `**2xx Success**` 行补足为 5 列，避免表格缺失数据
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
