# Changelog

## Unreleased

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

