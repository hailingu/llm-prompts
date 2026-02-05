# QA Report Schema (Reveal POC)

目的：定义 `qa_report.json` 的字段与评分规则，用于 POC 的自动化质量检查（内容、设计、可访问性、性能、技术验证）。

---

## 顶层结构（概览）

```json
{
  "overall_score": 0,
  "timestamp": "2026-02-05T00:00:00Z",
  "summary": "PASS|FAIL|WARN",
  "checks": { /* 按类别细分 */ },
  "issues": [ /* 每个 issue 的详细信息 */ ],
  "warnings": [],
  "fix_suggestions": []
}
```

---

## 字段说明

- overall_score (number): 0-100 的加权总分。
- timestamp (string, ISO 8601): 报告生成时间。
- summary (string): PASS/FAIL/WARN，基于总分和是否含 critical violations。

- checks (object): 包含若干子检查，每个检查包含 `status`、`score`、`details`:
  - content: {status, score, details}
  - design: {status, score, details}
  - accessibility: {status, score, details}
  - performance: {status, score, details}
  - technical: {status, score, details}

每个 `details` 可以包含子字段/计数（critical_issues / major / minor）及示例说明。

- issues (array): 每项是一个 issue 对象，字段：
  - id (string) e.g., "ISSUE-001"
  - slide (number|null) 如果与单页相关
  - severity ("critical"|"major"|"minor"|"info")
  - category ("content"|"design"|"accessibility"|"performance"|"technical")
  - title (string)
  - description (string)
  - location (string) e.g., "slide 5 > chart"
  - suggested_fix (string)

- warnings (array of string): 非阻塞问题
- fix_suggestions (array): 建议的修复动作与优先级

---

## 评分权重与门限

- 权重（默认）：
  - content: 25%
  - design: 25%
  - accessibility: 25%
  - performance: 15%
  - technical: 10%

- 验收阈值：
  - overall_score >= 70 AND critical_issues == 0 → PASS
  - critical_issues > 0 → FAIL (阻塞)
  - overall_score < 70 → WARN/FAIL 取决于 critical count

---

## 示例问题分类

- content: 缺少 Key Decisions、bullet 超限、speaker notes 缺失
- design: color token mismatch、font size 不满足 spec
- accessibility: 无 alt text、contrast < threshold、axe critical errors
- performance: images too large、PPTX/asset 超出预算（POC: assets < 10MB）
- technical: layout overflow、missing fonts（字库缺失）

---

## 输出文件

- `qa_report.json`（与 schema 对齐）
- `a11y_report.json`（由 axe 输出，嵌入或作为独立文件）
- `previews/*.png`（每页 preview，包含截图 meta）

---

## 使用提示

- Reveal 构建器在每次 build 完成后应生成 `qa_report.json`。
- 检查脚本应标注并汇总所有 critical issues 供人工复核。
