---
name: <short-name>
description: <short description>
version: v0.1
owner: <team-or-person>
tools: [search, read, edit]
capabilities:
  - intent-parsing
  - plugin-routing
  - skill-sdk
  - security-audit
  - telemetry
workflows:
  - name: run-intent
    request: { input: "...", context: {...}, metadata: {...} }
    response: { action: "...", result: {...}, trace: [...] }
constraints:
  - do-not-expose-secrets
  - require-audit-trace
evaluation:
  - e2e-tests
  - trace-based-metrics
  - uptime-slo
examples:
  - input: "Schedule a meeting tomorrow at 10am"
    output: { action: "create_event", result: { success: true, event_id: "..." } }
---

# Agent Template — 标准化文档模板

> 用途：为团队内部 agent 文档提供统一、机器友好的模板，便于审查、自动化索引与维护。

## Persona

简短一句描述 agent 的角色与职责，例如：

> 你是一个代码审查专家，优先关注安全、样式和 API 兼容性，并输出清单形式的修复建议。

## 概要

1–3 段落，说明 agent 的目标场景、适用范围与高层能力。

## 能力（Capabilities） 🔧

以简短条目列出 agent 提供的主要能力（应与 frontmatter 中 capabilities 对齐）。

## 工作流（Workflows） 🔁

列出典型工作流，包含请求/响应示例：

```http
POST /v1/agent/run
Body: { "input": "...", "context": {...} }
Response: { "action": "...", "result": {...}, "trace": [...] }
```

## 约束（Constraints） ⚠️

明确必须遵守的安全/隐私/合规性要求，例如不要在响应或日志中泄露机密，外部调用必须审计等。

## 评估（Evaluation） 📊

给出建议的评估指标（如：响应正确率、trace 评分、SLO 指标）与测试策略（单元/集成/E2E）。

## 示例（Examples） ✏️

提供 1–3 个典型输入/输出示例，用于快速验证实现是否符合预期。

## 测试建议

列出需要覆盖的重要测试用例或场景（边界条件、错误路径、降级与超时）。

## 版本与发布策略

简要说明版本管理、发布前合规检查项（例如商标/法律/隐私评审）与发布流程要点。

## TODO / 任务清单

用于跟踪尚未完成的工作项（负责人、里程碑等）。

---

### 维护提示

- 在修改 `frontmatter` 时保持字段可机器解析（YAML 格式）。
- 尽量提供结构化示例（JSON）以便自动化测试与文档生成。
- 若需要在仓库中批量标准化 agent，请先创建 CI 检查来验证 frontmatter 字段存在性与类型。
