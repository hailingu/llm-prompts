# Agent 协作规范与迭代控制

**Purpose**: 定义 Java 开发 Agent 之间的协作流程、迭代限制和升级机制，防止无限循环并确保高效协作。

**Version**: 1.0  
**Last Updated**: 2026-01-24

---

## 协作流程总览

```
                        ┌─────────────────────┐
                        │   java-tech-lead    │
                        │  (审批 + 仲裁)        │
                        └──────────┬──────────┘
                                   │
            ┌──────────────────────┼──────────────────────┐
            │ Review               │ Review               │ Review
            ▼                      ▼                      ▼
    ┌───────────────┐      ┌───────────────┐      ┌───────────────┐
    │java-architect │      │ java-coder-   │      │java-doc-writer│
    │  (Level 1)    │      │ specialist    │      │  (文档)        │
    └───────┬───────┘      └───────────────┘      └───────────────┘
            │                      ▲
            │ Handoff              │ Handoff
            ▼                      │
    ┌───────────────┐              │
    │java-api-      │──────────────┘
    │designer       │
    │  (Level 2)    │
    └───────────────┘
```

---

## 迭代限制规则

### 规则 1: 最大迭代次数 = 3

任何两个 Agent 之间的反馈循环，最多允许 **3 次迭代**。

| 场景 | 允许迭代 | 超过后处理 |
|------|---------|-----------|
| architect ↔ api-designer | 3 次 | 升级到 tech-lead |
| api-designer ↔ coder | 3 次 | 升级到 tech-lead |
| coder ↔ api-designer | 3 次 | 升级到 tech-lead |
| doc-writer ↔ api-designer | 3 次 | 升级到 tech-lead |

### 规则 2: 迭代计数方式

```markdown
Iteration 1: Agent A → Agent B (初始请求)
Iteration 2: Agent B → Agent A (反馈/修改请求)
Iteration 3: Agent A → Agent B (修改后重新提交)
Iteration 4: ❌ 超过限制，必须升级到 tech-lead
```

### 规则 3: 迭代追踪模板

每次反馈时，必须在消息中包含迭代计数：

```markdown
## Feedback (Iteration 2/3)

**From**: @java-coder-specialist
**To**: @java-api-designer
**Remaining Iterations**: 1

**Issue**: [问题描述]

**Request**: [请求内容]

---
⚠️ 注意：如果本次修改后仍有问题，下次反馈将自动升级到 @java-tech-lead
```

---

## 升级机制

### 自动升级触发条件

1. **迭代超时**: 迭代次数 > 3
2. **明确请求**: Agent 声明无法继续
3. **冲突僵局**: 两个 Agent 立场矛盾无法调和
4. **阻塞超时**: 等待响应 > 24 小时

### 升级消息模板

```markdown
@java-tech-lead 需要仲裁

## 升级类型
- [ ] 迭代超时 (Iteration > 3)
- [ ] 无法继续
- [ ] 立场冲突
- [ ] 阻塞超时

## 涉及 Agent
- @agent1
- @agent2

## 问题描述
[详细描述问题]

## 历史迭代摘要
| Iteration | From | To | Summary |
|-----------|------|-----|---------|
| 1 | @agent1 | @agent2 | [初始请求] |
| 2 | @agent2 | @agent1 | [反馈：问题X] |
| 3 | @agent1 | @agent2 | [修改后重新提交] |
| 4 | @agent2 | @agent1 | [仍有问题Y] ← 超过限制 |

## 双方立场
**@agent1 立场**: [描述]
**@agent2 立场**: [描述]

## 请求
请做出最终决策
```

---

## 降级产出策略

当设计文档或上游产出不完整时，不应完全阻塞，而是采用降级策略。

### 策略 1: 最小可行产出 (MVP Output)

```markdown
## 降级产出声明

**原因**: [上游产出不完整的具体问题]

**降级内容**: 
本次产出基于不完整的输入，以下部分标记为"待补充"：
- [ ] [待补充项1]
- [ ] [待补充项2]

**待上游补充后**: 
请 @[上游agent] 补充以下信息，我将更新产出：
- [需要的信息1]
- [需要的信息2]
```

### 策略 2: 假设并标注

```markdown
## 基于假设的产出

由于上游未明确以下信息，我基于假设进行产出：

| 项目 | 假设值 | 如果假设错误的影响 |
|------|-------|------------------|
| 错误处理策略 | 返回 null | 需要修改返回值处理 |
| 并发要求 | 100 QPS | 可能需要调整同步机制 |

⚠️ **风险**: 如果假设错误，需要返工

@[上游agent] 请确认这些假设是否正确
```

### 策略 3: 分阶段交付

```markdown
## 分阶段产出

由于上游产出不完整，采用分阶段交付：

### 阶段 1: 已完成 ✅
- [已完成的部分]

### 阶段 2: 待上游补充后完成 ⏳
- 依赖: [需要的上游输入]
- 预计完成: 收到上游补充后 1 天内

### 阶段 3: 可选优化 📋
- [后续可优化的部分]
```

---

## 质量门禁 (Quality Gates)

### Gate 1: Design Approved

**进入条件**:
- [ ] Level 1 Architecture Design 完成
- [ ] Level 2 API Specification 完成
- [ ] @java-tech-lead 审批通过
- [ ] 迭代次数 ≤ 3

**允许操作**: @java-coder-specialist 开始实现

### Gate 2: Implementation Approved

**进入条件**:
- [ ] 代码实现完成
- [ ] 所有 static analysis 通过
- [ ] 测试覆盖率 ≥ 80%
- [ ] @java-tech-lead 审批通过
- [ ] 迭代次数 ≤ 3

**允许操作**: @java-doc-writer 开始文档

### Gate 3: Documentation Approved

**进入条件**:
- [ ] 用户文档完成
- [ ] API 参考完成
- [ ] @java-tech-lead 审批通过

**允许操作**: 模块发布

---

## 反模式警示

### ❌ Anti-pattern 1: 无限循环

```
coder → api-designer → coder → api-designer → ...
```

**问题**: 缺少迭代限制，永远无法完成

**正确做法**: 3 次迭代后升级到 tech-lead

### ❌ Anti-pattern 2: 跳过审批

```
architect → coder (跳过 api-designer)
```

**问题**: 缺少 API 契约定义，coder 可能实现错误

**正确做法**: 严格按流程 architect → api-designer → coder

### ❌ Anti-pattern 3: 完全阻塞

```
doc-writer: "设计文档不完整，我无法产出任何内容"
```

**问题**: 完全阻塞，无进展

**正确做法**: 使用降级策略，产出最小可行内容

### ❌ Anti-pattern 4: 无记录反馈

```
coder: "API 设计有问题"
(没有说明具体问题、没有迭代计数)
```

**问题**: 模糊反馈，无法追踪

**正确做法**: 使用迭代追踪模板，明确问题和迭代次数

---

## 各 Agent 协作职责

### java-architect
- **产出**: Level 1 Architecture Design
- **接收反馈自**: @java-api-designer
- **提交审批给**: @java-tech-lead
- **升级条件**: 与 api-designer 迭代 > 3 次

### java-api-designer
- **产出**: Level 2 API Specification
- **接收反馈自**: @java-architect, @java-coder-specialist, @java-doc-writer
- **提交审批给**: @java-tech-lead
- **升级条件**: 与任何 agent 迭代 > 3 次

### java-coder-specialist
- **产出**: 代码实现
- **接收反馈自**: @java-api-designer
- **提交审批给**: @java-tech-lead
- **升级条件**: 与 api-designer 迭代 > 3 次

### java-doc-writer
- **产出**: 用户文档
- **接收反馈自**: @java-api-designer
- **提交审批给**: @java-tech-lead
- **升级条件**: 与 api-designer 迭代 > 3 次

### java-tech-lead
- **职责**: 审批、仲裁、质量把关
- **接收请求自**: 所有 agent
- **最终决策权**: 是

---

## 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| 1.0 | 2026-01-24 | 初始版本 |
