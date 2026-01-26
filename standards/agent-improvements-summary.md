# Agent 架构修复总结

**Date**: 2026-01-24  
**修复范围**: java-architect, java-api-designer agents 和 google-design-doc-standards

---

## 修复的主要问题

### 1. **章节编号不一致问题** ✅ 已修复

**问题**:
- 标准规范使用 Section 5 (Data Model), Section 6 (Concurrency Requirements)
- java-architect agent 说这些由 api-designer 负责（Section 11, 12）
- 造成引用混乱和职责不明

**修复方案**:
```
Level 1 (java-architect):
1. Context and Scope
2. Goals and Non-Goals
3. Design Overview
4. API Design Guidelines (4.1-4.4, 新增 4.4 API Overview)
5. Data Model Overview (5.1 Key Entities + 5.2 Data Architecture)
6. Concurrency Requirements Overview (6.1 Performance Targets + 6.2 Thread Safety Strategy)
7. Security Architecture
8. Cross-Cutting Concerns
9. Implementation Constraints
10. Alternatives Considered

Level 2 (java-api-designer):
- Section 4.1: API Interface Definition (补充完整方法签名)
- Section 4.2: Design Rationale (补充 Contract + Caller Guidance)
- Section 4.3: Dependency Interfaces (补充完整依赖接口)
- Section 5: Data Model Details (补充详细字段定义)
- Section 6: Concurrency Contract (补充每个方法的线程安全契约)
```

---

### 2. **缺少"接口概念设计"** ✅ 已修复

**问题**:
- java-architect 完全不提 API 的概念设计
- 直接 handoff 给 java-api-designer 可能导致方向偏差
- 不符合 Google Tech Lead 的实践（Tech Lead 通常会定义接口骨架）

**修复方案**:
在 java-architect 的 Section 4.4 新增 **API Overview (High-level)**:

```markdown
### API Overview
**Public APIs** (概念层面，完整签名由 @java-api-designer 定义):
- `verify(apiKey)`: 验证订阅状态，返回订阅信息或 null
- `startPeriodicCheck(interval)`: 启动周期性检查
- `stopPeriodicCheck()`: 停止周期性检查

**Dependency Interfaces** (概念层面):
- `HttpSender`: 发送 HTTP 请求
- `ConfigProvider`: 读取配置
```

**注意**: 这只是接口的骨架，不包括完整的参数类型、返回类型、异常声明

---

### 3. **Data Model 和 Concurrency Requirements 归属不明确** ✅ 已修复

**问题**:
- 标准规范说 Data Model 和 Concurrency Requirements 是必需章节（应该在 Level 1）
- architect agent 说这些由 api-designer 负责
- 导致 java-coder-specialist 无法直接使用 architect 的产出

**修复方案**:
- **architect 负责**: Data Model **Overview**（Section 5.1 Key Entities）+ Concurrency Requirements **Overview**（Section 6）
- **api-designer 负责**: Data Model **Details**（Section 5 详细字段定义）+ Concurrency **Contract**（Section 6 每个方法的并发行为）

**职责划分示例**:

| 内容 | architect (Level 1) | api-designer (Level 2) |
|------|---------------------|------------------------|
| Data Model | 关键实体概要（Subscription, Config） | 详细字段定义（类型、约束、Javadoc） |
| Concurrency | 性能目标（100 QPS, p95 < 50ms）| 每个方法的线程安全契约 |
| Concurrency | 线程安全策略（哪些组件需要线程安全） | 具体并发行为（如何保证线程安全） |

---

### 4. **串行依赖导致效率低** ✅ 部分改善

**问题**:
- java-coder-specialist 和 java-doc-writer 都必须等待 api-designer 完成
- 协作流程是：architect → api-designer → (coder + doc-writer)

**改善方案**:
现在 architect 提供了更完整的信息（API Overview, Data Model Overview, Concurrency Requirements），coder 可以：
1. 在 api-designer 补充细节之前，先基于 architect 的产出搭建框架
2. 理解架构方向和性能目标
3. 准备基础设施（数据库、缓存、框架配置）

**仍需等待**:
- 完整的 API Interface Definition (Section 4.1)
- Design Rationale (Section 4.2) - coder 需要理解 Contract
- Caller Guidance (Section 4.2) - doc-writer 需要提取用户指导

---

## 修改的文件

### 1. `/Users/guhailin/Git/app-center/.github/agents/java-architect.agent.md`

**主要变更**:
- ✅ 更新 Section 3 Design Overview：增加"API Overview (High-level)"
- ✅ 新增 Section 4.4: API Overview (High-level)
- ✅ 新增 Section 5: Data Model (Overview)
  - 5.1 Key Entities
  - 5.2 Data Architecture
- ✅ 新增 Section 6: Concurrency Requirements (Overview)
  - 6.1 Performance Targets
  - 6.2 Thread Safety Strategy
- ✅ 更新章节编号：Security Architecture (6→7), Cross-Cutting Concerns (7→8), Implementation Constraints (8→9), Alternatives Considered (9→10)
- ✅ 更新"不生成"列表，明确 api-designer 负责"补充细节"而非"从零开始定义"
- ✅ 更新 Handoff Message 模板，列出 architect 已完成的章节

### 2. `/Users/guhailin/Git/app-center/.github/agents/java-api-designer.agent.md`

**主要变更**:
- ✅ 更新 Phase 0 输入预期：增加对 Section 4.4 (API Overview), 5.1 (Data Model Overview), 6 (Concurrency Requirements Overview) 的读取
- ✅ 更新 Phase 0 验证清单：检查 architect 是否提供了 API Overview、Data Model Overview、Concurrency Requirements
- ✅ 更新反馈机制示例：增加"接口骨架缺失"场景
- ✅ 更新 Phase 1 目标：从"定义完整 API"改为"基于 API Overview 补充完整方法签名"
- ✅ 更新输出位置：从 Section 10.1 改为 Section 4.1
- ✅ 更新所有章节引用（4.1, 4.4, 5.1, 5.2, 6, 7, 9, 10）

### 3. `/Users/guhailin/Git/app-center/.github/standards/google-design-doc-standards.md`

**说明**: 该文件无需修改，因为：
- 标准规范是"理想状态"的文档结构
- 现在的 agent 职责划分与标准规范一致：
  - architect 负责 Level 1 (Section 1-10 概要)
  - api-designer 负责 Level 2 (补充 Section 4-6 细节)

---

## 对比：修复前 vs 修复后

### 修复前

```
java-architect (Level 1):
  - Section 1-3: Context, Goals, Design Overview
  - Section 4-8: API Guidelines, Data Architecture, Security, Cross-cutting, Constraints
  - Section 9: Alternatives
  - ❌ 不定义 API 接口（完全由 api-designer 负责）
  - ❌ 不定义 Data Model（完全由 api-designer 负责）
  - ❌ 不定义 Concurrency Requirements（完全由 api-designer 负责）
  
  ↓ handoff
  
java-api-designer (Level 2):
  - Section 10: API Interface Definition（从零开始设计接口）
  - Section 11: Data Model（从零开始定义实体）
  - Section 12: Concurrency Requirements（从零开始定义并发要求）
  
  ↓ handoff (parallel)
  ├→ java-coder-specialist (必须等 api-designer 完成，因为 architect 没提供 API 概念)
  └→ java-doc-writer (必须等 api-designer 完成，因为 architect 没提供 API 概念)
```

**问题**:
- ❌ 章节编号不连续（1-9, 10-12）
- ❌ api-designer 不知道应该设计哪些接口（architect 没提供方向）
- ❌ coder 完全无法在 api-designer 之前开始工作

### 修复后

```
java-architect (Level 1 - Overview):
  - Section 1-3: Context, Goals, Design Overview (包含 API Overview 骨架)
  - Section 4: API Design Guidelines (4.1-4.3) + API Overview (4.4 新增)
  - Section 5: Data Model Overview (5.1) + Data Architecture (5.2)
  - Section 6: Concurrency Requirements Overview (6.1 + 6.2)
  - Section 7-10: Security, Cross-cutting, Constraints, Alternatives
  
  ↓ handoff
  
java-api-designer (Level 2 - Details):
  - Section 4.1: API Interface Definition（基于 4.4 补充完整方法签名）
  - Section 4.2: Design Rationale（补充 Contract + Caller Guidance）
  - Section 4.3: Dependency Interfaces（基于 4.4 补充完整依赖接口）
  - Section 5: Data Model Details（基于 5.1 补充详细字段定义）
  - Section 6: Concurrency Contract（基于 6.1-6.2 补充每个方法的线程安全契约）
  
  ↓ handoff (parallel)
  ├→ java-coder-specialist (可以基于 architect 的 API Overview 和 Concurrency Requirements 开始准备工作)
  └→ java-doc-writer (可以基于 architect 的 API Overview 开始准备文档结构)
```

**改进**:
- ✅ 章节编号连续（1-10）
- ✅ api-designer 有明确的方向（基于 architect 的 API Overview）
- ✅ coder 可以提前开始准备工作（理解架构、搭建框架）
- ✅ 职责清晰：architect 负责"What to build"，api-designer 负责"How to interact"

---

## 符合 Google 实践的改进

### Google 的实际做法

```
Tech Lead 编写 Design Doc:
  - Context & Goals
  - High-level Architecture
  - **API Overview** (接口骨架，例如："需要一个 Verify(apiKey) 方法")
  - Data Model Overview（关键实体）
  - Performance Requirements（QPS 目标）
  
  ↓
  
Engineer 编写 .proto files (或 Java Interface):
  - **完整的方法签名**（基于 Tech Lead 的 API Overview）
  - 详细的字段定义（基于 Tech Lead 的 Data Model Overview）
  - 精确的行为契约（Contract）
```

### 我们的修复后做法

```
java-architect 编写 Level 1:
  - Context & Goals
  - High-level Architecture
  - **API Overview** (接口骨架) ← 新增，符合 Google 实践
  - Data Model Overview（关键实体） ← 新增，符合 Google 实践
  - Concurrency Requirements（QPS 目标） ← 新增，符合 Google 实践
  
  ↓
  
java-api-designer 编写 Level 2:
  - **完整的方法签名**（基于 architect 的 API Overview） ← 符合 Google 实践
  - 详细的字段定义（基于 architect 的 Data Model Overview） ← 符合 Google 实践
  - 精确的行为契约（Contract + Caller Guidance） ← 符合 Google 实践
```

**结论**: ✅ 修复后的设计**完全符合** Google Tech Lead → Engineer 的协作模式

---

## 后续建议

### 1. **验证修复效果**

建议使用真实场景测试：
1. java-architect 生成一个设计文档
2. 检查是否包含 Section 4.4 API Overview
3. 检查是否包含 Section 5.1 Data Model Overview
4. 检查是否包含 Section 6 Concurrency Requirements
5. handoff 给 java-api-designer
6. 检查 api-designer 是否能正确补充细节

### 2. **潜在优化（Optional）**

如果团队规模较小，可以考虑使用**标准规范中建议的 Option B**: Combined Agent
```
java-architect (Level 1 + Level 2)
  ↓ handoff (parallel)
  ├→ java-coder-specialist
  └→ java-doc-writer
```

优点：
- ✅ 减少一次 handoff
- ✅ 单一文档，内容连贯
- ✅ 适合中小型项目

缺点：
- ⚠️ java-architect 职责较重

### 3. **文档模板更新**

建议创建一个完整的设计文档模板文件：
`/Users/guhailin/Git/app-center/.github/standards/design-doc-template.md`

包含：
- 所有章节的标题和说明
- 每个章节的示例内容
- architect 和 api-designer 的职责标注

---

## 总结

✅ **所有核心问题已修复**：
1. ✅ 章节编号统一（1-10 连续）
2. ✅ 增加了"接口概念设计"（Section 4.4 API Overview）
3. ✅ 明确了 Data Model 和 Concurrency Requirements 的归属（architect 负责概要，api-designer 负责细节）
4. ✅ 部分改善了串行依赖问题（coder 可以提前开始准备工作）
5. ✅ 完全符合 Google Tech Lead → Engineer 的协作模式

🎯 **现在的架构设计流程**：
- **清晰的职责边界**：architect 定义"What to build"，api-designer 定义"How to interact"
- **完整的上下文传递**：api-designer 有明确的方向（API Overview），coder 有足够的信息开始准备
- **符合业界实践**：完全对标 Google 的 Design Doc + Protocol Buffers 模式
