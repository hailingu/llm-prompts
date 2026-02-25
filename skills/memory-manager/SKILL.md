---
name: memory-manager
description: "Persistent memory management for AI agents - file-based context persistence across sessions with theme-based working memory and global long-term memory."
metadata:
  version: 1.0.0
  author: cortana
---

# Memory Manager Skill

## 描述
记忆与持久化管理技能，用于在多轮会话和跨会话中保持上下文、沉淀知识、记录决策以及避免重复犯错。

## 核心原则
好记性不如烂笔头（Text > Brain）。每次会话都是全新的，必须依赖文件系统进行上下文的持久化。

## 记忆分级与存储规范

### 1. 主题工作记忆（Theme-based Working Memory）
- **组织原则**：按"内容主题-时间"的二维结构组织，避免所有信息混杂在单一的日常日志中。
- **位置**：`memory/<theme>/YYYY-MM-DD_HH.md`（`<theme>` 为当前任务、项目或话题的简写，如 `memory/agent-optimization/2026-02-25_14.md`，按小时粒度记录）。
- **触发时机**：
  - 开启新任务或新主题的会话时。
  - 发生需要跨会话保持的临时上下文。
  - 记录特定主题下的重要操作日志、临时决策、进行中的任务状态。
- **内容规范**：按时间戳或任务块记录，保持与该主题高度相关。
- **检索策略**：Agent 可通过 `list_dir` 扫描 `memory/` 目录识别主题，再通过文件名中的时间戳（`YYYY-MM-DD_HH`）快速定位特定时间段的记忆。

### 2. 全局长期记忆（Global Long-Term Memory）
- **位置**：`memory/global.md`
- **触发时机**：
  - 在主会话中自由读取与更新。
  - 定期回顾主题工作记忆，将具有跨主题、全局长期价值的内容提炼并转移至此。
- **内容规范**：提炼后的精华（重大决策、全局用户偏好、核心上下文、个人观点）。
- **安全红线**：除非用户明确要求，否则主动过滤并跳过密码/密钥等敏感信息。仅在主会话中加载和更新，严禁在共享上下文中读取。

### 3. 知识沉淀与纠错（Knowledge Accumulation & Correction）
- **经验内化**：当学到新教训时，除了更新记忆，还应主动更新对应的 `AGENTS.md`、`TOOLS.md` 或相关 Skill 文件。
- **错误免疫**：犯错后必须记录（Document it），确保不再重复相同的错误。

## 记忆模板规范

为了保持记忆的结构化和可检索性，建议在写入 `content` 时采用以下 Markdown 模板之一：

### 1. 决策记录 (Decision Record)
适用于记录关键架构选择、技术选型或方案变更。
```markdown
### 决策：[决策标题]
- **背景**：[简述面临的问题或场景]
- **选项**：
  1. [选项A] - [利弊分析]
  2. [选项B] - [利弊分析]
- **决定**：[最终选择]
- **理由**：[核心驱动因素]
```

### 2. 错误/复盘记录 (Error/Post-mortem)
适用于记录遇到的坑、报错及解决方案，防止重蹈覆辙。
```markdown
### 复盘：[错误现象/报错摘要]
- **现象**：[错误日志/表现]
- **根因**：[分析出的根本原因]
- **解决**：[最终修复方案]
- **教训**：[如何避免再次发生]
```

### 3. 任务状态快照 (Task Snapshot)
适用于跨会话保持长任务的进度。
```markdown
### 进度：[任务名称]
- **已完成**：
  - [x] [子任务1]
- **进行中**：[当前卡点/正在做的事]
- **待办**：
  - [ ] [下一步计划]
```

## 使用工作流
1. **会话初始化**：读取 `memory/global.md` 获取全局上下文和用户偏好。根据当前任务，通过 `list_dir` 扫描 `memory/` 目录识别相关主题，读取对应的 `memory/<theme>/*.md` 获取主题上下文。
2. **构建内容**：根据当前场景选择合适的**记忆模板**，填充内容。
3. **任务结束/复盘**：提取有全局长期价值的信息，追加或更新到 `memory/global.md`，并清理或归档已完结的主题记忆。

## CLI 使用示例

### 基础命令
```bash
# 读取全局记忆
python3 skills/memory-manager/scripts/memory_manager.py read-global

# 写入全局记忆（追加模式）
python3 skills/memory-manager/scripts/memory_manager.py write-global \
  --content "User prefers Python over Java for new projects" \
  --append

# 读取主题记忆
python3 skills/memory-manager/scripts/memory_manager.py read-theme \
  --theme agent-optimization \
  --hours-back 48

# 写入主题记忆
python3 skills/memory-manager/scripts/memory_manager.py write-theme \
  --theme stock-tracker \
  --content "Created stock price tracker skill with Yahoo Finance API"

# 列出所有主题
python3 skills/memory-manager/scripts/memory_manager.py list-themes

# 搜索记忆
python3 skills/memory-manager/scripts/memory_manager.py search \
  --query "stock" \
  --max-results 5
```

### Agent 集成示例
```python
import subprocess
import json

def read_global_memory():
    """读取全局记忆"""
    result = subprocess.run(
        ["python3", "skills/memory-manager/scripts/memory_manager.py", "read-global"],
        capture_output=True,
        text=True
    )
    return json.loads(result.stdout)

def write_theme_memory(theme, content):
    """写入主题记忆"""
    result = subprocess.run(
        ["python3", "skills/memory-manager/scripts/memory_manager.py", "write-theme",
         "--theme", theme, "--content", content],
        capture_output=True,
        text=True
    )
    return json.loads(result.stdout)

# 使用示例
memory = read_global_memory()
if memory["status"] == "success":
    print(f"Global memory: {memory['content'][:100]}...")
```

## 输出格式

### read-global 输出
```json
{
  "status": "success",
  "content": "# Global Memory\n\n## System Context..."
}
```

### read-theme 输出
```json
{
  "status": "success",
  "theme": "agent-optimization",
  "memories": [
    {
      "timestamp": "2026-02-25T14:00:00",
      "filename": "2026-02-25_14.md",
      "content": "## 2026-02-25 14:00:00\n\nOptimized cortana agent memory handling..."
    }
  ]
}
```

### search 输出
```json
{
  "status": "success",
  "query": "stock",
  "results": [
    {
      "type": "global",
      "match": "MEMORY.md",
      "content": "Stock price tracker skill created..."
    }
  ]
}
```

## 文件结构
```
memory/
├── global.md           # 全局长期记忆
├── agent-optimization/
│   ├── 2026-02-25_14.md
│   └── 2026-02-25_15.md
├── stock-tracker/
│   └── 2026-02-25_14.md
└── other-theme/
    └── 2026-02-25_13.md
```

## 错误处理
所有命令返回JSON格式，包含 `status` 字段：
- `"success"`: 操作成功
- `"error"`: 操作失败，`message` 字段包含错误信息
