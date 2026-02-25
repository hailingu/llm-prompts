---
name: memory-manager
description: "Persistent memory management for AI agents - file-based context persistence across sessions with theme-based working memory and global long-term memory."
metadata:
  version: 1.0.0
  author: cortana
---

# Memory Manager Skill

> **Language**: All memory content must be written in **English only**.

## Overview
Skill for memory and persistence management, enabling context preservation across multi-turn sessions, knowledge accumulation, decision recording, and error avoidance.

## Core Principle
"Text > Brain" - Every session is fresh. Context must be persisted to the filesystem.

## Memory Hierarchy

### 1. Theme-based Working Memory
- **Organization**: Organized by "theme-time" 2D structure, avoiding information clutter.
- **Location**: `memory/<theme>/YYYY-MM-DD_HH.md` (e.g., `memory/agent-optimization/2026-02-25_14.md`)
- **Triggers**:
  - Starting a new task or theme session
  - Context that needs to persist across sessions
  - Important logs, temporary decisions, task states for a specific theme
- **Retrieval**: Use `list_dir` to scan `memory/` for themes, then use filename timestamps (`YYYY-MM-DD_HH`) to locate specific time periods.

### 2. Global Long-Term Memory
- **Location**: `memory/global.md`
- **Triggers**:
  - Read/update freely in main sessions
  - Periodically review theme memories and distill cross-theme, long-term value
- **Content**: Distilled essence (major decisions, user preferences, core context, personal insights)
- **Security**: Filter sensitive info (passwords, keys) unless explicitly requested. Only load/update in main sessions.

### 3. Knowledge Accumulation & Correction
- **Experience Internalization**: When learning new lessons, update relevant Agent/Skill docs
- **Error Immunity**: Document mistakes to prevent repetition

## Memory Templates

Use these Markdown templates for structured, searchable content:

### 1. Decision Record
For recording key architecture choices, tech stack decisions, or plan changes.
```markdown
### Decision: [Title]
- **Context**: [Problem or scenario]
- **Options**:
  1. [Option A] - [Pros/Cons]
  2. [Option B] - [Pros/Cons]
- **Decision**: [Final choice]
- **Rationale**: [Key driver]
```

### 2. Error / Post-mortem
For recording pitfalls, errors, and solutions.
```markdown
### Post-mortem: [Error summary]
- **Symptom**: [Error log or behavior]
- **Root Cause**: [Underlying cause]
- **Fix**: [Solution applied]
- **Lesson**: [How to avoid recurrence]
```

### 3. Task Snapshot
For persisting long-running task progress across sessions.
```markdown
### Progress: [Task name]
- **Done**:
  - [x] [Subtask 1]
- **In Progress**: [Current blocker or work]
- **Todo**:
  - [ ] [Next step]
```

## Usage Workflow

1. **Session Init**: Read `memory/global.md` for global context and user preferences. Scan `memory/` for relevant themes using `list_dir`, read theme-specific files.
2. **Build Content**: Choose appropriate template, fill in content.
3. **Session End/Review**: Distill long-term value, update `memory/global.md`, archive completed themes.

## CLI Examples

```bash
# Read global memory
python3 skills/memory-manager/scripts/memory_manager.py read-global

# Write to global memory (append mode)
python3 skills/memory-manager/scripts/memory_manager.py write-global \
  --content "User prefers Python over Java for new projects" \
  --append

# Read theme memory
python3 skills/memory-manager/scripts/memory_manager.py read-theme \
  --theme agent-optimization \
  --hours-back 48

# Write theme memory
python3 skills/memory-manager/scripts/memory_manager.py write-theme \
  --theme stock-tracker \
  --content "Created stock price tracker skill with Yahoo Finance API"

# List all themes
python3 skills/memory-manager/scripts/memory_manager.py list-themes

# Search memory
python3 skills/memory-manager/scripts/memory_manager.py search \
  --query "stock" \
  --max-results 5
```

## Output Formats

### read-global
```json
{
  "status": "success",
  "content": "# Global Context\n\n## User Profile..."
}
```

### read-theme
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

### search
```json
{
  "status": "success",
  "query": "stock",
  "results": [
    {
      "type": "global",
      "match": "global.md",
      "content": "Stock price tracker skill created..."
    }
  ]
}
```

## File Structure
```
memory/
├── global.md           # Global long-term memory
├── agent-optimization/
│   ├── 2026-02-25_14.md
│   └── 2026-02-25_15.md
├── stock-tracker/
│   └── 2026-02-25_14.md
└── other-theme/
    └── 2026-02-25_13.md
```

## Error Handling
All commands return JSON with `status`:
- `"success"`: Operation successful
- `"error"`: Operation failed, `message` contains error details
