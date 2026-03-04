---
name: memory-manager
description: "Persistent memory management for AI agents - 3-tier file-based memory system with auto-capture, quality scoring, and session lifecycle hooks."
metadata:
  version: 2.0.0
  author: cortana
---

# Memory Manager Skill

> **Language**: Use the workspace primary language for memory content (English or Chinese are both supported).

## Overview

A comprehensive memory and persistence management system enabling AI agents to:
- **Preserve context** across multi-turn sessions
- **Capture learnings** automatically with quality filtering
- **Distill knowledge** from raw logs to long-term memory
- **Maintain session continuity** through lifecycle hooks

## Core Principle

"Text > Brain" - Every session is fresh. Context must be persisted to the filesystem.

## 3-Tier Memory Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│  L3: GLOBAL MEMORY (Long-term)                              │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  memory/global.md                                      │  │
│  │  • User preferences, core decisions, major learnings   │  │
│  │  • Manually curated or auto-distilled from L2          │  │
│  │  • Persists indefinitely                               │  │
│  └───────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│  L2: THEME-BASED (Working Memory)                           │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  memory/<theme>/YYYY-MM-DD_HH.md                      │  │
│  │  • Categorized by topic (coding, architecture, etc.)  │  │
│  │  • Structured notes with templates                    │  │
│  │  • Auto-archived after 90 days                        │  │
│  └───────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│  L1: SESSION LOGS (Raw Capture)                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  memory/sessions/YYYY-MM-DD.md                        │  │
│  │  • Every conversation turn                            │  │
│  │  • Zero-friction auto-capture                         │  │
│  │  • Auto-archived after 30 days                        │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### L1: Session Logs (Raw)
- **Purpose**: Automatic capture of every conversation turn
- **Location**: `memory/sessions/YYYY-MM-DD.md`
- **Retention**: 30 days → auto-archive
- **Trigger**: Automatic - no quality check

### L2: Theme-Based (Working)
- **Purpose**: Categorized, semi-structured notes
- **Location**: `memory/<theme>/YYYY-MM-DD_HH.md`
- **Retention**: 90 days → auto-archive
- **Trigger**: Smart detection or quick capture

### L3: Global (Long-term)
- **Purpose**: Distilled knowledge, user preferences, key decisions, and central mission control
- **Location**: `memory/global.md`
- **Retention**: Indefinite
- **Trigger**: Manual or mission status change
- **Structure Requirement**:
```markdown
# Global Knowledge & Context Control
> **System Status**: Active Session - [Scenario Name]
> **Current Date (Simulated)**: [Date]

## 1. Active Mission: [Mission Name]
**Goal**: [Objective]
**Current Phase**: [Status]
### Key Constraints
- [Fact 1]
- [Fact 2]

## 2. Research Index (L2 Memory)
- [Report Title](research/filename.md) - *Description*

## 3. Decisions (Technical & Process)
- [Decision Item]

## 4. User Preferences
- [Preference Item]
```

## Key Features

### Standard Invocation (Agent Skill Style)

Use direct script invocation as the primary contract:

```bash
python3 skills/memory-manager/scripts/memory_manager.py <command> [args]
```

Optional compatibility wrapper (project-specific):

```bash
./tools/memory-manager <command> [args]
```

### 1. Session Lifecycle Hooks

Automatic memory loading at session start and summarization at end.

```bash
# On session start - auto-load memories
python3 skills/memory-manager/scripts/memory_manager.py session-init

# On session end - auto-generate summary
python3 skills/memory-manager/scripts/memory_manager.py session-end \
  --summary '{"key_decisions": 2, "errors_encountered": 1}'
```

### 2. Smart Capture

Quality-aware persistence with automatic tier selection.

```bash
# Let system decide where to store based on quality
python3 skills/memory-manager/scripts/memory_manager.py smart-capture \
  --content "Decision: use Redis for caching layer due to high read throughput" \
  --context '{"is_decision": true}'

# Returns: {"quality_score": 85, "recommendation": "persist_l3_global", ...}
```

### 3. Automatic Turn Analysis

Analyze conversation to decide if worth capturing.

```bash
python3 skills/memory-manager/scripts/memory_manager.py should-capture \
  --user-msg "That didn't work, still getting timeout errors" \
  --agent-response "Error: connection timeout after 30s" \
  --tools '["read", "edit", "execute"]' \
  --turn-count 5

# Returns: {"should_capture": true, "capture_type": "error_log", "signals": {...}}
```

### 4. Quick Notes (Minimal Friction)

Fast capture with optional auto-theme detection.

```bash
# Quick note with manual theme
python3 skills/memory-manager/scripts/memory_manager.py quick-note \
  --content "Refactored auth module to use JWT" \
  --theme "coding"

# Quick note with auto-theme detection
python3 skills/memory-manager/scripts/memory_manager.py quick-note \
  --content "Fixed the deployment pipeline in GitHub Actions" \
  --auto-theme
  # Auto-detected as: "devops"
```

### 5. Content Quality Scoring

Before persisting, content is scored on:
- **Length** (0-30): Adequate detail?
- **Information Density** (0-40): Contains valuable patterns (decisions, errors, lessons)?
- **Uniqueness** (0-30): Not duplicate of existing memory?
- **Recency Boost** (0-20): Error or decision context?

**Total Score → Action:**
- ≥70: Persist to L3 (Global)
- ≥50: Persist to L2 (Theme)
- ≥30: Persist to L1 (Log only)
- <30: Skip

## Memory Templates

### Decision Record
```markdown
## Decision: [Title] - YYYY-MM-DD HH:MM

**Context**: [Problem or scenario]

**Options**:
1. [Option A] - [Pros/Cons]
2. [Option B] - [Pros/Cons]

**Decision**: [Final choice]

**Rationale**: [Key driver]
```

### Error Post-mortem
```markdown
## Error Post-mortem: [Summary] - YYYY-MM-DD HH:MM

**Symptom**: [Error log or behavior]

**Root Cause**: [Underlying cause]

**Fix**: [Solution applied]

**Lesson**: [How to avoid recurrence]
```

### Task Progress
```markdown
## Task Progress: [Name] - YYYY-MM-DD HH:MM

**Done**:
- [x] [Subtask 1]

**In Progress**: [Current blocker or work]

**Todo**:
- [ ] [Next step]
```

## CLI Reference

Prefix all commands below with:

```bash
python3 skills/memory-manager/scripts/memory_manager.py
```

Optional from any working directory:

```bash
python3 skills/memory-manager/scripts/memory_manager.py --workspace /path/to/workspace <command> ...
```

### Session Lifecycle
```bash
# Initialize session - loads global + recent themes
session-init [--context '{"session_id": "abc"}']

# End session - generates summary + optional distillation  
session-end [--summary '{"key_decisions": 2}'] [--no-distill]
```

### L1: Session Logs
```bash
# Log a conversation turn
log-turn --entry-type user --content "Hello" \
  [--tools '["search"]'] [--session-id "abc"]

# Read recent logs
read-logs [--days-back 7]
```

### L2: Theme Memory
```bash
# Quick capture
quick-note --content "Fixed bug" [--theme "coding"] [--auto-theme]

# Write with template
write-theme --theme "architecture" --content "..." [--template decision|error|task]

# Read theme
read-theme --theme "coding" [--hours-back 24]
```

### L3: Global Memory
```bash
read-global
write-global --content "..." [--append]
```

### Smart Features
```bash
# Quality-aware capture
smart-capture --content "..." [--theme auto] [--context '{...}']

# Analyze if turn worth capturing
should-capture --user-msg "..." --agent-response "..." --tools '[...]' --turn-count N

# Atomic capture for one turn (log + decide + persist)
capture-turn --user-msg "..." --agent-response "..." --tools '[...]' --turn-count N [--summary-content "..."]

# Score content quality
score-quality --content "..." [--context '{...}']
```

`capture-turn` returns an audit marker in JSON:
- `MEMORY_CAPTURED`: persistence happened (L1/L2/L3 path included)
- `MEMORY_SKIPPED`: no persistence (quality/trigger not met)

### Search & Maintenance
```bash
list-themes
search --query "delegation" [--max-results 10]
cleanup [--days-keep-l1 30] [--days-keep-l2 90]
```

## Auto-Theme Detection

The system automatically detects themes from content keywords:

| Theme | Keywords |
|-------|----------|
| `travel` | travel, trip, route, commute, flight, hotel, transit, taxi, itinerary, 出行, 路线, 通勤 |
| `research` | research, investigation, survey, benchmark, comparison, analysis, 调研, 对比, 评估 |
| `coding` | code, function, class, refactor, bug, fix |
| `architecture` | design, api, system, component, interface |
| `devops` | deploy, pipeline, ci/cd, docker, kubernetes |
| `data` | data, database, query, model, analytics |
| `error` | error, exception, fail, crash, bug |
| `decision` | decision, choose, select, option, alternatives |
| `misc` | (default) |

## Smart Triggers

The system automatically captures when detecting:

| Signal | Pattern | Action |
|--------|---------|--------|
| User correction | "不对", "错了", "应该是", "wrong", "incorrect" | → Error post-mortem |
| Error in response | "Error:", "Exception", "timeout" | → Error log |
| Emotional + complex | Satisfied/dissatisfied + 5+ turns | → Task summary |
| Tool-heavy milestone | 3+ tools, every 5 turns | → Progress snapshot |

## Agent Integration Guide

### For cortana.agent.md

**Session Start (Automatic):**
```yaml
on_session_start:
  - call: memory-manager/session-init
  - inject_result_into: context
```

**Every Turn (Automatic):**
```yaml
on_turn_complete:
  - call: memory-manager/log-turn
    with: {entry_type, content, tools_used}
  - call: memory-manager/should-capture
    with: {user_msg, agent_response, tools_used, turn_count}
  - if should_capture:
      call: memory-manager/smart-capture
```

**Session End (Automatic):**
```yaml
on_session_end:
  - call: memory-manager/session-end
    with: {session_summary, auto_distill: true}
```

**Explicit User Request:**
```yaml
when_user_says: ["记住", "不要忘记", "remember"]
  - call: memory-manager/quick-note
    with_auto_theme: true
```

## File Structure

```
memory/
├── global.md                    # L3: Long-term memory
├── sessions/
│   ├── 2026-02-27.md           # L1: Today's raw logs
│   └── 2026-02-26.md           # L1: Yesterday's logs
├── coding/
│   └── 2026-02-27_14.md        # L2: Coding theme
├── architecture/
│   └── 2026-02-27_15.md        # L2: Architecture theme
├── daily-summaries/
│   └── 2026-02-27_18.md        # L2: Auto-generated daily summary
└── archive/                     # Archived old memories
    ├── session_2026-01-27.md
    └── coding/
        └── 2026-01-15_10.md
```

## Migration from v1.0

**Breaking Changes:**
- Global memory moved from `MEMORY.md` to `memory/global.md`
- New L1 session logs layer
- Added quality scoring

**Migration:**
```bash
# Old global memory still readable
python3 skills/memory-manager/scripts/memory_manager.py read-global
# Will check legacy location if new location empty
```

## Error Handling

All commands return JSON:
```json
{
  "status": "success|error",
  "message": "...",
  "...": "..."
}
```
