---
name: memory-manager
description: "Persistent memory management for AI agents - 3-tier file-based memory system with mandatory L1 raw logs and optional L2/L3 extraction."
metadata:
  version: 2.0.0
  author: cortana
---

# Memory Manager Skill

> **Language**: Use the workspace primary language for memory content (English or Chinese are both supported).

## Overview

A comprehensive memory and persistence management system enabling AI agents to:
- **Preserve context** across multi-turn sessions
- **Write raw interaction logs** on every material turn and extract reusable knowledge only when warranted
- **Distill knowledge** from raw logs to long-term memory
- **Persist useful acquired content** so agents do not need to rediscover it
- **Maintain session continuity** through lightweight session initialization and recent-memory lookup

## Core Principle

"Text > Brain" - Every session is fresh. Context must be persisted to the filesystem.

## Storage Policy

- For this repository, the canonical memory location is the workspace `memory/` directory.
- Persist project memory through `python3 skills/memory-manager/scripts/memory_manager.py ...` so writes land in `./memory/`.
- Do not route this repository's reusable memory outside `./memory/`.

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
│  │  • Append-only raw turn logging                       │  │
│  │  • Retention follows repo memory policy               │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### L1: Session Logs (Raw)
- **Purpose**: Raw record of every material conversation turn
- **Location**: `memory/sessions/YYYY-MM-DD.md`
- **Retention**: Managed by repository policy
- **Trigger**: `persist-turn` or `log-turn`

### L2: Theme-Based (Working)
- **Purpose**: Categorized, semi-structured notes, fetched facts, and interim research
- **Location**: `memory/<theme>/YYYY-MM-DD_HH.md`
- **Retention**: 90 days → auto-archive
- **Trigger**: Smart detection or quick capture

### L3: Global (Long-term)
- **Purpose**: Distilled knowledge, user preferences, key decisions
- **Location**: `memory/global.md`
- **Retention**: Indefinite
- **Trigger**: Manual or auto-distill from high-value sessions
- **Read Contract**: Cortana reads `memory/global.md` in this order before complex, preference-sensitive, or routing tasks:
  1. `Active Mission`
  2. `Key Constraints`
  3. `Research Index`
  4. `User Preferences`

`Research Index` is a discovery aid, not a hard reuse gate:
- Reuse an entry when it is directly relevant and still usable.
- If no entry is a close fit, continue with fresh search or tool verification.
- After gathering useful material, write it into `memory/` even if you have not fully distilled it yet.

Recommended top-level skeleton:

```markdown
# Global Memory

Read Order: Active Mission -> Key Constraints -> Research Index -> User Preferences

## 1. Active Mission
### Mission Snapshot
### Key Constraints (Immutable)
### Next Useful Reads

## 2. Research Index

## 3. Decisions

## 4. User Preferences
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

### 1. Session Initialization

Use `session-init` once per new chat/session to load global memory and a recent memory index.

**Practical default**: call `session-init` once per new chat/session. Do **not** call it on every reply.

```bash
python3 skills/memory-manager/scripts/memory_manager.py session-init --session-id "abc"
```

### 2. Core Persistence Flow

The latest standard is deliberately simple:

1. write raw interaction content to `L1` in `memory/sessions/...`
2. write extracted reusable content to `L2` in `memory/<theme>/...` when useful
3. optionally append durable constraints/preferences to `L3` in `memory/global.md`

**Repository-specific requirement for this workspace**: every interaction must first produce an `L1` raw write in `memory/sessions/...`. Any `L2` or `L3` write is additive and does not replace the mandatory `L1` write.

Use `persist-turn` as the default command for interactive work because it enforces the required order in one call.

```bash
python3 skills/memory-manager/scripts/memory_manager.py persist-turn \
  --entry-type assistant \
  --raw-content "User asked for a memory cleanup plan. Assistant decided to standardize L1 and L2 writes." \
  --extracted-content "Decision: use persist-turn as the default two-level persistence command." \
  --theme decision \
  --template decision
```

If you only have raw content and no extracted value yet, use `log-turn` first and write `L2` later in the same turn when appropriate.

```bash
python3 skills/memory-manager/scripts/memory_manager.py log-turn \
  --entry-type assistant \
  --content "Raw verification notes and working output."
```

### 3. Direct L2 And L3 Writes

Use `write-theme` for deliberate extracted-note writes outside the main turn flow, such as backfilling verified facts or importing distilled notes.

```bash
python3 skills/memory-manager/scripts/memory_manager.py write-theme \
  --theme preferences \
  --content "User prefers concise conclusions first, then evidence." \
  --promote-global
```

`write-theme` also writes an `L1` session entry before the `L2` note. Use
`--raw-content` if the raw turn text should differ from the extracted note.

Use `write-global` only when you explicitly need to append durable constraints or preferences to `memory/global.md`.

```bash
python3 skills/memory-manager/scripts/memory_manager.py write-global \
  --content "## Durable Preference\n\nDefault to conclusion first, then evidence." \
  --append
```

`write-global` also writes an `L1` session entry before appending to `L3`.

### 4. CSV Data Memory

Use `write-data` when the memory payload is tabular and should live as a CSV file.

```bash
python3 skills/memory-manager/scripts/memory_manager.py write-data \
  --name market_snapshot.csv \
  --csv-content "date,price\n2026-03-07,91.2" \
  --description "Daily market snapshot" \
  --source-label "manual_capture"
```

- **Location**: `memory/data/*.csv`
- **Manifest**: `memory/data/manifest.json` tracks source label, description, columns, update time, row count, and file size
- **Semantics**: the command writes an `L1` session log entry first, then stores the CSV file under `memory/data/` and updates the manifest
- **Input**: provide exactly one of `--csv-content` or `--source-file`
- **Reading**: use `read-data` to load one dataset or `list-data` to inspect all stored CSV files and their metadata

Memory is a general workflow, not a research-only feature:

- question / search / discussion
- write `L1` raw memory immediately
- extract to `L2` when value appears
- promote to `L3` only for durable repo-wide facts or preferences

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
session-init [--session-id "abc"] [--recent-days 7] [--theme-limit 8] [--no-log]
```

### L1: Session Logs
```bash
# Log a conversation turn
log-turn --entry-type user --content "Hello" \
  [--tools '["search"]'] [--session-id "abc"]

# Preferred repository flow: raw L1 first, then optional L2 extraction
persist-turn --entry-type assistant --raw-content "Raw interaction log" \
  [--extracted-content "Reusable conclusion"] [--theme research] \
  [--template decision|error|task] [--title "Optional heading"] \
  [--tools '["read"]'] [--promote-global]

# Read recent logs
read-logs [--days-back 7] [--limit 20]
```

### L2: Theme Memory
```bash
# Write with a caller-provided theme; an L1 entry is written first
write-theme --theme research --content "..." [--template decision|error|task] \
  [--title "Optional heading"] [--promote-global] [--raw-content "..."]

# Read theme
read-theme --theme "coding" [--hours-back 24] [--limit 20]
```

### L3: Global Memory
```bash
read-global
write-global --content "..." [--append] [--raw-content "..."]
```

### CSV Data Memory
```bash
write-data --name "metrics.csv" --csv-content "date,value\n2026-03-07,1" \
  [--description "..."] [--source-label "..."] [--columns-json '["date","value"]']
write-data --name "metrics.csv" --source-file /tmp/metrics.csv [--replace]
read-data --name "metrics.csv" [--head 5]
list-data
```

## Preferred Two-Level Workflow

For this repository, prefer `persist-turn` when you need one command that obeys
the mandatory storage order:

1. write raw interaction content to `L1` in `memory/sessions/...`
2. write extracted reusable content to `L2` when provided
3. optionally promote durable extracted content to `L3`

Use `write-theme` and `write-global` only for deliberate writes that happen outside
the main interactive turn. Both commands still emit an `L1` session log entry
before writing `L2` or `L3` content.

### Utilities
```bash
list-themes
```

## Theme Selection

Theme selection is provided by the calling agent. The script no longer performs
keyword-based auto-detection internally.

- `persist-turn` requires `--theme` when `--extracted-content` is present
- `write-theme` requires `--theme`
- Passing `--theme auto` is rejected so the theme source stays outside the skill

| Signal | Pattern | Action |
|--------|---------|--------|
| User correction | "not right", "wrong", "it should be", "wrong", "incorrect" | → Error post-mortem |
| Explicit memory request | "remember", "do not forget", "remember", "preference", "constraint" | → Preference record |
| Error in response | "Error:", "Exception", "timeout" | → Error log |
| Emotional + complex | Satisfied/dissatisfied + 5+ turns | → Task summary |
| Tool-heavy milestone | 3+ tools, every 5 turns | → Progress snapshot |

## Agent Integration Guide

### For cortana.agent.md

Authoritative contract: `docs/specs/cortana-memory-contract.md`

**Contract mapping from the current Cortana prompt:**

Use the mapping in `docs/specs/cortana-memory-contract.md` as the single source of truth.

**Session Start (Automatic):**
```yaml
on_session_start:
  - call: memory-manager/session-init
  - inject_result_into: context
```

If the task is complex and the injected context is insufficient, explicitly read `memory/global.md` and follow:

```yaml
read_order:
  - Active Mission
  - Key Constraints
  - Research Index
  - User Preferences
```

Treat `Research Index` as a starting point only. If no existing note is a close fit, continue with fresh search and write the new material back to `memory/`.

**Every Turn (Automatic):**
```yaml
on_any_answered_turn:
  - call: memory-manager/persist-turn
    with: {raw_content, extracted_content?, theme?, tools_used?, session_id?}

if_extracted_content_is_not_ready_yet:
  - call: memory-manager/log-turn
    with: {raw_content, tools_used?, session_id?}
  - later_same_turn_when_value_is_clear:
    call: memory-manager/write-theme
    with: {content, theme, promote_global?}
```

**Explicit User Request:**
```yaml
when_user_says: ["remember", "do not forget", "remember"]
  - call: memory-manager/persist-turn
    with: {raw_content, extracted_content, theme: preferences, promote_global: true}
```

Notes:
- `persist-turn` is the default interactive command because it guarantees `L1` first.
- `write-theme --promote-global` is the manual path for durable extracted notes outside the main turn flow.
- Do not require full reuse or full distillation before persisting useful content.

**Global Updates (Selective):**
```yaml
when_new_long_term_preference_or_active_mission_change:
  - call: memory-manager/write-global
    with: {content, append: true}
```

**Verified route / POI / commute tasks:**
```yaml
when_route_or_location_answer_was_tool_verified:
  - follow: docs/specs/cortana-memory-contract.md
  - preferred_call: memory-manager/persist-turn
    with: {raw_content, extracted_content, theme: travel}
```

## File Structure

```
memory/
├── global.md                    # L3: Long-term memory
├── data/
│   ├── market_snapshot.csv      # CSV memory data
│   └── manifest.json            # CSV data manifest
├── sessions/
│   ├── 2026-02-27.md           # L1: Today's raw logs
│   └── 2026-02-26.md           # L1: Yesterday's logs
├── coding/
│   └── 2026-02-27_14.md        # L2: Coding theme
├── architecture/
│   └── 2026-02-27_15.md        # L2: Architecture theme
└── research/
    └── 2026-02-27_18.md        # L2: Research theme
```

## Migration from Legacy CLI

**Breaking Changes:**
- Removed the legacy one-off capture, quality-gating, session-finalization, search, and cleanup command families

**Current Standard:**
- `persist-turn` is the default interactive command
- `write-theme` is the manual extracted-note command
- `write-global` is reserved for durable global updates
- `session-init` only loads context and optionally logs the initialization event

**Migration:**
```bash
# Normal turn or milestone
python3 skills/memory-manager/scripts/memory_manager.py persist-turn \
  --raw-content "..." \
  --extracted-content "..." \
  --theme research

# Manual extracted-note write
python3 skills/memory-manager/scripts/memory_manager.py write-theme \
  --theme research \
  --content "..."

# CSV data write
python3 skills/memory-manager/scripts/memory_manager.py write-data \
  --name metrics.csv \
  --csv-content "date,value\n2026-03-07,1" \
  --description "Sample metric series"

# CSV data read
python3 skills/memory-manager/scripts/memory_manager.py read-data \
  --name metrics.csv \
  --head 5
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
