# Global Memory

> Read Order: Active Mission -> Key Constraints -> Research Index -> User Preferences
> Source of Truth: `memory/`
> Last Reset: 2026-03-07

## 1. Active Mission

**Status**: None
**Goal**: None
**Current Phase**: None
**Last Updated**: 2026-03-07

### Mission Snapshot

- No active mission recorded.

### Key Constraints (Immutable)

- Memory for this repository must remain in `./memory/`.
- Every interaction must write `L1` raw content to `memory/sessions/...`.
- Valuable extracted content may then be written to `L2` theme files.
- `L2` does not replace `L1`.

### Next Useful Reads

- `memory/preferences/user_preferences.md`
- `memory/README.md`

## 2. Research Index

- No active research memory recorded.

## 3. Decisions

### Decision: Memory Organization

- **Rule**: Keep the memory tree minimal, clean, and repo-local.
- **Naming**: Use consistent lowercase snake_case or date-first naming.
- **Classification**: Avoid `misc`; place files only in `preferences`, `projects`, `research`, or `sessions`.

## 4. User Preferences

### Durable Preferences

- Prefer repo-local memory over external memory mounts.
- Keep memory structure clean and standardized.

### Task-Scoped Preferences

- None recorded.
