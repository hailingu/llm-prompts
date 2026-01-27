---
agent: "agent"
model: "Raptor mini (Preview)"
description: "You are an AI agent responsible for analyzing this repository’s architecture and writing clear architecture documentation into the docs/ folder. Output language: English."
---

Analyze the current repository’s architecture and write the results as Markdown documentation under `docs/`.

Instructions:

Workflow (use this single workflow end-to-end):

1) Establish scope and requirements (do not skip)
- Your documentation must clearly explain: project structure, boundaries, responsibilities, dependencies, and interactions.
- Minimum dimensions you MUST cover:
  - Overall architecture diagram
  - What each component implements (capabilities / responsibilities)
  - Each component’s scope (what it covers / does not cover)
  - Each component’s dependencies (internal + external)
  - Interaction relationships between components (calls, events, data flow)
- Output language: English.

2) Scan the repository and gather facts
- Inspect the repo layout and key files (e.g., `README.md`, build files, dependency manifests, configs, entrypoints, scripts).
- Identify: main language(s), frameworks, module boundaries, directory conventions, and (if visible) deployment/runtime shape.
- Define “component” for this repo as any unit with a clear responsibility boundary (service, module/package, library, CLI, worker, web frontend, data-access layer, adapter/integration layer).

3) Identify and model components
- Enumerate components (do not invent). For each component, capture:
  - Name (stable, unambiguous)
  - Location (relative path, entrypoint, key files)
  - Responsibilities (what it does)
  - Scope (what it explicitly does NOT do)
  - Dependencies
    - Internal dependencies (which repo components it depends on)
    - External dependencies (third-party packages/services)
    - Configuration dependencies (config files, environment variables)
  - Exposed interfaces (as applicable): CLI commands, HTTP APIs, library APIs, message topics, file I/O, etc.

4) Analyze interactions and data flow
- Describe component interactions: direction, trigger, sync/async, and the main payload/data shape (high-level).
- If storage/external systems exist, document read/write paths and key data flows.
- If failure handling is visible, document retries/timeouts/fallbacks/idempotency.
- If something is unclear, label it explicitly as “Needs confirmation” and list concrete questions.

5) Write the required documentation into docs/
- If `docs/` does not exist, create it.
- Create/update the following files (all are required):

  A) `docs/architecture.md`
  - Project overview: goals, primary capabilities, and core boundaries.
  - Overall architecture diagram: at least 1 renderable Mermaid diagram.
    - Prefer a C4-style diagram (System/Container/Component) if it fits; otherwise use the most accurate diagram style for this repo.
  - Key architectural decisions (only what can be inferred from repo content): layering, boundaries, major technology choices, rationale.

  B) `docs/components.md`
  - A component inventory table (recommended columns: name, type, path, responsibility, upstream/downstream dependencies, notes).
  - Per-component sections: responsibilities, scope, dependencies, and key implementation points (architecture-relevant only).

  C) `docs/interactions.md`
  - Component interactions: include at least 1 renderable Mermaid diagram (sequenceDiagram or flowchart).
  - Describe at least 1 end-to-end “key path / typical use case” from entrypoint to core processing to outputs.

  D) `docs/dependencies.md`
  - Internal dependency view (between components/modules).
  - External dependency list (major third-party libraries/services/runtime dependencies).
  - For important dependencies, note purpose and any risk points (tight coupling, single point of failure, replaceability).

6) Add only the necessary extra context (keep it minimal but sufficient)
- Add details that materially improve understanding and maintainability, when present in the repo:
  - Development & runtime: how to run, key scripts, key configuration and environment variables
  - Configuration model: sources, precedence, defaults (if visible)
  - Observability: logs/metrics/tracing (if present)
  - Security boundaries: authn/authz, sensitive data flows, credential handling (structure only; never output secrets)
  - Extensibility: how to add a component/module/command, common change paths
  - Reliability/performance: caching, queues, concurrency model, retries/timeouts (if present)

7) Self-check before finishing
- Diagrams are Mermaid and renderable.
- Only relative paths are used (no local absolute paths).
- No non-existent components/interfaces were invented.
- `docs/architecture.md`, `docs/components.md`, `docs/interactions.md`, `docs/dependencies.md` exist.
- The minimum dimensions are fully covered.
