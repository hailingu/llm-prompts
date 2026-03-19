# Frontend Workflow

## Workflow Overview

### Phase 0: Understand the Problem and Contracts

Before coding, read design docs and extract:

- API contract (models, errors, pagination, retry semantics)
- UX contract (state machine, acceptance criteria, edge states)
- A11y contract (keyboard/focus/announcements/labels)
- Non-functional constraints (performance, security, telemetry)

**Contract Implementability Checklist**:

- [ ] Request/response schema complete and typed
- [ ] Error categories mapped to explicit UI states
- [ ] Loading/empty/error/success states all specified
- [ ] Focus/keyboard behavior is explicit
- [ ] Validation strategy and user feedback are explicit
- [ ] Retry/cancel/timeout behavior is explicit
- [ ] Telemetry events and required dimensions are defined
- [ ] Performance constraints are measurable
- [ ] No contradictory requirements

If checklist fails, handoff to @frontend-api-designer or @frontend-architect.

### Phase 1: Recon and Setup

- Identify stack from `package.json`
- Identify rendering mode (CSR/SSR/hybrid)
- Identify data layer and caching strategy
- Ensure quality gates exist or add minimal setup

**Setup Checklist**:

- [ ] `typecheck` script
- [ ] `lint` script
- [ ] `test` script
- [ ] `build` script
- [ ] strict TypeScript mode
- [ ] ESLint includes a11y rules

### Phase 2: Design the Implementation Slice

Break work into small vertical slices with clear value:

1. Boundary types and API normalization
2. State machine and side-effect orchestration
3. Semantic UI structure
4. Interaction behavior and form validation
5. Error recovery and retry paths
6. Telemetry hooks
7. Tests and hardening

**Design Rules**:

- Keep state local unless sharing is required
- Keep side effects at edges
- Prefer explicit state transitions over implicit booleans
- Prefer deterministic rendering over clever abstractions

### Phase 3: Implement with Reliability Patterns

**Mandatory implementation qualities**:

- Deterministic state transitions
- Explicit stale/cancel handling for async flows
- Strong typing at boundaries
- Accessible interactive controls
- Error boundaries and fallback UI where applicable

See [frontend-patterns.md](frontend-patterns.md) for reference code patterns.

### Phase 4: Validate Like Production

Run all relevant checks:

- `npm run typecheck`
- `npm run lint`
- `npm run test`
- `npm run build`

Add focused manual verification for:

- keyboard-only navigation
- screen reader labels/announcements
- slow network and retry behavior
- responsive layout at major breakpoints

Add performance verification for:

- Core Web Vitals trend (field data if available, otherwise lab proxy)
- bundle/report diff for impacted routes
- interaction responsiveness on critical flows

**Validation Matrix**:

| Area          | Validation Method             | Status |
|---------------|-------------------------------|--------|
| Types         | tsc --noEmit                  | PASS/FAIL |
| Lint          | eslint                        | PASS/FAIL |
| Tests         | vitest/jest                   | PASS/FAIL |
| Build         | framework build               | PASS/FAIL |
| Accessibility | keyboard + a11y lint          | PASS/FAIL |
| Performance   | budget and vitals checks      | PASS/FAIL |

### Phase 5: Delivery and Handoff

Provide concise and reviewable summary:

- what changed and why
- requirement-to-implementation mapping
- risk/tradeoff notes
- validation outcomes
- remaining follow-ups

If unresolved ambiguity remains, escalate instead of guessing.

---

## Reference

See [frontend-escalation.md](frontend-escalation.md) for escalation rules and definition of done.
