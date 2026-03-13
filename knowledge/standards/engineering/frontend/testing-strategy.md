# Frontend Testing Strategy

**Version**: 1.0  
**Last Updated**: 2026-03-13

---

## 1. Testing Pyramid for Frontend

Use balanced layers:

1. **Unit tests**: pure functions, state reducers, mappers, validators.
2. **Component/integration tests**: real interactions and UI state transitions.
3. **E2E tests**: top critical journeys and production-like wiring.

Prefer broad coverage at unit/integration level, focused E2E for business-critical flows.

---

## 2. What Must Be Tested

- API normalization and parsing boundaries
- Error mapping and recovery behavior
- Loading/empty/error/success state transitions
- Form validation and submission flows
- Permissions/feature-flag behavior

---

## 3. Test Quality Rules

- Assert user-visible behavior, not implementation details.
- Avoid brittle selector coupling.
- Keep tests deterministic: control timers, random values, network responses.
- Keep fixtures realistic but minimal.

---

## 4. Integration Test Patterns

- Mock network at boundary (MSW or equivalent).
- Verify retries, cancellation, and stale response handling.
- Verify optimistic update rollback behavior when mutation fails.
- Verify accessibility-critical interactions (keyboard submit, focus movement).

---

## 5. E2E Scope Rules

E2E should cover:

- Login/session continuity
- Primary revenue or retention flow
- Critical write operations (create/update/delete)
- One unhappy-path recovery per critical flow

Do not push all combinations into E2E; keep E2E lean and high-signal.

---

## 6. Coverage Guidance

- Use coverage as guardrail, not objective.
- Prioritize risk-weighted coverage over uniform coverage.
- Critical domain paths should have strong branch coverage.

---

## 7. CI Testing Gates

Recommended CI order:

1. Type check
2. Lint
3. Unit/integration tests
4. Build
5. E2E smoke (or nightly full E2E)

Fail fast on deterministic failures.

---

## 8. Review Checklist

- [ ] Contract-sensitive logic has direct tests.
- [ ] Error and retry flows are covered.
- [ ] Accessibility-relevant interactions are tested.
- [ ] Tests avoid internal implementation coupling.
- [ ] CI gates enforce reliable feedback.
