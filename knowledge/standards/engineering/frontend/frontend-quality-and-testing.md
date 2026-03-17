# Frontend Quality and Testing

**Version**: 1.0  
**Last Updated**: 2026-03-13

---

## 1. Quality Gates (Static Analysis)

### Required Tooling

- Type checking: `typescript` (`tsc --noEmit`)
- Linting: `eslint`
- Formatting: `prettier`
- Testing: `vitest` or `jest`
- Accessibility linting: `eslint-plugin-jsx-a11y` (or framework equivalent)

### package.json Scripts (Minimum)

```json
{
  "scripts": {
    "typecheck": "tsc --noEmit",
    "lint": "eslint .",
    "format:check": "prettier --check .",
    "test": "vitest run"
  }
}
```

If using Jest, replace `vitest run` with project-appropriate Jest command.

### TypeScript Baseline (`tsconfig.json`)

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ESNext",
    "strict": true,
    "noUncheckedIndexedAccess": true,
    "exactOptionalPropertyTypes": true,
    "noFallthroughCasesInSwitch": true,
    "moduleResolution": "Bundler",
    "jsx": "react-jsx",
    "skipLibCheck": true
  }
}
```

Adjust `jsx` and module options to framework/bundler conventions.

### CI Gate Recommendation

Run in this order:

1. `npm run typecheck`
2. `npm run lint`
3. `npm run format:check`
4. `npm test`
5. `npm run build` (if defined)

Fail fast on any non-zero exit code.

---

## 2. Testing Pyramid for Frontend

Use balanced layers:

1. **Unit tests**: pure functions, reducers, mappers, validators.
2. **Component/integration tests**: interactions and UI state transitions.
3. **E2E tests**: top critical journeys and production-like wiring.

Prefer broad coverage at unit/integration level, focused E2E for business-critical flows.

---

## 3. What Must Be Tested

- API normalization and parsing boundaries
- Error mapping and recovery behavior
- Loading/empty/error/success state transitions
- Form validation and submission flows
- Permissions/feature-flag behavior

---

## 4. Test Quality Rules

- Assert user-visible behavior, not implementation details.
- Avoid brittle selector coupling.
- Keep tests deterministic: control timers, randomness, and network responses.
- Keep fixtures realistic but minimal.

Integration patterns:

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

## 7. Agent Auto-Configuration Policy

When agent detects missing quality gates:

- Add missing scripts to `package.json`
- Add minimal config files (`tsconfig.json`, ESLint config, Prettier config) if absent
- Prefer non-breaking incremental setup
- Report all changes clearly to the user

---

## 8. Review Checklist

- [ ] Type, lint, test, and build gates are defined.
- [ ] Contract-sensitive logic has direct tests.
- [ ] Error/retry/stale-response paths are covered.
- [ ] Accessibility-relevant interactions are tested.
- [ ] Tests avoid internal implementation coupling.
- [ ] CI gates enforce reliable feedback.
