# Frontend Quality Gates

## Static Analysis and Build Gates

Before implementation and before delivery, ensure:

- Type safety: `tsc --noEmit` (or framework equivalent)
- Lint: ESLint with framework + a11y rules
- Format: Prettier check
- Tests: unit/integration tests for critical behavior
- Build: production build succeeds

## Auto-Configuration Policy

In Phase 1, if quality gates are missing:

- Add missing scripts in `package.json`
- Add minimal config files (`tsconfig`, ESLint, Prettier) using local standard docs
- Prefer incremental, non-breaking setup
- Report what was added and why

## Non-Negotiable Constraints

- Do not use broad lint disables to hide design issues
- Do not use `any` where narrow types are feasible
- Do not skip loading/empty/error/success states
- Do not break keyboard/focus behavior for visual polish
- Do not leak secrets or internal-only values into client bundles

## Web Performance Budget (Default Baseline)

Use design-doc budgets first. If missing, apply this default baseline:

### Core Web Vitals (p75, mobile)

- LCP <= 2.5s
- INP <= 200ms
- CLS <= 0.10

### Route-level JS Budget

- Critical initial JS per major route <= 170KB gzip (baseline target)

### Lab Guardrails

- Avoid long tasks > 50ms on critical interaction path where feasible
- Keep hydration/initial interaction responsive on median mobile hardware

If a feature exceeds budget, document tradeoffs and mitigation plan (split, defer, preload strategy, caching) before merge.

---

## Reference

See [frontend-checklists.md](frontend-checklists.md) for detailed implementation checklists.
