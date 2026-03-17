# Frontend NFR Standards

**Version**: 1.0  
**Last Updated**: 2026-03-13

This document covers non-functional requirements: performance, accessibility, security, observability, and release reliability.

---

## 1. Performance Budget Policy

If product design doc does not specify budgets, use defaults:

- **Core Web Vitals (p75, mobile)**:
  - LCP <= 2.5s
  - INP <= 200ms
  - CLS <= 0.10
- **Critical route JS budget**: <= 170KB gzip per major route
- **Long task guardrail**: avoid >50ms tasks on primary interactions where feasible

Budget governance:

- Any budget regression must include mitigation plan.
- Budget waivers require explicit owner and sunset date.

---

## 2. Rendering, Network, and Bundle Performance

Rendering:

- Prefer server rendering/streaming for content-heavy first paint when stack supports it.
- Avoid unnecessary re-renders via stable props and memoization where measured.
- Use list virtualization for large datasets.
- Defer non-critical UI and third-party scripts.

Network:

- Collapse avoidable sequential fetches into parallel requests.
- Use prefetch/preload only for high-probability next actions.
- Avoid over-fetching; request only required fields when API supports it.
- Prefer caching strategy aligned with data volatility.

Bundle and assets:

- Split code by route and heavy feature boundaries.
- Keep shared dependencies deduplicated.
- Optimize images (responsive sizes, modern formats, lazy loading).
- Block accidental inclusion of large debug/dev-only dependencies.

---

## 3. Accessibility Standards

Reference baseline: WCAG 2.2 AA and WAI-ARIA Authoring Practices.

Semantic and keyboard:

- Use native semantic elements before ARIA roles.
- Headings preserve logical hierarchy.
- All interactive actions are keyboard-operable.
- Visible focus indicator must not be removed.

Forms and dynamic content:

- Each input has associated label.
- Errors are programmatically connected (`aria-describedby`).
- Dialogs/trays trap focus and restore focus on close.
- Loading/status updates use appropriate live regions.

Visual accessibility:

- Contrast meets WCAG AA.
- Content remains usable at 200% zoom.
- Motion respects `prefers-reduced-motion`.
- Information must not rely only on color.

---

## 4. Security and Privacy Baseline

- Treat all external inputs as untrusted.
- Avoid unsafe HTML injection; sanitize when unavoidable.
- Keep auth/session handling aligned with backend security architecture.
- Never expose secrets in client code, source maps, or telemetry.

Dependency and privacy hygiene:

- Keep dependency set minimal and maintained.
- Track high/critical vulnerabilities and remediate quickly.
- Collect only necessary analytics fields.
- Redact PII or sensitive values in logs/events.

---

## 5. Observability and Alerting

Telemetry baseline:

- Page/screen load events
- Core journey start/success/failure
- API failure categories with correlation IDs
- Frontend runtime errors with release/build metadata
- Core Web Vitals and route-level timings

Alerting baseline:

- Error rate spikes on critical flows
- Sustained vitals regression
- Increased abandonment on key funnels

Each alert should include severity, owner, runbook, and rollback recommendation.

---

## 6. Release Guardrails and Rollback

Release guardrails:

- Use feature flags for risky changes.
- Prefer canary/phased rollout for high-impact paths.
- Define monitoring watch window for each release.
- Predefine rollback triggers before deployment.

Rollback triggers:

- Sustained error spike on changed critical flow
- Sustained Core Web Vitals regression beyond budget
- Critical accessibility regression
- Severe functional regression affecting completion rate

Execution priorities:

1. Disable risky flag/entry point first
2. Roll back deployment if issue persists
3. Publish incident summary and prevention actions

---

## 7. Measurement Workflow

Before merge:

1. Compare bundle diff for impacted routes.
2. Validate key interactions in throttled network/CPU mode.
3. Verify no major vitals regressions in lab tools.
4. Run keyboard-only and screen-reader spot checks.

After deployment:

1. Observe telemetry watch window.
2. Confirm no sustained regressions.
3. Roll back or flag-off quickly if thresholds breach.

---

## 8. Review Checklist

- [ ] Performance budgets are defined and evaluated.
- [ ] Accessibility baseline is validated.
- [ ] Security-sensitive boundaries are protected.
- [ ] Telemetry is actionable and privacy-safe.
- [ ] Release guardrails and rollback path are explicit.
