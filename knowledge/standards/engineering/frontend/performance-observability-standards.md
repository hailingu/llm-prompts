# Frontend Performance and Observability Standards

**Version**: 1.0  
**Last Updated**: 2026-03-13

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

## 2. Rendering Performance

- Prefer server rendering/streaming for content-heavy first paint when stack supports it.
- Avoid unnecessary re-renders via stable props and memoization where measured.
- Use list virtualization for large datasets.
- Defer non-critical UI and third-party scripts.

---

## 3. Network Performance

- Collapse avoidable sequential fetches into parallel requests.
- Use prefetch/preload only for high-probability next actions.
- Avoid over-fetching; request only required fields when API supports it.
- Prefer caching strategy aligned with data volatility.

---

## 4. Bundle and Asset Governance

- Split code by route and heavy feature boundaries.
- Keep design-system/shared dependencies deduplicated.
- Optimize images (responsive sizes, modern formats, lazy loading).
- Block accidental inclusion of large debug/dev-only dependencies.

---

## 5. Telemetry Baseline

Track at least:

- Page/screen load events
- Core user journey start/success/failure
- API failure categories with correlation IDs
- Frontend runtime errors with release/build metadata
- Core Web Vitals and route-level timings

Event requirements:

- Stable event names and versioning
- Consistent context fields (route, device class, locale, release)
- Privacy-safe payloads (no sensitive data)

---

## 6. Alerting and SLO Signals

Define alerts for:

- Error rate spikes on critical flows
- Sustained vitals regression
- Increased user-abandonment on key funnels

For each alert:

- Include severity and owner
- Include runbook link
- Include rollback/kill-switch recommendation

---

## 7. Measurement Workflow

Before merge:

1. Compare bundle diff for impacted routes.
2. Validate key interactions in throttled network/CPU mode.
3. Verify no major vitals regressions in lab tools.

After deployment:

1. Observe telemetry watch window.
2. Confirm no sustained regressions.
3. Roll back or flag-off quickly if thresholds breach.

---

## 8. Review Checklist

- [ ] Performance budgets are defined and evaluated.
- [ ] Bundle impact is measured and justified.
- [ ] Critical flows have instrumentation coverage.
- [ ] Error telemetry includes actionable context.
- [ ] Alert ownership and response path are clear.
