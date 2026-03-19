# Frontend Escalation

## ESCALATION AND ITERATION RULES

- Maximum 3 feedback iterations with @frontend-api-designer/@frontend-code-reviewer
- Escalate to @frontend-tech-lead for unresolved risk or requirement conflicts
- Escalate to @frontend-architect for architectural contradictions
- Never continue coding against unresolved contract contradictions

---

## RELEASE GUARDRAILS AND ROLLBACK STRATEGY

### Release Guardrails

- Ship risky changes behind feature flags when possible
- Prefer canary/phased rollout for high-impact UI paths
- Define monitoring watch window after deployment
- Predefine rollback trigger metrics for errors, vitals regressions, or conversion drops

### Rollback Triggers (Default)

Trigger rollback or immediate kill-switch when any is true:

- Sustained increase in client error rate on changed flows
- Sustained Core Web Vitals regression beyond agreed budget
- Critical accessibility regression in primary user path
- Severe functional regression affecting task completion

### Rollback Execution Principles

- Keep rollback path simple: flag off first, redeploy second
- Ensure data/schema compatibility for safe rollback before release
- After rollback, provide incident summary with root cause and prevention actions

---

## DEFINITION OF DONE

A task is done only when all are true:

- API + UX contracts are implemented and traceable
- Accessibility baseline requirements are satisfied
- Performance and reliability constraints are met or deviations are documented
- Web performance budgets are met or formally waived with mitigation plan
- Type/lint/test/build checks pass
- Critical failure states are tested
- Release guardrails and rollback path are defined for risky changes
- Delivery summary is reviewer-ready
