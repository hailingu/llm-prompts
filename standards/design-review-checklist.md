# Design Review Checklist (Based on Google Practices)

**Purpose**: Provide a systematic design review checklist to ensure design documents meet Google-style quality standards.

**When to use**: Use this checklist at the following stages:

- After the `java-architect` completes Level 1 (self-review)
- After the `java-api-designer` completes Level 2 (self-review)
- Before implementation by the `java-coder-specialist` (pre-implementation review)
- Before writing documentation by the `java-doc-writer` (documentation review)

**Version**: 1.0  
**Last Updated**: 2026-01-24

---

## Level 1: Architecture Design Review (java-architect)

### 1. Context and Scope

- [ ] Problem statement is clear and specific
- [ ] Target users are identified (internal services / external API / CLI)
- [ ] System boundary is defined (which external systems to interact with)
- [ ] Out-of-scope items are explicitly listed

**Questions to ask**:

- Can a new team member understand WHY this module is needed?
- Is it clear WHO will use this module?
- Are the integration points well-defined?

---

### 2. Goals and Non-Goals

- [ ] Goals are measurable and specific (avoid vague words like "fast", "scalable")
- [ ] Non-Goals explicitly state what's NOT included
- [ ] Success criteria are defined (e.g., "support 100 QPS with p95 < 200ms")

**Anti-patterns**:

- ❌ "Improve performance" → ✅ "Reduce p95 latency from 500ms to 200ms"
- ❌ "Support high concurrency" → ✅ "Support 100 concurrent requests (QPS)"

---

### 3. Design Overview

- [ ] Component Diagram exists and is readable
- [ ] Each component's responsibility is clearly described
- [ ] Component interactions are defined (REST/gRPC/Database/MQ)
- [ ] Technology stack is specified (Spring Boot version, database type, etc.)

**Component Diagram Quality**:

- [ ] Uses standard notation (boxes for services, arrows for dependencies)
- [ ] Shows external dependencies (databases, external APIs)
- [ ] Is simple enough to fit on one slide

---

### 4. API Design Guidelines

- [ ] **Error Handling Strategy is defined** (Business failure vs Infrastructure failure)
- [ ] **API Versioning Strategy is defined** (URL versioning, package versioning)
- [ ] **Authentication/Authorization Strategy is defined** (API key, OAuth, JWT)
- [ ] **API Overview is provided** (method names and purposes, NOT full signatures)

**Example of good API Overview**:

```markdown
### 4.4 API Overview
- `verify(apiKey)`: Verify subscription status
- `startPeriodicCheck(interval)`: Start periodic verification
- `stopPeriodicCheck()`: Stop periodic verification
```

**What NOT to include** (belongs to Level 2):

- ❌ Complete method signatures with parameter types
- ❌ Exception declarations
- ❌ @ThreadSafe annotations

---

### 5. Data Model Overview

- [ ] Key entities are listed (User, Subscription, Order, etc.)
- [ ] Entity relationships are described (User has many Subscriptions)
- [ ] **NOT detailed field definitions** (belongs to Level 2)

---

### 6. Concurrency Requirements Overview

- [ ] Performance targets are defined (QPS, response time)
- [ ] Thread safety strategy is stated (which components need to be thread-safe)
- [ ] **NOT method-level concurrency contracts** (belongs to Level 2)

**Example**:

```markdown
### 6.1 Performance Targets
- QPS: 100 concurrent requests
- Response Time: p95 < 200ms, p99 < 500ms

### 6.2 Thread Safety Strategy
- SubscriptionVerifier: MUST be thread-safe (called from request handler threads)
- ConfigProvider: Thread-safe (immutable after initialization)
- PeriodicChecker: NOT thread-safe (single-threaded scheduler)
```

---

### 7. Cross-Cutting Concerns

- [ ] Observability: Logging and monitoring strategy is defined
- [ ] Security: Threat model and mitigation are described
- [ ] Reliability: Error handling and retry strategy are outlined

---

### 8. Alternatives Considered

- [ ] At least 2 alternative approaches are listed
- [ ] Each alternative has Pros/Cons/Decision
- [ ] Trade-offs are clearly explained

**Anti-pattern**:

- ❌ "We considered Option B but chose Option A because it's better"
- ✅ "Option A: Lower latency (50ms) but higher memory (2GB). Option B: Higher latency (100ms) but lower memory (500MB). Decision: Choose A because latency is more critical than memory for our use case."

---

## Level 2: API Specification Review (java-api-designer)

### 9. API Interface Definition (Section 10.1)

- [ ] All interfaces are complete Java code (compilable)
- [ ] Every public method has Javadoc with @param, @return, @throws
- [ ] **@ThreadSafe annotation is present** (Yes/No with justification)
- [ ] **@Idempotent annotation is present** (if applicable)
- [ ] Return types follow Pattern 1.1 or 1.2 (null vs Optional)
- [ ] Exceptions follow Pattern 1.1 (IOException for infrastructure, IllegalArgumentException for caller bug)

**Checklist for each method**:

```java
/**
 * [Method description]
 * 
 * @param [paramName] [param description]
 * @return [return description, specify null case]
 * @throws [ExceptionType] [when this exception is thrown]
 * @ThreadSafe [Yes/No + justification]
 * @Idempotent [Yes/No]
 */
```

---

### 10. Design Rationale (Section 10.2) ⭐⭐⭐ MOST CRITICAL

**10.1 Contract Precision**:

- [ ] **Contract is in table format** (for HTTP APIs)
- [ ] **All HTTP status codes are defined** (200/401/404/500/503)
- [ ] **All exception types are specific** (SocketTimeoutException vs IOException)
- [ ] **All edge cases are covered** (null input, empty string, invalid format)
- [ ] **Retry strategy is specified** (Yes/No for each scenario)
- [ ] **Pattern reference is included** (Pattern 1.1, Pattern 3.1, etc.)

**Example of good Contract**:

| Scenario      | HTTP Status | Response Body         | Return Value | Exception                           | Retry?   | Pattern |
| ------------- | ----------- | --------------------- | ------------ | ----------------------------------- | -------- | ------- |
| ------------- | ----------- | --------------        | -----------  | -----------                         | -------- | ------- |
| Success       | 200         | {"status":"active"}   | Subscription | -                                   | No       | -       |
| Not found     | 404         | {"error":"not_found"} | null         | -                                   | No       | 1.1     |
| Timeout       | -           | -                     | -            | IOException(SocketTimeoutException) | Yes      | 2.1     |
| Null param    | -           | -                     | -            | IllegalArgumentException            | No       | 1.1     |

**Anti-patterns**:

- ❌ "Returns null or throws exception depending on the situation"
- ❌ "Throws IOException when network fails"
- ✅ "Throws IOException(SocketTimeoutException) when connection times out (HTTP client configured with 5s timeout)"

---

**10.2 Caller Guidance Completeness**:

- [ ] **Contains 50-100 lines of executable code** (NOT pseudo-code)
- [ ] **Includes input validation** with HTTP status codes
- [ ] **Includes retry logic** with specific parameters (maxRetries=3, initialDelay=1000ms, backoffFactor=2.0)
- [ ] **Includes logging statements** with log levels (logger.warn, logger.error, logger.info)
- [ ] **Includes metrics reporting** (metrics.incrementCounter)
- [ ] **Includes HTTP status code mapping** for each scenario (return Response.status(401) for invalid subscription)
- [ ] **Includes error messages** (user-facing messages)
- [ ] **Pattern references are included** (// Pattern 2.1: Exponential Backoff)

**Quality Test**: Can a junior developer copy-paste this code into production?

- ✅ Yes: Code is complete, runnable, and handles all errors
- ❌ No: Code is incomplete, has TODOs, or missing error handling

**Example** (see `.github/standards/api-patterns.md` Pattern 7 for complete example)

---

**10.3 Rationale Clarity**:

- [ ] Explains WHY this design was chosen
- [ ] Refers to architectural constraints (performance, simplicity, maintainability)
- [ ] Justifies trade-offs (e.g., "Chose null over Optional for performance in hot path")

---

**10.4 Alternatives Considered**:

- [ ] At least 1 alternative per key decision
- [ ] Each alternative has Pros/Cons
- [ ] Rejection reason is specific (not just "not suitable")

---

### 11. Dependency Interfaces (Section 10.3)

- [ ] All external dependencies are defined as Java interfaces
- [ ] Dependency interfaces follow same standards as main API (Javadoc, @ThreadSafe, etc.)
- [ ] Dependency injection strategy is clear (constructor injection recommended)

---

### 12. Data Model (Section 11)

- [ ] All entities have complete field definitions
- [ ] Field types, constraints, and nullability are specified
- [ ] Immutability is indicated (@Immutable annotation or final fields)
- [ ] Validation rules are documented (e.g., "apiKey: non-null, 32-64 chars")

---

### 13. Concurrency Requirements (Section 12)

- [ ] Every public method has thread safety contract
- [ ] Performance requirements are specific (QPS, response time target)
- [ ] Synchronization strategy is stated (synchronized vs ConcurrentHashMap vs immutable)

**Example**:

```markdown
### 12. Concurrency Requirements

| Method               | Thread-Safe?   | Expected QPS    | Response Time   | Synchronization Strategy       |
| -------------------- | -------------- | --------------- | --------------- | ------------------------------ |
| --------             | -------------- | --------------  | --------------- | -------------------------      |
| verify()             | Yes            | 100             | p95 < 200ms     | Stateless (no sync needed)     |
| startPeriodicCheck() | No             | 1 (called once) | N/A             | Single-threaded caller assumed |
```

---

## Pre-Implementation Review (java-coder-specialist)

Before starting implementation, verify:

### 14. Contract Implementability

- [ ] All scenarios in Contract table can be detected in code (e.g., distinguish HTTP 404 vs 500)
- [ ] All exceptions in Contract can be caught and mapped
- [ ] All retry scenarios are feasible (e.g., can distinguish SocketTimeoutException from general IOException)
- [ ] No conflicting requirements (e.g., "must be stateless" vs "must cache results")

**Test**: For each row in Contract table, can you write code to handle it?

---

### 15. Caller Guidance Executability

- [ ] Retry parameters are realistic (maxRetries=3, not 1000)
- [ ] HTTP status codes are RESTful standard (200/400/401/404/500/503)
- [ ] Logging levels match severity (ERROR for system failure, WARN for retry, INFO for business events)
- [ ] Metrics names follow conventions (dot-separated, lowercase)

---

### 16. Dependency Availability

- [ ] All dependency interfaces can be implemented (no impossible contracts)
- [ ] Required libraries/frameworks are compatible (Spring Boot version, Java version)
- [ ] External services are accessible (database connection, external API endpoint)

---

## Documentation Review (java-doc-writer)

Before generating user documentation:

### 17. Caller Guidance Quality

- [ ] Code examples are copy-pasteable (complete, no placeholders)
- [ ] Error handling covers all exceptions from Contract
- [ ] HTTP status codes are consistent with Contract
- [ ] User-facing error messages are clear and actionable

---

### 18. Documentation Completeness

- [ ] Every public method has usage example
- [ ] Common use cases are documented (based on Goals section)
- [ ] Troubleshooting section covers all expected failures
- [ ] Performance guidelines match Concurrency Requirements

---

## Summary: Critical Checks

**Top 5 Most Important Checks** (if you do nothing else, do these):

1. ✅ **Contract is in table format with all HTTP status codes** (Section 10.2)
2. ✅ **Caller Guidance includes 50-100 lines of executable code** (Section 10.2)
3. ✅ **All edge cases are covered** (null/empty/invalid input in Contract table)
4. ✅ **Error handling follows Pattern 1.1** (Business vs Infrastructure failure)
5. ✅ **Every public method has @ThreadSafe annotation** (Section 10.1)

---

## Usage Example

### For java-architect (after completing Level 1)

```bash
# Self-review checklist
- [ ] Section 1-2: Context and Goals are clear
- [ ] Section 3: Component Diagram is readable
- [ ] Section 4: API Design Guidelines define error handling strategy
- [ ] Section 4.4: API Overview lists method names (NOT full signatures)
- [ ] Section 5-6: Overview sections provide high-level guidance
- [ ] Section 10: Alternatives are listed with trade-offs

# If all pass → handoff to @java-api-designer
# If any fail → revise before handoff
```

---

### For java-api-designer (after completing Level 2)

```bash
# Self-review checklist (CRITICAL)
- [ ] Section 10.1: All methods have @ThreadSafe annotation
- [ ] Section 10.2 Contract: Table format with all status codes
- [ ] Section 10.2 Caller Guidance: 50-100 lines of executable code
- [ ] Section 10.2 Caller Guidance: Includes logging, metrics, retry logic
- [ ] Section 11: Data Model has complete field definitions
- [ ] Section 12: Concurrency Requirements specify thread safety per method

# If all pass → handoff to @java-coder-specialist and @java-doc-writer
# If any fail → revise Section 10.2 (most common issue)
```

---

### For java-coder-specialist (before implementation)

```bash
# Pre-implementation review
- [ ] Contract Implementability: Can map all HTTP status codes to exceptions
- [ ] Caller Guidance: Can copy-paste and run (no TODOs or placeholders)
- [ ] Dependency Interfaces: All dependencies are defined
- [ ] No conflicting requirements found

# If all pass → start implementation
# If any fail → handoff back to @java-api-designer with specific issues
```

---

## References

- [Google Engineering Practices: Design Docs](https://google.github.io/eng-practices/design-docs/)
- [Design Docs at Google (Article)](https://www.industrialempathy.com/posts/design-docs-at-google/)
- [Google SRE Book - Monitoring](https://sre.google/sre-book/monitoring-distributed-systems/)
