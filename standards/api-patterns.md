# Standard API Patterns (Based on Google AIP)

**Purpose**: 提供标准化的 API 设计模式，确保一致性和最佳实践。基于 Google API Improvement Proposals (AIP) 和 Java 业界标准。

**Version**: 1.0  
**Last Updated**: 2026-01-24

---

## Table of Contents

1. [Error Handling Patterns](#1-error-handling-patterns)
2. [Retry Strategy Patterns](#2-retry-strategy-patterns)
3. [HTTP Status Code Mapping](#3-http-status-code-mapping)
4. [Thread Safety Patterns](#4-thread-safety-patterns)
5. [Logging Patterns](#5-logging-patterns)
6. [Metrics Patterns](#6-metrics-patterns)

---

## 1. Error Handling Patterns

### Pattern 1.1: Business Failure vs Infrastructure Failure

**Google AIP Reference**: [AIP-193: Errors](https://google.aip.dev/193)

**Rule**:

- **Business failure** (expected scenario) → Return null or empty Optional
- **Infrastructure failure** (unexpected) → Throw checked exception
- **Programming error** (caller bug) → Throw unchecked exception

**Example Contract Table**:

| Scenario | Category | Return Value | Exception | HTTP Status |
| ------------- | ---------- | -------------- | ----------- | -------- |
| Resource not found | Business | `null` | - | 404 |
| Resource expired | Business | `null` | - | 200 (with status field) |
| Network timeout | Infrastructure | - | `IOException(SocketTimeoutException)` | 503 |
| Invalid parameter | Programming | - | `IllegalArgumentException` | 400 |

**Code Template**:

```java
/**
 * Retrieves resource by ID.
 * 
 * @return Resource object if found, null if not found/expired
 * @throws IOException if network/database failure occurs
 * @throws IllegalArgumentException if id is null
 */
public Resource getResource(String id) throws IOException {
    if (id == null) {
        throw new IllegalArgumentException("id must not be null");
    }
    
    try {
        // Infrastructure operation
        return repository.findById(id);  // null if not found
    } catch (SQLException | ConnectException e) {
        throw new IOException("Failed to access database", e);
    }
}
```

---

### Pattern 1.2: Optional vs Null

**Google Practice**: Use null for "not found" scenario to avoid Optional overhead in hot paths.

**When to use Optional**:

- ✅ Method returns value that is **frequently absent** (>20% of calls)
- ✅ Chaining operations (Optional.map, Optional.flatMap)
- ✅ Public API where null safety is critical

**When to use null**:

- ✅ Performance-critical code (avoid Optional allocation)
- ✅ Internal/private methods
- ✅ Database/repository layer (JDBC returns null)

**Example**:

```java
// Public API - use Optional for safety
public Optional<User> findUserByEmail(String email);

// Internal repository - use null for performance
User findById(Long id);  // null if not found
```

---

## 2. Retry Strategy Patterns

### Pattern 2.1: Exponential Backoff with Jitter

**Google SRE Reference**: [The SRE Book - Handling Overload](https://sre.google/sre-book/handling-overload/)

**Standard Parameters**:

```java
int maxRetries = 3;
int initialDelayMs = 1000;      // 1 second
int maxDelayMs = 10000;          // 10 seconds
double backoffFactor = 2.0;      // exponential
double jitter = 0.1;             // ±10% randomness
```

**Retry Sequence**:

- Attempt 1: immediate
- Attempt 2: 1s ± 100ms (900ms - 1100ms)
- Attempt 3: 2s ± 200ms (1800ms - 2200ms)
- Attempt 4: 4s ± 400ms (3600ms - 4400ms)

**Code Template**:

```java
private static final int MAX_RETRIES = 3;
private static final int INITIAL_DELAY_MS = 1000;
private static final double BACKOFF_FACTOR = 2.0;
private static final Random JITTER_RANDOM = new Random();

private <T> T retryable(Supplier<T> operation, String operationName) throws IOException {
    int delay = INITIAL_DELAY_MS;
    IOException lastException = null;
    
    for (int attempt = 0; attempt <= MAX_RETRIES; attempt++) {
        try {
            return operation.get();
        } catch (SocketTimeoutException | ConnectException e) {
            lastException = new IOException("Retryable failure", e);
            
            if (attempt == MAX_RETRIES) {
                logger.error("Failed {} after {} retries", operationName, MAX_RETRIES, e);
                throw lastException;
            }
            
            // Add jitter: ±10%
            int jitterMs = (int) (delay * 0.1 * (JITTER_RANDOM.nextDouble() - 0.5) * 2);
            int actualDelay = delay + jitterMs;
            
            logger.warn("Retry {} for {} after {}ms due to: {}", 
                attempt + 1, operationName, actualDelay, e.getMessage());
            
            Thread.sleep(actualDelay);
            delay = (int) (delay * BACKOFF_FACTOR);
        } catch (IOException e) {
            // Non-retryable error (e.g., 401 Unauthorized)
            throw e;
        }
    }
    
    throw lastException;  // unreachable but required by compiler
}
```

**Which Exceptions to Retry**:

- ✅ `SocketTimeoutException` - Network timeout
- ✅ `ConnectException` - Connection refused (server down)
- ✅ `UnknownHostException` - DNS failure (may be transient)
- ✅ HTTP 503 Service Unavailable
- ✅ HTTP 502 Bad Gateway
- ❌ `IllegalArgumentException` - Caller bug (won't fix itself)
- ❌ HTTP 401 Unauthorized - Invalid credentials
- ❌ HTTP 404 Not Found - Resource doesn't exist

---

## 3. HTTP Status Code Mapping

### Pattern 3.1: Standard Mapping Table

**Google AIP Reference**: [AIP-193: HTTP Status Codes](https://google.aip.dev/193)

| HTTP Status | Scenario | Return Value | Exception | Retry? |
| ------------- | ---------- | -------------- | ----------- | -------- |
| **2xx Success** | | | | |
| 200 OK | Successful operation | Object | - | No |
| 201 Created | Resource created | Object | - | No |
| 204 No Content | Successful deletion | void | - | No |
| **3xx Redirection** | | | | |
| 304 Not Modified | Resource unchanged | null | - | No |
| **4xx Client Errors** (DO NOT RETRY) | | | | |
| 400 Bad Request | Invalid input format | - | `IllegalArgumentException` | No |
| 401 Unauthorized | Invalid credentials | - | `SecurityException` | No |
| 403 Forbidden | Permission denied | - | `SecurityException` | No |
| 404 Not Found | Resource doesn't exist | null | - | No |
| 409 Conflict | Version conflict | - | `ConcurrentModificationException` | No |
| 429 Too Many Requests | Rate limit exceeded | - | `RateLimitException` | Yes (with backoff) |
| **5xx Server Errors** (RETRY) | | | | |
| 500 Internal Server Error | Server bug | - | `IOException` | Yes |
| 502 Bad Gateway | Proxy error | - | `IOException` | Yes |
| 503 Service Unavailable | Server overload/maintenance | - | `IOException` | Yes |
| 504 Gateway Timeout | Upstream timeout | - | `IOException` | Yes |

**Code Template**:

```java
public Response handleHttpResponse(int statusCode, String body) throws IOException {
    switch (statusCode) {
        case 200:
        case 201:
            return parseResponse(body);
            
        case 204:
            return null;
            
        case 400:
            throw new IllegalArgumentException("Invalid request: " + body);
            
        case 401:
        case 403:
            throw new SecurityException("Authentication/authorization failed: " + body);
            
        case 404:
            return null;  // Not found is expected
            
        case 429:
            throw new RateLimitException("Rate limit exceeded, retry later");
            
        case 500:
        case 502:
        case 503:
        case 504:
            throw new IOException("Server error: " + statusCode + " - " + body);
            
        default:
            throw new IOException("Unexpected status code: " + statusCode);
    }
}
```

---

## 4. Thread Safety Patterns

### Pattern 4.1: Immutable Objects (Recommended)

**Effective Java Item 17**: Minimize mutability

**Benefits**:

- ✅ Inherently thread-safe (no synchronization needed)
- ✅ Can be freely shared between threads
- ✅ Simple to reason about

**Template**:

```java
/**
 * Immutable value object.
 * @ThreadSafe Yes (immutable)
 */
public final class Subscription {
    private final String apiKey;
    private final Status status;
    private final Instant expiryDate;
    
    public Subscription(String apiKey, Status status, Instant expiryDate) {
        this.apiKey = Objects.requireNonNull(apiKey);
        this.status = Objects.requireNonNull(status);
        this.expiryDate = Objects.requireNonNull(expiryDate);
    }
    
    // Only getters, no setters
    public String getApiKey() { return apiKey; }
    public Status getStatus() { return status; }
    public Instant getExpiryDate() { return expiryDate; }
}
```

---

### Pattern 4.2: Synchronized Methods for Simple State

**When to use**: Low contention, simple state updates

**Template**:

```java
/**
 * Stateful component with synchronized access.
 * @ThreadSafe Yes (synchronized methods)
 */
public class Counter {
    private long count = 0;
    
    public synchronized void increment() {
        count++;
    }
    
    public synchronized long getCount() {
        return count;
    }
}
```

---

### Pattern 4.3: Concurrent Collections for High Throughput

**When to use**: High contention, frequent reads

**Template**:

```java
/**
 * Cache with concurrent access.
 * @ThreadSafe Yes (ConcurrentHashMap)
 */
public class SubscriptionCache {
    private final ConcurrentMap<String, Subscription> cache = new ConcurrentHashMap<>();
    
    public void put(String key, Subscription value) {
        cache.put(key, value);
    }
    
    public Subscription get(String key) {
        return cache.get(key);
    }
    
    public Subscription computeIfAbsent(String key, Function<String, Subscription> loader) {
        return cache.computeIfAbsent(key, loader);
    }
}
```

---

## 5. Logging Patterns

### Pattern 5.1: Log Levels (Google Style)

**Reference**: [Google Java Style Guide - Logging](https://google.github.io/styleguide/javaguide.html)

| Level | When to Use | Example |
| ------------- | ----------- | ------- |
| **ERROR** | System failure, requires immediate attention | Database connection failure, external API down |
| **WARN** | Unexpected but handled situation | Retry attempt, deprecated API usage, rate limit approached |
| **INFO** | Business-level events | User login, order created, subscription expired |
| **DEBUG** | Detailed diagnostic info (disabled in production) | Method entry/exit, variable values |
| **TRACE** | Very detailed flow (disabled in production) | Loop iterations, conditional branches |

**Code Template**:

```java
private static final Logger logger = LoggerFactory.getLogger(SubscriptionService.class);

public Response verifySubscription(String apiKey) {
    logger.debug("Verifying subscription for apiKey: ***");  // Mask sensitive data
    
    if (apiKey == null) {
        logger.warn("Received null apiKey from caller");
        return Response.status(400).build();
    }
    
    try {
        Subscription sub = verifier.verify(apiKey);
        if (sub == null) {
            logger.info("Invalid subscription for apiKey: *** (not found or expired)");
            return Response.status(401).build();
        }
        
        logger.debug("Subscription verified successfully: status={}", sub.getStatus());
        return Response.ok(sub).build();
        
    } catch (IOException e) {
        logger.error("Failed to verify subscription due to infrastructure failure", e);
        return Response.status(503).build();
    }
}
```

**Critical Rules**:

- ✅ Always log exceptions with stack trace: `logger.error("msg", exception)`
- ✅ Mask sensitive data (API keys, passwords, PII): `apiKey: ***`
- ✅ Include context (user ID, request ID): `logger.info("msg userId={}", userId)`
- ❌ Never log in loops (use sampling or aggregate)
- ❌ Never log passwords or credentials

---

## 6. Metrics Patterns

### Pattern 6.1: Standard Metrics (Google SRE Golden Signals)

**Google SRE Reference**: [The Four Golden Signals](https://sre.google/sre-book/monitoring-distributed-systems/#xref_monitoring_golden-signals)

**Four Golden Signals**:

1. **Latency**: Time to service a request
2. **Traffic**: Number of requests per second
3. **Errors**: Rate of failed requests
4. **Saturation**: Resource utilization (CPU, memory, connections)

**Standard Metrics to Track**:

```java
// Counters
metrics.incrementCounter("subscription.verify.success");
metrics.incrementCounter("subscription.verify.invalid");
metrics.incrementCounter("subscription.verify.failure");
metrics.incrementCounter("subscription.verify.retry");

// Timers (latency)
Timer.Sample sample = Timer.start(registry);
try {
    result = operation();
} finally {
    sample.stop(Timer.builder("subscription.verify.latency")
        .tag("status", "success")
        .register(registry));
}

// Gauges (saturation)
registry.gauge("subscription.cache.size", cache, Cache::size);
registry.gauge("subscription.connection.pool.active", pool, Pool::getActiveCount);
```

---

## 7. Complete Production-Ready Example

Combining all patterns:

```java
/**
 * Production-ready subscription verifier implementation.
 * Follows Google API design best practices.
 */
public class SubscriptionVerifierImpl implements SubscriptionVerifier {
    private static final Logger logger = LoggerFactory.getLogger(SubscriptionVerifierImpl.class);
    private static final int MAX_RETRIES = 3;
    private static final int INITIAL_DELAY_MS = 1000;
    
    private final HttpClient httpClient;
    private final MeterRegistry metrics;
    
    @Override
    public Subscription verify(String apiKey) throws IOException {
        // Step 1: Input validation
        if (apiKey == null || apiKey.isBlank()) {
            logger.warn("Invalid apiKey: null or blank");
            metrics.incrementCounter("subscription.verify.invalid_input");
            throw new IllegalArgumentException("apiKey must not be null or blank");
        }
        
        // Step 2: Execute with retry and metrics
        Timer.Sample sample = Timer.start(metrics);
        try {
            Subscription result = retryableVerify(apiKey);
            
            // Step 3: Record metrics
            sample.stop(Timer.builder("subscription.verify.latency")
                .tag("status", result == null ? "not_found" : "success")
                .register(metrics));
            
            if (result == null) {
                logger.info("Subscription not found or expired for apiKey: ***");
                metrics.incrementCounter("subscription.verify.not_found");
            } else {
                logger.debug("Subscription verified: status={}", result.getStatus());
                metrics.incrementCounter("subscription.verify.success");
            }
            
            return result;
            
        } catch (IOException e) {
            sample.stop(Timer.builder("subscription.verify.latency")
                .tag("status", "failure")
                .register(metrics));
            metrics.incrementCounter("subscription.verify.failure");
            throw e;
        }
    }
    
    private Subscription retryableVerify(String apiKey) throws IOException {
        int delay = INITIAL_DELAY_MS;
        
        for (int attempt = 0; attempt <= MAX_RETRIES; attempt++) {
            try {
                HttpResponse response = httpClient.get("/subscriptions/" + apiKey);
                return handleResponse(response);
                
            } catch (SocketTimeoutException | ConnectException e) {
                if (attempt == MAX_RETRIES) {
                    logger.error("Failed to verify after {} retries", MAX_RETRIES, e);
                    throw new IOException("Verification failed after retries", e);
                }
                
                logger.warn("Retry {} due to: {}", attempt + 1, e.getMessage());
                metrics.incrementCounter("subscription.verify.retry");
                
                Thread.sleep(delay);
                delay *= 2;
            }
        }
        
        throw new IllegalStateException("Unreachable");
    }
    
    private Subscription handleResponse(HttpResponse response) throws IOException {
        switch (response.statusCode()) {
            case 200:
                return parseSubscription(response.body());
            case 404:
                return null;  // Not found
            case 401:
            case 403:
                throw new SecurityException("Authentication failed: " + response.body());
            case 500:
            case 502:
            case 503:
                throw new IOException("Server error: " + response.statusCode());
            default:
                throw new IOException("Unexpected status: " + response.statusCode());
        }
    }
}
```

---

## References

- [Google API Improvement Proposals (AIP)](https://google.aip.dev/)
- [Google SRE Book - Monitoring](https://sre.google/sre-book/monitoring-distributed-systems/)
- [Effective Java, 3rd Edition](https://www.oreilly.com/library/view/effective-java-3rd/9780134686097/)
- [Java Concurrency in Practice](https://jcip.net/)
