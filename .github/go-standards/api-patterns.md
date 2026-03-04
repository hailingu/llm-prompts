# Standard API Patterns for Go

**Purpose**: Provide standardized Go API design patterns ensuring consistency and best practices. Based on **Effective Go**, **Go Code Review Comments**, and Google SRE practices.

**Version**: 1.0  
**Last Updated**: 2026-01-26

---

## Table of Contents

1. [Error Handling Patterns](#1-error-handling-patterns)
2. [Retry Strategy Patterns](#2-retry-strategy-patterns)
3. [HTTP Status Code Mapping](#3-http-status-code-mapping)
4. [Concurrency Patterns](#4-concurrency-patterns)
5. [Logging Patterns](#5-logging-patterns)
6. [Context Patterns](#6-context-patterns)

---

## 1. Error Handling Patterns

### Pattern 1.1: Sentinel Errors vs Wrapped Errors

**Effective Go Reference**: [Errors](https://go.dev/doc/effective_go#errors)

**Rule**:

- **Domain/business errors** (expected) → Sentinel errors (package-level `var`)
- **Infrastructure errors** (unexpected) → Wrap with context using `fmt.Errorf("%w", err)`
- **Input validation errors** → Sentinel errors with details

**Example Contract Table**:

| Scenario           | Category       | Return Value   | Error                                                  | HTTP Status |
| ------------------ | -------------- | -------------- | ------------------------------------------------------ | ----------- |
| -------------      | ----------     | -------------- | -----------                                            | --------    |
| Resource not found | Domain         | `nil`          | `ErrUserNotFound`                                      | 404         |
| Invalid input      | Validation     | `nil`          | `ErrInvalidInput`                                      | 400         |
| DB timeout         | Infrastructure | `nil`          | `fmt.Errorf("db query: %w", context.DeadlineExceeded)` | 503         |
| DB unavailable     | Infrastructure | `nil`          | `ErrDatabaseUnavailable`                               | 503         |

**Code Template**:

```go
package user

import (
    "context"
    "errors"
    "fmt"
)

// Domain errors (sentinel errors)
var (
    ErrUserNotFound        = errors.New("user not found")
    ErrDuplicateUser       = errors.New("user already exists")
    ErrInvalidInput        = errors.New("invalid input")
    ErrDatabaseUnavailable = errors.New("database unavailable")
)

// GetUserByID retrieves a user by ID.
//
// Returns:
//   - *User: User object if found
//   - error: ErrUserNotFound if not found,
//            ErrInvalidInput if id is invalid,
//            or wrapped infrastructure error
func (s *Service) GetUserByID(ctx context.Context, id string) (*User, error) {
    // Input validation
    if id == "" {
        return nil, fmt.Errorf("%w: id is empty", ErrInvalidInput)
    }

    // Domain logic
    user, err := s.repo.FindByID(ctx, id)
    if err != nil {
        // Check for expected domain errors
        if errors.Is(err, ErrUserNotFound) {
            return nil, err
        }
        
        // Wrap infrastructure errors with context
        return nil, fmt.Errorf("failed to get user %s: %w", id, err)
    }

    return user, nil
}
```

---

### Pattern 1.2: Error Checking with errors.Is and errors.As

**Go Code Review Comments**: [Error Handling](https://github.com/golang/go/wiki/CodeReviewComments#error-strings)

**When to use**:

- ✅ `errors.Is(err, target)` - Check if error matches a sentinel error
- ✅ `errors.As(err, &target)` - Extract specific error type
- ❌ String comparison (`err.Error() == "..."`) - Brittle, avoid

**Code Template**:

```go
func (c *Client) CallAPI(ctx context.Context, req *Request) (*Response, error) {
    resp, err := c.service.Process(ctx, req)
    if err != nil {
        // Check for specific domain errors
        if errors.Is(err, ErrUserNotFound) {
            // Handle not found case
            return nil, fmt.Errorf("user lookup failed: %w", err)
        }

        // Check for validation errors
        if errors.Is(err, ErrInvalidInput) {
            // Handle validation failure
            return nil, err // Don't retry
        }

        // Check for context errors (timeout/cancellation)
        if errors.Is(err, context.DeadlineExceeded) {
            // Retry with longer timeout
            return c.retryWithBackoff(ctx, req)
        }

        // Infrastructure error - wrap and return
        return nil, fmt.Errorf("api call failed: %w", err)
    }

    return resp, nil
}

// Example using errors.As to extract custom error type
func handleError(err error) {
    var validationErr *ValidationError
    if errors.As(err, &validationErr) {
        // Access specific fields
        for field, msg := range validationErr.Fields {
            log.Printf("validation error: %s: %s", field, msg)
        }
    }
}

type ValidationError struct {
    Fields map[string]string
}

func (e *ValidationError) Error() string {
    return "validation failed"
}
```

---

## 2. Retry Strategy Patterns

### Pattern 2.1: Exponential Backoff with Jitter

**Google SRE Reference**: [The SRE Book - Handling Overload](https://sre.google/sre-book/handling-overload/)

**Standard Parameters**:

```go
const (
    MaxRetries    = 3
    InitialDelay  = 100 * time.Millisecond
    MaxDelay      = 10 * time.Second
    BackoffFactor = 2.0
    JitterFactor  = 0.1 // ±10% randomness
)
```

**Retry Sequence**:

- Attempt 1: immediate
- Attempt 2: 100ms ± 10ms (90ms - 110ms)
- Attempt 3: 200ms ± 20ms (180ms - 220ms)
- Attempt 4: 400ms ± 40ms (360ms - 440ms)

**Code Template**:

```go
package retry

import (
    "context"
    "errors"
    "fmt"
    "math/rand"
    "time"
)

const (
    MaxRetries    = 3
    InitialDelay  = 100 * time.Millisecond
    BackoffFactor = 2.0
    JitterFactor  = 0.1
)

// WithRetry executes operation with exponential backoff retry.
func WithRetry(ctx context.Context, operation func() error, isRetryable func(error) bool) error {
    delay := InitialDelay
    var lastErr error

    for attempt := 0; attempt <= MaxRetries; attempt++ {
        // Execute operation
        err := operation()
        if err == nil {
            return nil // Success
        }

        lastErr = err

        // Check if error is retryable
        if !isRetryable(err) {
            return fmt.Errorf("non-retryable error (attempt %d): %w", attempt+1, err)
        }

        // Last attempt - don't sleep
        if attempt == MaxRetries {
            return fmt.Errorf("max retries exceeded (%d): %w", MaxRetries, lastErr)
        }

        // Add jitter: ±10%
        jitter := time.Duration(float64(delay) * JitterFactor * (rand.Float64()*2 - 1))
        actualDelay := delay + jitter

        // Wait with context cancellation support
        select {
        case <-time.After(actualDelay):
            // Continue to next attempt
        case <-ctx.Done():
            return fmt.Errorf("retry cancelled: %w", ctx.Err())
        }

        // Increase delay exponentially
        delay = time.Duration(float64(delay) * BackoffFactor)
    }

    return lastErr
}

// IsRetryableError determines if an error should be retried.
func IsRetryableError(err error) bool {
    // Retry on infrastructure/transient errors
    if errors.Is(err, context.DeadlineExceeded) {
        return true
    }
    if errors.Is(err, ErrDatabaseUnavailable) {
        return true
    }
    
    // Don't retry on domain errors
    if errors.Is(err, ErrUserNotFound) {
        return false
    }
    if errors.Is(err, ErrInvalidInput) {
        return false
    }
    
    // Default: retry on unknown errors (conservative)
    return true
}
```

**Usage Example**:

```go
func (s *Service) GetUserWithRetry(ctx context.Context, id string) (*User, error) {
    var user *User
    
    err := retry.WithRetry(ctx, func() error {
        var err error
        user, err = s.repo.FindByID(ctx, id)
        return err
    }, retry.IsRetryableError)
    
    if err != nil {
        return nil, fmt.Errorf("get user failed: %w", err)
    }
    
    return user, nil
}
```

---

## 3. HTTP Status Code Mapping

### Pattern 3.1: Standard Mapping Table

**Reference**: [Go HTTP Status Constants](https://pkg.go.dev/net/http#pkg-constants)

| HTTP Status                          | Scenario                    | Return Value   | Error                   | Retry?             |
| ------------------------------------ | --------------------------- | -------------- | ----------------------- | ------------------ |
| -------------                        | ----------                  | -------------- | -----------             | --------           |
| **2xx Success**                      |                             |                |                         |                    |
| 200 OK                               | Successful operation        | `*Object`      | `nil`                   | No                 |
| 201 Created                          | Resource created            | `*Object`      | `nil`                   | No                 |
| 204 No Content                       | Successful deletion         | `nil`          | `nil`                   | No                 |
| **3xx Redirection**                  |                             |                |                         |                    |
| 304 Not Modified                     | Resource unchanged          | `nil`          | `nil`                   | No                 |
| **4xx Client Errors** (DO NOT RETRY) |                             |                |                         |                    |
| 400 Bad Request                      | Invalid input format        | `nil`          | `ErrInvalidInput`       | No                 |
| 401 Unauthorized                     | Invalid credentials         | `nil`          | `ErrUnauthorized`       | No                 |
| 403 Forbidden                        | Permission denied           | `nil`          | `ErrForbidden`          | No                 |
| 404 Not Found                        | Resource doesn't exist      | `nil`          | `ErrNotFound`           | No                 |
| 409 Conflict                         | Version conflict            | `nil`          | `ErrConflict`           | No                 |
| 429 Too Many Requests                | Rate limit exceeded         | `nil`          | `ErrRateLimitExceeded`  | Yes (with backoff) |
| **5xx Server Errors** (RETRY)        |                             |                |                         |                    |
| 500 Internal Server Error            | Server bug                  | `nil`          | wrapped error           | Yes                |
| 502 Bad Gateway                      | Proxy error                 | `nil`          | wrapped error           | Yes                |
| 503 Service Unavailable              | Server overload/maintenance | `nil`          | `ErrServiceUnavailable` | Yes                |
| 504 Gateway Timeout                  | Upstream timeout            | `nil`          | wrapped context error   | Yes                |

**Code Template**:

```go
package httpclient

import (
    "encoding/json"
    "errors"
    "fmt"
    "io"
    "net/http"
)

// Domain errors
var (
    ErrInvalidInput        = errors.New("invalid input")
    ErrUnauthorized        = errors.New("unauthorized")
    ErrForbidden           = errors.New("forbidden")
    ErrNotFound            = errors.New("not found")
    ErrConflict            = errors.New("conflict")
    ErrRateLimitExceeded   = errors.New("rate limit exceeded")
    ErrServiceUnavailable  = errors.New("service unavailable")
)

// HandleHTTPResponse maps HTTP status codes to Go errors.
func HandleHTTPResponse(resp *http.Response, v interface{}) error {
    defer resp.Body.Close()

    // Read body for error messages
    body, err := io.ReadAll(resp.Body)
    if err != nil {
        return fmt.Errorf("failed to read response body: %w", err)
    }

    switch resp.StatusCode {
    case http.StatusOK, http.StatusCreated:
        // Success - decode response
        if v != nil {
            if err := json.Unmarshal(body, v); err != nil {
                return fmt.Errorf("failed to decode response: %w", err)
            }
        }
        return nil

    case http.StatusNoContent:
        // Success - no body
        return nil

    case http.StatusNotModified:
        // Not modified
        return nil

    case http.StatusBadRequest:
        return fmt.Errorf("%w: %s", ErrInvalidInput, body)

    case http.StatusUnauthorized:
        return fmt.Errorf("%w: %s", ErrUnauthorized, body)

    case http.StatusForbidden:
        return fmt.Errorf("%w: %s", ErrForbidden, body)

    case http.StatusNotFound:
        return ErrNotFound

    case http.StatusConflict:
        return fmt.Errorf("%w: %s", ErrConflict, body)

    case http.StatusTooManyRequests:
        return ErrRateLimitExceeded

    case http.StatusInternalServerError,
         http.StatusBadGateway,
         http.StatusServiceUnavailable,
         http.StatusGatewayTimeout:
        return fmt.Errorf("server error %d: %s", resp.StatusCode, body)

    default:
        return fmt.Errorf("unexpected status code %d: %s", resp.StatusCode, body)
    }
}

// IsRetryableHTTPError determines if HTTP error should be retried.
func IsRetryableHTTPError(err error) bool {
    // Retry 5xx errors
    if errors.Is(err, ErrServiceUnavailable) {
        return true
    }
    
    // Retry rate limit with backoff
    if errors.Is(err, ErrRateLimitExceeded) {
        return true
    }
    
    // Don't retry 4xx errors
    if errors.Is(err, ErrInvalidInput) ||
       errors.Is(err, ErrUnauthorized) ||
       errors.Is(err, ErrForbidden) ||
       errors.Is(err, ErrNotFound) ||
       errors.Is(err, ErrConflict) {
        return false
    }
    
    return true
}
```

---

## 4. Concurrency Patterns

### Pattern 4.1: Stateless Services (Recommended)

**Effective Go Reference**: [Concurrency](https://go.dev/doc/effective_go#concurrency)

**Benefits**:

- ✅ Naturally goroutine-safe (no shared mutable state)
- ✅ No synchronization overhead
- ✅ Horizontally scalable
- ✅ Simple to reason about

**Template**:

```go
package user

import (
    "context"
)

// Service is a stateless user service.
// All methods are safe for concurrent use by multiple goroutines.
type Service struct {
    repo Repository // Stateless dependency (or connection pool)
}

// NewService creates a new stateless service.
func NewService(repo Repository) *Service {
    return &Service{repo: repo}
}

// GetUserByID is safe for concurrent use (stateless implementation).
func (s *Service) GetUserByID(ctx context.Context, id string) (*User, error) {
    // No shared mutable state - naturally goroutine-safe
    return s.repo.FindByID(ctx, id)
}
```

**Goroutine-Safety Annotation**:
```go
// Service provides user management operations.
// Goroutine-safety: Yes (stateless design, no shared mutable state)
type Service struct { ... }
```

---

### Pattern 4.2: Mutex-Protected State (Low Contention)

**When to use**: Simple shared state, low contention

**Template**:

```go
package cache

import (
    "sync"
    "time"
)

// Cache is a simple in-memory cache with mutex protection.
// All methods are safe for concurrent use by multiple goroutines.
type Cache struct {
    mu    sync.RWMutex
    items map[string]*Item
}

type Item struct {
    Value      interface{}
    Expiration time.Time
}

// NewCache creates a new cache.
func NewCache() *Cache {
    return &Cache{
        items: make(map[string]*Item),
    }
}

// Get retrieves an item from the cache.
// Goroutine-safety: Yes (RWMutex-protected reads)
func (c *Cache) Get(key string) (interface{}, bool) {
    c.mu.RLock()
    defer c.mu.RUnlock()

    item, exists := c.items[key]
    if !exists || time.Now().After(item.Expiration) {
        return nil, false
    }

    return item.Value, true
}

// Set stores an item in the cache.
// Goroutine-safety: Yes (Mutex-protected writes)
func (c *Cache) Set(key string, value interface{}, ttl time.Duration) {
    c.mu.Lock()
    defer c.mu.Unlock()

    c.items[key] = &Item{
        Value:      value,
        Expiration: time.Now().Add(ttl),
    }
}
```

---

### Pattern 4.3: sync.Map for High Contention

**When to use**: High contention, mostly reads

**Template**:

```go
package cache

import (
    "sync"
    "time"
)

// ConcurrentCache is a high-performance cache using sync.Map.
// All methods are safe for concurrent use by multiple goroutines.
type ConcurrentCache struct {
    items sync.Map
}

type cacheItem struct {
    Value      interface{}
    Expiration time.Time
}

// NewConcurrentCache creates a new concurrent cache.
func NewConcurrentCache() *ConcurrentCache {
    return &ConcurrentCache{}
}

// Get retrieves an item from the cache.
// Goroutine-safety: Yes (sync.Map)
func (c *ConcurrentCache) Get(key string) (interface{}, bool) {
    v, ok := c.items.Load(key)
    if !ok {
        return nil, false
    }

    item := v.(*cacheItem)
    if time.Now().After(item.Expiration) {
        c.items.Delete(key)
        return nil, false
    }

    return item.Value, true
}

// Set stores an item in the cache.
// Goroutine-safety: Yes (sync.Map)
func (c *ConcurrentCache) Set(key string, value interface{}, ttl time.Duration) {
    c.items.Store(key, &cacheItem{
        Value:      value,
        Expiration: time.Now().Add(ttl),
    })
}
```

---

### Pattern 4.4: Worker Pool Pattern

**When to use**: Bounded concurrency, rate limiting

**Template**:

```go
package worker

import (
    "context"
    "sync"
)

// Pool manages a fixed number of worker goroutines.
type Pool struct {
    workers int
    jobs    chan func()
    wg      sync.WaitGroup
}

// NewPool creates a worker pool with n workers.
func NewPool(workers int) *Pool {
    p := &Pool{
        workers: workers,
        jobs:    make(chan func(), workers*2), // Buffered channel
    }
    
    // Start workers
    for i := 0; i < workers; i++ {
        p.wg.Add(1)
        go p.worker()
    }
    
    return p
}

// Submit submits a job to the pool.
func (p *Pool) Submit(ctx context.Context, job func()) error {
    select {
    case p.jobs <- job:
        return nil
    case <-ctx.Done():
        return ctx.Err()
    }
}

// worker processes jobs from the jobs channel.
func (p *Pool) worker() {
    defer p.wg.Done()
    
    for job := range p.jobs {
        job()
    }
}

// Shutdown gracefully shuts down the pool.
func (p *Pool) Shutdown() {
    close(p.jobs)
    p.wg.Wait()
}
```

---

## 5. Logging Patterns

### Pattern 5.1: Structured Logging with log/slog

**Go 1.21+ Standard**: [log/slog](https://pkg.go.dev/log/slog)

**Log Levels**:

| Level         | When to Use                                       | Example                                                    |
| ------------- | ------------------------------------------------- | ---------------------------------------------------------- |
| ------------- | -----------                                       | -------                                                    |
| **Error**     | System failure, requires immediate attention      | Database connection failure, external API down             |
| **Warn**      | Unexpected but handled situation                  | Retry attempt, deprecated API usage, rate limit approached |
| **Info**      | Business-level events                             | User login, order created, subscription expired            |
| **Debug**     | Detailed diagnostic info (disabled in production) | Method entry/exit, variable values                         |

**Code Template**:

```go
package main

import (
    "context"
    "errors"
    "log/slog"
    "os"
)

var logger *slog.Logger

func init() {
    // Initialize structured logger
    logger = slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{
        Level: slog.LevelInfo, // Change to LevelDebug for development
    }))
}

func (s *Service) GetUserByID(ctx context.Context, id string) (*User, error) {
    logger.Debug("getting user by id",
        "user_id", id,
        "operation", "GetUserByID")

    if id == "" {
        logger.Warn("invalid user id provided",
            "user_id", id,
            "error", "empty id")
        return nil, ErrInvalidInput
    }

    user, err := s.repo.FindByID(ctx, id)
    if err != nil {
        if errors.Is(err, ErrUserNotFound) {
            logger.Info("user not found",
                "user_id", id)
            return nil, err
        }

        logger.Error("failed to get user",
            "user_id", id,
            "error", err)
        return nil, fmt.Errorf("get user failed: %w", err)
    }

    logger.Debug("user retrieved successfully",
        "user_id", id,
        "user_status", user.Status)

    return user, nil
}
```

**Critical Rules**:

- ✅ Use structured fields: `logger.Info("msg", "key", value)`
- ✅ Mask sensitive data: `"api_key", "***"`
- ✅ Include context: request ID, user ID, trace ID
- ✅ Log errors with full context: `"error", err`
- ❌ Never log in tight loops (use sampling)
- ❌ Never log passwords or credentials

---

## 6. Context Patterns

### Pattern 6.1: Context as First Parameter

**Effective Go**: [Context](https://go.dev/blog/context)

**Rule**: Always accept `context.Context` as the **first parameter**

**Template**:

```go
// ✅ Correct: context as first parameter
func (s *Service) GetUserByID(ctx context.Context, id string) (*User, error) {
    // Use context for cancellation, timeout, values
}

// ❌ Wrong: context not first
func (s *Service) GetUserByID(id string, ctx context.Context) (*User, error) {
    // Violates Go conventions
}
```

---

### Pattern 6.2: Context Timeout and Cancellation

**Template**:

```go
package main

import (
    "context"
    "time"
)

func (c *Client) CallAPIWithTimeout(userID string) (*User, error) {
    // Create context with timeout
    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
    defer cancel() // Always call cancel to release resources

    // Pass context to operation
    user, err := c.service.GetUserByID(ctx, userID)
    if err != nil {
        // Check for timeout
        if errors.Is(err, context.DeadlineExceeded) {
            return nil, fmt.Errorf("request timed out: %w", err)
        }
        return nil, err
    }

    return user, nil
}

// Respect context cancellation in long-running operations
func (s *Service) ProcessUsers(ctx context.Context, userIDs []string) error {
    for _, id := range userIDs {
        // Check for cancellation before each iteration
        select {
        case <-ctx.Done():
            return fmt.Errorf("processing cancelled: %w", ctx.Err())
        default:
            // Continue processing
        }

        if err := s.processUser(ctx, id); err != nil {
            return err
        }
    }
    return nil
}
```

---

## 7. Complete Production-Ready Example

Combining all patterns:

```go
package user

import (
    "context"
    "errors"
    "fmt"
    "log/slog"
    "time"
)

// Domain errors
var (
    ErrUserNotFound        = errors.New("user not found")
    ErrInvalidInput        = errors.New("invalid input")
    ErrDatabaseUnavailable = errors.New("database unavailable")
)

// Service provides user management operations.
// All methods are safe for concurrent use by multiple goroutines.
// Goroutine-safety: Yes (stateless design)
type Service struct {
    repo   Repository
    cache  *Cache
    logger *slog.Logger
}

// NewService creates a new user service.
func NewService(repo Repository, cache *Cache, logger *slog.Logger) *Service {
    return &Service{
        repo:   repo,
        cache:  cache,
        logger: logger,
    }
}

// GetUserByID retrieves a user by ID with retry, caching, and logging.
//
// Parameters:
//   - ctx: Context for cancellation and timeout
//   - id: User ID (must be non-empty)
//
// Returns:
//   - *User: User object if found
//   - error: ErrUserNotFound if not found,
//            ErrInvalidInput if id is invalid,
//            or wrapped infrastructure error
//
// Goroutine-safety: Yes (stateless)
// Idempotent: Yes
func (s *Service) GetUserByID(ctx context.Context, id string) (*User, error) {
    s.logger.Debug("getting user",
        "user_id", id,
        "operation", "GetUserByID")

    // Input validation
    if id == "" {
        s.logger.Warn("invalid user id", "error", "empty id")
        return nil, fmt.Errorf("%w: id is empty", ErrInvalidInput)
    }

    // Check cache
    if cached, ok := s.cache.Get(id); ok {
        s.logger.Debug("cache hit", "user_id", id)
        return cached.(*User), nil
    }

    // Fetch with retry
    var user *User
    err := withRetry(ctx, func() error {
        var err error
        user, err = s.repo.FindByID(ctx, id)
        return err
    }, s.isRetryable, s.logger)

    if err != nil {
        if errors.Is(err, ErrUserNotFound) {
            s.logger.Info("user not found", "user_id", id)
            return nil, err
        }

        s.logger.Error("failed to get user",
            "user_id", id,
            "error", err)
        return nil, fmt.Errorf("get user failed: %w", err)
    }

    // Cache result
    s.cache.Set(id, user, 5*time.Minute)

    s.logger.Debug("user retrieved",
        "user_id", id,
        "status", user.Status)

    return user, nil
}

// withRetry executes operation with exponential backoff.
func withRetry(ctx context.Context, operation func() error, isRetryable func(error) bool, logger *slog.Logger) error {
    const (
        maxRetries    = 3
        initialDelay  = 100 * time.Millisecond
        backoffFactor = 2.0
    )

    delay := initialDelay
    var lastErr error

    for attempt := 0; attempt <= maxRetries; attempt++ {
        err := operation()
        if err == nil {
            return nil
        }

        lastErr = err

        if !isRetryable(err) {
            return err
        }

        if attempt == maxRetries {
            return fmt.Errorf("max retries exceeded: %w", lastErr)
        }

        logger.Warn("retrying after error",
            "attempt", attempt+1,
            "delay_ms", delay.Milliseconds(),
            "error", err)

        select {
        case <-time.After(delay):
        case <-ctx.Done():
            return ctx.Err()
        }

        delay = time.Duration(float64(delay) * backoffFactor)
    }

    return lastErr
}

// isRetryable determines if an error should be retried.
func (s *Service) isRetryable(err error) bool {
    if errors.Is(err, context.DeadlineExceeded) {
        return true
    }
    if errors.Is(err, ErrDatabaseUnavailable) {
        return true
    }
    if errors.Is(err, ErrUserNotFound) || errors.Is(err, ErrInvalidInput) {
        return false
    }
    return true
}
```

---

## References

- [Effective Go](https://go.dev/doc/effective_go) - Official Go documentation
- [Go Code Review Comments](https://github.com/golang/go/wiki/CodeReviewComments) - Style guide
- [Google SRE Book - Monitoring](https://sre.google/sre-book/monitoring-distributed-systems/)
- [Go Blog: Error handling and Go](https://go.dev/blog/error-handling-and-go)
- [Go Blog: Context](https://go.dev/blog/context)
- [Go Blog: Errors are values](https://go.dev/blog/errors-are-values)
