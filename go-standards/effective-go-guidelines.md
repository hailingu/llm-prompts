# Effective Go Guidelines & Best Practices

**Version**: 1.0  
**Last Updated**: 2026-01-26  
**Author**: Based on [Effective Go](https://go.dev/doc/effective_go) and [Go Code Review Comments](https://github.com/golang/go/wiki/CodeReviewComments)

---

## Table of Contents

1. [Naming Conventions](#1-naming-conventions)
2. [Package Design](#2-package-design)
3. [Formatting](#3-formatting)
4. [Commentary](#4-commentary)
5. [Control Structures](#5-control-structures)
6. [Functions and Methods](#6-functions-and-methods)
7. [Data Structures](#7-data-structures)
8. [Interfaces](#8-interfaces)
9. [Concurrency](#9-concurrency)
10. [Error Handling](#10-error-handling)
11. [Testing](#11-testing)
12. [Performance](#12-performance)

---

## 1. Naming Conventions

### 1.1 General Principles

#### Use MixedCaps or mixedCaps, never underscores

Use MixedCaps or mixedCaps for identifier names; avoid underscores in identifiers to follow Go conventions.

✅ **Correct**:

```go
type UserService struct {}
func getUserByID() {}
const MaxRetryCount = 3
```

❌ **Incorrect**:

```go
type User_Service struct {}
func get_user_by_id() {}
const MAX_RETRY_COUNT = 3
```

### 1.2 Package Names

#### Packages should have short, lowercase, single-word names

Package names should be short, all lowercase, and a single word; they should also match the directory name.

✅ **Correct**:

```go
package user
package http
package encoding
```

❌ **Incorrect**:

```go
package userService
package HTTPClient
package encoding_utils
```

**Package name should match directory name**:

- Directory: `user/` → Package: `package user`
- Directory: `httputil/` → Package: `package httputil`

### 1.3 Exported vs Unexported

#### Exported names start with uppercase, unexported with lowercase

Exported identifiers begin with an uppercase letter and are visible outside the package; unexported identifiers begin with a lowercase letter and are package-private.

✅ **Correct**:

```go
// Exported (visible outside package)
type User struct {
    Name string      // exported field
}

func GetUser() {}   // exported function

// Unexported (package-private)
type userCache struct {
    data map[string]*User  // unexported field
}

func validateEmail() {}  // unexported function
```

### 1.4 Acronyms and Initialisms

#### Keep acronyms consistent: all uppercase or all lowercase

Keep acronyms consistent: use all-uppercase for exported identifiers (e.g., `HTTP`) and all-lowercase for unexported identifiers (e.g., `http`).

✅ **Correct**:

```go
type HTTPServer struct {}   // All caps when exported
var httpServer *HTTPServer  // All lowercase when unexported
func ServeHTTP() {}         // Consistent with type

type userID string          // All lowercase when unexported
```

❌ **Incorrect**:

```go
type HttpServer struct {}   // Mixed case
func ServeHttp() {}         // Inconsistent
```

**Common acronyms**: URL, HTTP, ID, API, JSON, XML, SQL, HTML, CSS

### 1.5 Getters and Setters

#### Don't use Get prefix for getters

Prefer method names without the `Get` prefix (e.g., use `Name()` instead of `GetName()`); this follows Go conventions and avoids redundant naming.

✅ **Correct**:

```go
type User struct {
    name string
}

func (u *User) Name() string { return u.name }           // getter
func (u *User) SetName(name string) { u.name = name }   // setter
```

❌ **Incorrect**:

```go
func (u *User) GetName() string { return u.name }  // Don't use Get prefix
```

### 1.6 Interface Names

#### Single-method interfaces use -er suffix

Prefer single-method interfaces to use the `-er` suffix (e.g., `Reader`, `Writer`) to convey capability and follow Go naming conventions.

✅ **Correct**:

```go
type Reader interface {
    Read(p []byte) (n int, err error)
}

type Writer interface {
    Write(p []byte) (n int, err error)
}

type Stringer interface {
    String() string
}
```

**Multi-method interfaces describe the concept**:

```go
type UserRepository interface {
    FindByID(id string) (*User, error)
    Save(user *User) error
}
```

---

## 2. Package Design

### 2.1 Package Organization

**Good package design**:

- Small, focused packages
- Clear, single responsibility
- No circular dependencies

✅ **Correct structure**:

```text
myapp/
├── cmd/
│   └── server/
│       └── main.go
├── internal/
│   ├── user/
│   │   ├── service.go
│   │   └── repository.go
│   └── auth/
│       └── jwt.go
└── pkg/
    └── logger/
        └── logger.go
```

### 2.2 Import Grouping

#### Group imports: stdlib, third-party, local

Group imports into three sections: standard library, third-party packages, and local packages. Separate groups with a blank line to improve readability and follow Go import ordering conventions.

✅ **Correct**:

```go
import (
    // Standard library
    "context"
    "fmt"
    "time"

    // Third-party
    "github.com/google/uuid"
    "github.com/stretchr/testify/assert"

    // Local
    "github.com/myorg/myapp/internal/user"
)
```

### 2.3 Internal Packages

**Use `internal/` for package-private code**

```text
myapp/
├── internal/
│   └── auth/
│       └── jwt.go  ← Can only be imported by myapp/
└── pkg/
    └── logger/
        └── logger.go  ← Can be imported by external projects
```

---

## 3. Formatting

### 3.1 Automatic Formatting

**Always use `gofmt` or `goimports`**

```bash
# Format all files
gofmt -w .

# Format and fix imports
goimports -w .
```

**Never format manually. Let tools handle it.**

### 3.2 Line Length

#### No strict limit, but aim for 80-100 characters

Aim for readable lines (80–100 characters); break long function signatures and long expressions into multiple lines for clarity.

✅ **Reasonable**:

```go
func GetUserByID(ctx context.Context, id string) (*User, error) {
    // ...
}
```

**If too long, break into multiple lines**:

```go
func CreateUser(
    ctx context.Context,
    name string,
    email string,
    role string,
) (*User, error) {
    // ...
}
```

### 3.3 Indentation

#### Use tabs for indentation (handled by gofmt)

Use tabs for indentation; rely on `gofmt`/`goimports` to enforce formatting automatically.

---

## 4. Commentary

### 4.1 Package Comments

#### Every package should have a package comment

Each package should include a package-level comment (typically in `doc.go` or the package's primary file) that briefly describes the package's purpose and responsibilities.

✅ **Correct** (in `user.go` or `doc.go`):

```go
// Package user provides user management functionality.
// It includes user creation, retrieval, and authentication.
package user
```

### 4.2 Exported Item Comments

#### All exported types, functions, constants must have comments

All exported types, functions, and constants must have clear godoc comments that begin with the item name and describe its behavior for external callers.

✅ **Correct**:

```go
// User represents a system user with authentication credentials.
type User struct {
    ID    string
    Email string
}

// GetUserByID retrieves a user by their unique identifier.
// It returns ErrUserNotFound if the user does not exist.
func GetUserByID(ctx context.Context, id string) (*User, error) {
    // ...
}

// MaxRetryCount is the maximum number of retry attempts.
const MaxRetryCount = 3
```

**Comment starts with the name**:

```go
// User represents...     ✅ Good
// This struct...         ❌ Bad
```

### 4.3 Internal Comments

**Use `//` for single-line comments**

```go
// This is a single-line comment
x := 5

// This is a multi-line comment
// explaining something complex
// across multiple lines
y := 10
```

**Use `/* */` sparingly (mostly for disabling code)**

---

## 5. Control Structures

### 5.1 If Statements

#### Prefer short if with initialization

Prefer short `if` statements that perform initialization (e.g., `if err := doSomething(); err != nil { ... }`) for concise and idiomatic error handling.

✅ **Correct**:

```go
if err := doSomething(); err != nil {
    return err
}
```

❌ **Verbose**:

```go
err := doSomething()
if err != nil {
    return err
}
```

### 5.2 For Loops

#### Three forms of for loop

Cover the three common `for` loop forms: classic, while-style, and infinite loops, plus ranging over slices/maps.

```go
// Classic for
for i := 0; i < 10; i++ {
    fmt.Println(i)
}

// While-style
for condition {
    // ...
}

// Infinite loop
for {
    // ...
    if done {
        break
    }
}

// Range over slice/map
for i, val := range slice {
    // ...
}

for key, val := range myMap {
    // ...
}
```

**Ignore unused variables with `_`**:

```go
for _, val := range slice {  // Don't need index
    fmt.Println(val)
}
```

### 5.3 Switch Statements

#### Cases break automatically (no fallthrough by default)

Cases do not fall through by default in Go; use the `fallthrough` keyword explicitly when a subsequent case's code should execute.

✅ **Correct**:

```go
switch status {
case "active":
    // ...
case "inactive":
    // ...
default:
    // ...
}
```

**Use switch without expression for long if-else chains**:

```go
switch {
case x < 0:
    return "negative"
case x == 0:
    return "zero"
default:
    return "positive"
}
```

### 5.4 Defer

#### Defer executes when function returns

Defer statements run when the surrounding function returns; use `defer` to ensure resources are released even on early returns or errors.

```go
func ReadFile(filename string) ([]byte, error) {
    f, err := os.Open(filename)
    if err != nil {
        return nil, err
    }
    defer f.Close()  // Always close, even on error

    return ioutil.ReadAll(f)
}
```

**Multiple defers execute in LIFO order**:

```go
defer fmt.Println("1")
defer fmt.Println("2")
defer fmt.Println("3")
// Prints: 3, 2, 1
```

---

## 6. Functions and Methods

### 6.1 Function Signatures

#### Return error as last return value

Return `error` as the last return value following Go conventions; this makes error handling consistent and idiomatic.

✅ **Correct**:

```go
func GetUser(id string) (*User, error) {
    // ...
}
```

❌ **Incorrect**:

```go
func GetUser(id string) (error, *User) {  // Error should be last
    // ...
}
```

### 6.2 Receivers

**Use pointer receivers when:**

- Method modifies the receiver
- Receiver is large (> a few fields)
- Consistency (if some methods use pointer, all should)

✅ **Correct**:

```go
type User struct {
    Name  string
    Email string
}

// Modifies receiver → pointer
func (u *User) SetName(name string) {
    u.Name = name
}

// Doesn't modify → can be value or pointer
// Use pointer for consistency if other methods use pointer
func (u *User) GetName() string {
    return u.Name
}
```

### 6.3 Named Return Values

#### Use named returns for documentation, not for brevity

Use named return values when they document the intent of a function's results; avoid using them merely to shorten code as they can obscure control flow.

✅ **Good use** (documents intent):

```go
func Split(path string) (dir, file string) {
    i := strings.LastIndex(path, "/")
    return path[:i], path[i+1:]
}
```

❌ **Bad use** (obscures logic):

```go
func Calculate(x int) (result int) {
    result = x * 2
    if result > 100 {
        result = 100
        return
    }
    return
}
```

### 6.4 Variadic Functions

**Use `...` for variadic parameters**

```go
func Sum(nums ...int) int {
    total := 0
    for _, n := range nums {
        total += n
    }
    return total
}

// Call
Sum(1, 2, 3)
Sum(nums...)  // Unpack slice
```

---

## 7. Data Structures

### 7.1 Struct Literals

#### Use field names for clarity

Prefer struct literals that use field names to make code clearer and resilient to future field reordering.

✅ **Correct**:

```go
u := User{
    ID:    "123",
    Email: "test@example.com",
}
```

❌ **Fragile** (breaks if fields reordered):

```go
u := User{"123", "test@example.com"}
```

### 7.2 Zero Values

#### Design structs to have useful zero values

Design types so their zero value is useful and ready-to-use; this reduces initialization boilerplate and lowers the chance of nil-pointer issues.

✅ **Good design**:

```go
type Buffer struct {
    buf []byte
}

var b Buffer  // Ready to use, buf is nil but still works
b.Write([]byte("hello"))
```

### 7.3 Composite Literals

**Use `&` for pointer to struct**

```go
u := &User{
    ID: "123",
}
// u is *User
```

### 7.4 Maps

#### Check if key exists before using

Check for a key's presence using the two-value assignment (`val, ok := myMap[key]`) before using the value to avoid confusing zero values with missing keys.

✅ **Correct**:

```go
val, ok := myMap[key]
if ok {
    // Key exists, use val
}
```

**Initialize maps with `make`**:

```go
m := make(map[string]int)  // Empty map
m["key"] = 42
```

### 7.5 Slices

#### Understand slice capacity

Be mindful of slice length and capacity; preallocate when the size is known to avoid repeated allocations and improve performance.

```go
s := make([]int, 0, 10)  // len=0, cap=10
s = append(s, 1)         // len=1, cap=10
```

**Preallocate if size is known**:

```go
// Known size
users := make([]*User, 0, 100)
```

---

## 8. Interfaces

### 8.1 Small Interfaces

#### Prefer small, focused interfaces

Keep interfaces small and focused; single-method interfaces (the `-er` pattern) promote simple, composable code and are idiomatic in Go.

✅ **Good** (single method):

```go
type Reader interface {
    Read(p []byte) (n int, err error)
}
```

❌ **Bad** (too many methods):

```go
type DataAccess interface {
    GetUser(id string) (*User, error)
    SaveUser(u *User) error
    DeleteUser(id string) error
    ListUsers() ([]*User, error)
    // ... 10 more methods
}
```

### 8.2 Accept Interfaces, Return Structs

✅ **Correct**:

```go
// Accept interface (flexible)
func ProcessData(r io.Reader) error {
    // ...
}

// Return concrete type (clear)
func NewUser() *User {
    return &User{}
}
```

### 8.3 Empty Interface

**Use `interface{}` or `any` sparingly**

```go
// Go 1.18+
func Print(v any) {
    fmt.Println(v)
}

// Pre-1.18
func Print(v interface{}) {
    fmt.Println(v)
}
```

---

## 9. Concurrency

### 9.1 Goroutines

**Launch goroutines with `go` keyword**

```go
go func() {
    // Runs concurrently
}()
```

**Always ensure goroutines terminate**:

```go
ctx, cancel := context.WithCancel(context.Background())
defer cancel()

go func(ctx context.Context) {
    for {
        select {
        case <-ctx.Done():
            return  // Cleanup
        default:
            // Work
        }
    }
}(ctx)
```

### 9.2 Channels

#### Use channels to communicate between goroutines

Use channels to coordinate work and transfer data safely between goroutines; prefer channel ownership patterns and close channels when the sender is finished.

```go
ch := make(chan int)

// Send
go func() {
    ch <- 42
}()

// Receive
val := <-ch
```

**Close channels when done (sender closes)**:

```go
ch := make(chan int)

go func() {
    for i := 0; i < 10; i++ {
        ch <- i
    }
    close(ch)  // Signal no more values
}()

for val := range ch {  // Exits when channel closed
    fmt.Println(val)
}
```

### 9.3 Mutexes

**Use `sync.Mutex` for shared state**

```go
type Counter struct {
    mu    sync.Mutex
    count int
}

func (c *Counter) Increment() {
    c.mu.Lock()
    defer c.mu.Unlock()
    c.count++
}
```

**Use `sync.RWMutex` for read-heavy workloads**:

```go
type Cache struct {
    mu   sync.RWMutex
    data map[string]string
}

func (c *Cache) Get(key string) string {
    c.mu.RLock()
    defer c.mu.RUnlock()
    return c.data[key]
}

func (c *Cache) Set(key, val string) {
    c.mu.Lock()
    defer c.mu.Unlock()
    c.data[key] = val
}
```

### 9.4 WaitGroups

**Use `sync.WaitGroup` to wait for goroutines**

```go
var wg sync.WaitGroup

for i := 0; i < 10; i++ {
    wg.Add(1)
    go func(i int) {
        defer wg.Done()
        fmt.Println(i)
    }(i)
}

wg.Wait()  // Block until all Done()
```

### 9.5 Context

**Use `context.Context` for cancellation and timeout**

```go
ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
defer cancel()

result, err := DoWork(ctx)
if errors.Is(err, context.DeadlineExceeded) {
    // Timeout
}
```

---

## 10. Error Handling

### 10.1 Check All Errors

#### Never ignore errors

Always check and handle errors; ignoring them can hide failures and produce subtle bugs.

✅ **Correct**:

```go
f, err := os.Open(filename)
if err != nil {
    return err
}
defer f.Close()
```

❌ **Incorrect**:

```go
f, _ := os.Open(filename)  // Ignoring error
```

### 10.2 Error Types

#### Use sentinel errors for known error cases

Use sentinel errors (package-level error variables) for well-known conditions so callers can reliably check them with `errors.Is`; prefer wrapping errors to add context for callers.

```go
var (
    ErrUserNotFound = errors.New("user not found")
    ErrInvalidInput = errors.New("invalid input")
)

// Check with errors.Is
if errors.Is(err, ErrUserNotFound) {
    // Handle not found
}
```

**Wrap errors with context**:

```go
if err := validateUser(u); err != nil {
    return fmt.Errorf("validate user: %w", err)
}
```

**Custom error types for structured errors**:

```go
type ValidationError struct {
    Field string
    Msg   string
}

func (e *ValidationError) Error() string {
    return fmt.Sprintf("%s: %s", e.Field, e.Msg)
}
```

### 10.3 Panic and Recover

#### Don't use panic for normal error handling

Reserve `panic` for programmer errors or unrecoverable situations; use returned errors for regular error handling and recovery.

✅ **Correct** (return error):

```go
func Divide(a, b int) (int, error) {
    if b == 0 {
        return 0, errors.New("division by zero")
    }
    return a / b, nil
}
```

❌ **Incorrect** (panic):

```go
func Divide(a, b int) int {
    if b == 0 {
        panic("division by zero")
    }
    return a / b
}
```

**Panic only for programmer errors or unrecoverable situations**:

```go
func init() {
    if config == nil {
        panic("config must be initialized")
    }
}
```

---

## 11. Testing

### 11.1 Test File Naming

**Test files end with `_test.go`**

```text
user.go        ← Implementation
user_test.go   ← Tests
```

### 11.2 Test Function Naming

**Tests start with `Test`**

```go
func TestGetUserByID(t *testing.T) {
    // ...
}

func TestUserService_CreateUser(t *testing.T) {
    // ...
}
```

### 11.3 Table-Driven Tests

#### Use table-driven tests for comprehensive coverage

Table-driven tests let you express many cases concisely and make adding cases easy; use `t.Run` for subtests so failures are reported per case.

✅ **Correct**:

```go
func TestSum(t *testing.T) {
    tests := []struct {
        name string
        a, b int
        want int
    }{
        {"positive", 1, 2, 3},
        {"negative", -1, -2, -3},
        {"zero", 0, 0, 0},
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            got := Sum(tt.a, tt.b)
            if got != tt.want {
                t.Errorf("Sum(%d, %d) = %d, want %d", tt.a, tt.b, got, tt.want)
            }
        })
    }
}
```

### 11.4 Subtests

**Use `t.Run` for subtests**

```go
func TestUser(t *testing.T) {
    t.Run("ValidEmail", func(t *testing.T) {
        // ...
    })

    t.Run("InvalidEmail", func(t *testing.T) {
        // ...
    })
}
```

### 11.5 Test Helpers

**Mark test helpers with `t.Helper()`**

```go
func assertNoError(t *testing.T, err error) {
    t.Helper()  // Marks this as helper
    if err != nil {
        t.Fatalf("unexpected error: %v", err)
    }
}
```

---

## 12. Performance

### 12.1 Benchmarks

**Benchmark functions start with `Benchmark`**

```go
func BenchmarkSum(b *testing.B) {
    for i := 0; i < b.N; i++ {
        Sum(1, 2)
    }
}
```

**Run benchmarks**:

```bash
go test -bench=.
```

### 12.2 Avoid Allocations

#### Reuse buffers when possible

Reuse pooled buffers (e.g., `sync.Pool`) for frequently used temporary buffers to reduce allocations and GC pressure.

```go
var bufPool = sync.Pool{
    New: func() interface{} {
        return new(bytes.Buffer)
    },
}

func ProcessData(data []byte) {
    buf := bufPool.Get().(*bytes.Buffer)
    defer bufPool.Put(buf)
    buf.Reset()
    
    // Use buf
}
```

### 12.3 String Building

**Use `strings.Builder` for string concatenation**

✅ **Correct**:

```go
var sb strings.Builder
for _, s := range strs {
    sb.WriteString(s)
}
result := sb.String()
```

❌ **Slow**:

```go
result := ""
for _, s := range strs {
    result += s  // Creates new string each time
}
```

---

## Common Anti-Patterns to Avoid

### ❌ Anti-Pattern 1: Getters with Get prefix

```go
func (u *User) GetName() string  // Don't use Get
```

### ❌ Anti-Pattern 2: Ignoring errors

```go
_ = doSomething()  // Never ignore errors
```

### ❌ Anti-Pattern 3: Using panic for normal errors

```go
if err != nil {
    panic(err)  // Use return err instead
}
```

### ❌ Anti-Pattern 4: Global mutable state

```go
var userCache = make(map[string]*User)  // Unsafe for concurrent use
```

### ❌ Anti-Pattern 5: Not closing resources

```go
f, _ := os.Open(filename)
// Missing: defer f.Close()
```

---

## References

- [Effective Go](https://go.dev/doc/effective_go) - Official Go documentation
- [Go Code Review Comments](https://github.com/golang/go/wiki/CodeReviewComments) - Common review feedback
- [Standard Go Project Layout](https://github.com/golang-standards/project-layout) - Project structure
- [Go Proverbs](https://go-proverbs.github.io/) - Go philosophy

---

### End of Guidelines

This document summarizes recommended Go practices; consult the References above for source materials and further reading.
