# Go Static Analysis Setup

## Overview

This document describes the static code analysis tools configured for Go projects to enforce **Effective Go** and **Go Code Review Comments** best practices and detect potential issues early.

## Configured Tools

### 1. gofmt (Code Formatting) ⭐ NON-NEGOTIABLE

**Purpose**: Automatically format Go source code according to the official Go formatting standard.

**Installation**: Included with Go installation (no additional setup needed)

**Configuration**: No configuration file needed - gofmt enforces the standard Go format

**What it detects**:

- Incorrect indentation (tabs vs spaces)
- Inconsistent brace placement
- Improper spacing around operators
- Line breaks in wrong places

**Usage**:

```bash
# Check for unformatted files
gofmt -l .

# Fix formatting automatically
gofmt -w .

# Format and show diff
gofmt -d .
```

**Expected**: Zero unformatted files

**Note**: All Go code MUST be gofmt-formatted before submission. This is non-negotiable.

---

### 2. goimports (Import Management)

**Purpose**: Automatically add missing imports and remove unused imports.

**Installation**:

```bash
go install golang.org/x/tools/cmd/goimports@latest
```

**Configuration**: No configuration file needed

**What it detects**:

- Missing import statements
- Unused import statements
- Incorrect import grouping (stdlib vs third-party)

**Usage**:

```bash
# Check and fix imports
goimports -w .

# Show diff without modifying
goimports -d .
```

**Expected**: All imports organized with standard library imports first, followed by third-party imports

---

### 3. go vet (Standard Static Analysis) ⭐ PRIMARY TOOL

**Purpose**: Detect common mistakes and suspicious constructs in Go code.

**Installation**: Included with Go installation

**Configuration**: No configuration file needed - uses built-in analyzers

**What it detects**:

- Unreachable code
- Unused variables and function parameters
- Printf format string mismatches
- Potential nil pointer dereferences
- Incorrect use of sync types (e.g., copying a mutex)
- Invalid struct tags
- Suspicious boolean conditions
- Incorrect use of testing.T in goroutines

**Usage**:

```bash
# Run on all packages
go vet ./...

# Run on specific package
go vet ./pkg/user

# Enable all analyzers (verbose)
go vet -vettool=$(which vet) ./...
```

**Expected**: Zero issues detected

**Report Location**: Console output (stderr)

---

### 4. staticcheck (Advanced Static Analysis)

**Purpose**: Comprehensive static analysis with hundreds of checks beyond go vet.

**Installation**:

```bash
go install honnef.co/go/tools/cmd/staticcheck@latest
```

**Configuration**: Optional `.staticcheck.conf` file for customization

**What it detects**:

- Inefficient code patterns
- Deprecated API usage
- Unused code (functions, variables, constants)
- Race conditions and goroutine leaks
- Incorrect use of standard library
- Performance issues (e.g., string concatenation in loops)
- Style violations
- Potential bugs

**Usage**:

```bash
# Run on all packages
staticcheck ./...

# Run on specific package
staticcheck ./pkg/user

# Show all checks (including non-critical)
staticcheck -checks=all ./...
```

**Expected**: Zero critical/high issues; medium issues should be fixed or justified

**Report Location**: Console output

**Optional Configuration** (`.staticcheck.conf`):

```toml
checks = ["all", "-ST1000", "-ST1003"]
```

---

### 5. golangci-lint (Comprehensive Linting) ⭐ RECOMMENDED

**Purpose**: Meta-linter that runs multiple linters in parallel with unified configuration.

**Installation**:

```bash
# macOS
brew install golangci-lint

# Linux/macOS/Windows (script)
curl -sSfL https://raw.githubusercontent.com/golangci/golangci-lint/master/install.sh | sh -s -- -b $(go env GOPATH)/bin

# Go install (alternative)
go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest
```

**Configuration**: `.golangci.yml` file in project root

**Minimal Configuration** (`.golangci.yml`):

```yaml
run:
  timeout: 5m
  go: '1.21'

linters:
  enable:
    - gofmt        # Format check
    - goimports    # Import organization
    - govet        # Go vet
    - staticcheck  # Advanced analysis
    - errcheck     # Unchecked errors
    - gosimple     # Simplification suggestions
    - ineffassign  # Ineffectual assignments
    - unused       # Unused code
    - typecheck    # Type checking
    - gocritic     # Go code review comments

linters-settings:
  errcheck:
    check-type-assertions: true
    check-blank: true
  
  gocritic:
    enabled-tags:
      - diagnostic
      - style
      - performance
  
  staticcheck:
    checks: ["all"]

issues:
  exclude-use-default: false
  max-issues-per-linter: 0
  max-same-issues: 0
```

**What it detects**: Combines all of the above tools plus:

- Unchecked error returns (errcheck)
- Ineffectual assignments (ineffassign)
- Unused code (unused)
- Type checking errors (typecheck)
- Code simplifications (gosimple)
- Code review comments (gocritic)

**Usage**:

```bash
# Run with default config
golangci-lint run

# Run on specific path
golangci-lint run ./pkg/...

# Run with auto-fix
golangci-lint run --fix

# Run with all linters
golangci-lint run --enable-all
```

**Expected**: Zero critical/high issues

**Report Location**: Console output (supports JSON, JUnit XML, etc.)

---

### 6. go build (Compilation Check)

**Purpose**: Verify that code compiles successfully.

**Installation**: Included with Go installation

**Usage**:

```bash
# Build all packages
go build ./...

# Build specific package
go build ./cmd/server

# Build with verbose output
go build -v ./...
```

**Expected**: Successful compilation with no errors

---

### 7. go test (Unit Tests)

**Purpose**: Run unit tests and measure code coverage.

**Installation**: Included with Go installation

**Usage**:

```bash
# Run all tests
go test ./...

# Run with verbose output
go test -v ./...

# Run with coverage
go test -cover ./...

# Generate coverage report
go test -coverprofile=coverage.out ./...
go tool cover -html=coverage.out
```

**Expected**: All tests pass, ≥80% coverage

---

## Validation Workflow

### Phase 1: Code Formatting

```bash
gofmt -w .
goimports -w .
```

**Expected**: All files formatted, imports organized

### Phase 2: Static Analysis

```bash
go vet ./...
staticcheck ./...
```

**Expected**: Zero issues detected

### Phase 3: Comprehensive Linting (Recommended)

```bash
golangci-lint run
```

**Expected**: Zero critical/high issues

### Phase 4: Compilation Check

```bash
go build ./...
```

**Expected**: Successful compilation

### Phase 5: Unit Tests

```bash
go test -cover ./...
```

**Expected**: All tests pass, ≥80% coverage

### Phase 6: IDE Errors Check

Use `get_errors` tool to verify no unresolved IDE warnings

**Expected**: Zero unresolved issues

---

## Integration with go-coder-specialist Agent

The agent's Phase 3 Validation includes:

1. **Format check**: `gofmt -l .` → 0 files
2. **Import organization**: `goimports -w .`
3. **Go vet**: `go vet ./...` → 0 issues
4. **Build check**: `go build ./...` → success
5. **Static analysis**: `staticcheck ./...` or `golangci-lint run` → 0 critical/high issues
6. **IDE errors**: `get_errors` tool → 0 unresolved
7. **Unit tests**: `go test -cover ./...` → 100% pass, ≥80% coverage

**All tools must pass with zero violations before proceeding.**

---

## Pre-Delivery Checklist

Before marking any task complete, verify:

- ✅ **gofmt**: `gofmt -l .` shows 0 unformatted files
- ✅ **goimports**: Imports organized (stdlib first, third-party second)
- ✅ **go vet**: `go vet ./...` shows 0 issues
- ✅ **staticcheck**: `staticcheck ./...` shows 0 critical/high issues
- ✅ **golangci-lint** (recommended): `golangci-lint run` passes
- ✅ **go build**: `go build ./...` succeeds
- ✅ **IDE errors**: `get_errors` shows 0 unresolved issues
- ✅ **Unit tests**: `go test -cover ./...` all pass with ≥80% coverage

---

## Priority Levels

**Critical** (MUST fix before submission):

- Nil pointer dereferences
- Data races
- Unchecked critical errors
- Type safety violations
- Format violations (gofmt)

**High** (Fix before code review):

- Unchecked errors
- Unused variables/imports
- Inefficient patterns in hot paths
- Missing godoc for exported items

**Medium** (Fix or justify):

- Minor style violations
- Non-critical inefficiencies
- Simplification suggestions

---

## Example Issues Detected

### gofmt Violations

```text
❌ Before:
func getUserByID(id int)string{
return fmt.Sprintf("user-%d",id)
}

✅ After:
func getUserByID(id int) string {
    return fmt.Sprintf("user-%d", id)
}
```

### go vet Issues

```text
[ERROR] main.go:25: Printf format %s has arg err of wrong type int
```

**Fix**: Use correct format verb `%d` for int

### staticcheck Issues

```text
[ERROR] user.go:45: SA1019: User.Name is deprecated: use User.FullName instead
```

**Fix**: Replace deprecated API usage

```text
[ERROR] handler.go:78: SA4006: this value of err is never used
```

**Fix**: Check error return value

### golangci-lint Issues

```text
[ERROR] server.go:102: Error return value is not checked (errcheck)
    defer resp.Body.Close()
```

**Fix**: Check error or explicitly ignore with `_ = resp.Body.Close()`

```text
[ERROR] config.go:15: ineffectual assignment to err (ineffassign)
    err := loadConfig()
    err = validateConfig()  // Previous err not used
```

**Fix**: Use `if err := loadConfig(); err != nil { ... }`

---

## Common Anti-Patterns Detected

### 1. Unchecked Errors

```go
❌ Bad:
file, _ := os.Open("config.yaml")

✅ Good:
file, err := os.Open("config.yaml")
if err != nil {
    return fmt.Errorf("failed to open config: %w", err)
}
defer file.Close()
```

### 2. Unused Variables

```go
❌ Bad:
result, err := doSomething()
return nil  // result never used

✅ Good:
_, err := doSomething()
return err
```

### 3. Printf Format Mismatches

```go
❌ Bad:
fmt.Printf("User ID: %s", userID)  // userID is int

✅ Good:
fmt.Printf("User ID: %d", userID)
```

### 4. Goroutine Leaks

```go
❌ Bad:
go func() {
    for {
        doWork()  // No exit condition
    }
}()

✅ Good:
ctx, cancel := context.WithCancel(context.Background())
defer cancel()

go func() {
    for {
        select {
        case <-ctx.Done():
            return
        default:
            doWork()
        }
    }
}()
```

### 5. Inefficient String Concatenation

```go
❌ Bad:
var result string
for i := 0; i < 1000; i++ {
    result += strconv.Itoa(i) + ","
}

✅ Good:
var builder strings.Builder
for i := 0; i < 1000; i++ {
    builder.WriteString(strconv.Itoa(i))
    builder.WriteString(",")
}
result := builder.String()
```

---

## References

- [Effective Go](https://go.dev/doc/effective_go) - Official Go best practices
- [Go Code Review Comments](https://github.com/golang/go/wiki/CodeReviewComments) - Common review comments
- [staticcheck Documentation](https://staticcheck.io/docs/)
- [golangci-lint Documentation](https://golangci-lint.run/)
- [Go Standard Project Layout](https://github.com/golang-standards/project-layout)
