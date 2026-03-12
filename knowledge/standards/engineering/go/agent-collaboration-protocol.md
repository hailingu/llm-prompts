# Go Collaboration Supplement

This file is a Go-specific supplement.

Canonical protocol:

- `knowledge/standards/common/agent-collaboration-protocol.md`

## Go-Specific Additions

1. Required static checks at Gate 2 include `gofmt`, `go vet`, and project-selected lint/static tools.
2. Review stage should explicitly verify goroutine safety where concurrency is involved.
3. Error handling conventions should follow repository Go standards.
