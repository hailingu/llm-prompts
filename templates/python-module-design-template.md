# [Module Name] Design Document

**Author**: [Your Name]  
**Date**: [YYYY-MM-DD]  
**Status**: Draft | In Review | Approved  
**Reviewers**: @python-architect, @python-api-designer

---

## 1. Context and Scope

### 1.1 Problem Statement

**What problem are we solving?**

[Describe the specific problem this module addresses. Be concrete and specific.]

**Why is this important?**

[Explain the business or technical value. What happens if we don't solve this?]

### 1.2 Target Users

**Who will use this module?**

- [ ] Internal services (which ones?)
- [ ] External API consumers
- [ ] CLI tools
- [ ] Other: [specify]

### 1.3 System Boundary

**What systems will this module interact with?**

- **Upstream dependencies**: [List services/databases/APIs this module calls]
- **Downstream consumers**: [List services/components that will call this module]

### 1.4 Out of Scope

**What will NOT be included in this design?**

- [Explicitly list features or use cases that are out of scope]
- [This helps prevent scope creep]

---

## 2. Goals and Non-Goals

### 2.1 Goals

**What are the measurable success criteria?**

- [ ] Goal 1: [e.g., Support 1000 requests per second with p95 latency < 100ms]
- [ ] Goal 2: [e.g., Reduce error rate from 5% to < 0.1%]
- [ ] Goal 3: [e.g., Enable feature X for 100% of users by Q2]

### 2.2 Non-Goals

**What are we explicitly NOT trying to achieve?**

- [e.g., NOT supporting protocol Y (will be added in v2)]
- [e.g., NOT optimizing for batch processing (focus is on real-time)]

---

## 3. Design Overview

### 3.1 Architecture Diagram

**High-level component view:**

```mermaid
graph LR
    Client[Client/Consumer] --> API[Module API]
    API --> Service[Core Service]
    Service --> Cache[(Cache)]
    Service --> DB[(Database)]
    Service --> External[External API]
```

### 3.2 Component Responsibilities

| Component        | Responsibility                         | Technology                    |
| ---------------- | -------------------------------------- | ----------------------------- |
| API Layer        | HTTP handlers, request validation      | FastAPI, Pydantic             |
| Service Layer    | Business logic, orchestration          | Python stdlib, async/await    |
| Repository Layer | Data access, caching                   | SQLAlchemy 2.0, Redis        |
| Client Layer     | External API integration               | httpx                         |

### 3.3 Technology Stack

- **Python Version**: 3.12+ (or specify minimum version)
- **Framework**: FastAPI / Django / Flask (specify)
- **ORM / Data Access**: SQLAlchemy 2.0 (async), raw SQL, or other
- **Database**: PostgreSQL, MySQL, or NoSQL
- **Cache**: Redis, in-memory, or none
- **Messaging**: Celery, RabbitMQ, Kafka, or none
- **Observability**: OpenTelemetry, Prometheus, structlog
- **Package Manager**: uv / pip / poetry

---

## 4. API Design Guidelines

### 4.1 Error Handling Strategy

**How will errors be communicated to callers?**

- **Domain Errors** (business logic failures):
  - Raise custom exception types (e.g., `UserNotFoundError`, `InvalidInputError`)
  - Exception chaining for wrapped errors: `raise XxxError(msg) from e`
  - HTTP: Map to appropriate status codes (400, 404, 409)

- **Infrastructure Errors** (system failures):
  - Raise custom exception types with chaining
  - Examples: `DatabaseError`, `ExternalServiceError`
  - HTTP: Map to 500 or 503

- **Error Response Format** (for HTTP APIs):

  ```json
  {
    "error": {
      "code": "USER_NOT_FOUND",
      "message": "User with ID 123 not found",
      "details": {}
    }
  }
  ```

- **Exception Hierarchy**:

  ```python
  class AppError(Exception):
      """Base exception for all application errors."""

  class UserNotFoundError(AppError):
      """Raised when user is not found."""

  class InvalidInputError(AppError):
      """Raised when input validation fails."""

  class DatabaseError(AppError):
      """Raised when database operations fail."""
  ```

### 4.2 API Versioning Strategy

**How will we handle breaking changes?**

- [ ] URL versioning (e.g., `/api/v1/users`, `/api/v2/users`)
- [ ] Package versioning (e.g., `myproject.v2`)
- [ ] Header versioning (e.g., `Accept: application/vnd.api+json; version=1`)

**Recommendation**: Use URL versioning with FastAPI router prefixes.

### 4.3 Authentication/Authorization Strategy

**How will callers authenticate?**

- [ ] API Key (via header or query parameter)
- [ ] JWT (JSON Web Token)
- [ ] OAuth 2.0
- [ ] None (internal-only service)

### 4.4 API Overview

**What are the main operations?** (High-level only; detailed signatures in Section 10)

- `get_user_by_id(user_id: str) -> User`: Retrieve user by ID
- `create_user(user: CreateUserRequest) -> User`: Create a new user
- `update_user(user_id: str, data: UpdateUserRequest) -> User`: Update existing user
- `delete_user(user_id: str) -> None`: Delete user by ID

---

## 5. Data Model Overview

### 5.1 Key Entities

**What are the main domain objects?**

- **User**: Represents a system user
- **Subscription**: Represents a user subscription
- **Order**: Represents a purchase order

**Entity Relationships**:

- User has many Subscriptions (1:N)
- Subscription belongs to User (N:1)
- Order references User and Subscription (N:1, N:1)

**Note**: Detailed field definitions are in Section 11.

---

## 6. Concurrency Requirements Overview

### 6.1 Performance Targets

**What are the expected throughput and latency?**

- **Expected QPS**: 1000 requests per second (average), 2000 (peak)
- **Response Time**:
  - p50 < 50ms
  - p95 < 100ms
  - p99 < 200ms

### 6.2 Concurrency Strategy

**Which components need to be thread-safe?**

| Component       | Thread-Safe? | Strategy                                    | Async? |
| --------------- | ------------ | ------------------------------------------- | ------ |
| UserService     | Yes          | Stateless (no shared mutable state)         | Yes    |
| ConfigLoader    | Yes          | Immutable after initialization              | No     |
| Cache           | Yes          | Use async Redis client or threading.Lock    | Yes    |
| PeriodicChecker | No           | Single asyncio task                         | Yes    |

**GIL Considerations**:

- I/O-bound work: Use `asyncio` (GIL released during I/O)
- CPU-bound work: Use `multiprocessing` or `concurrent.futures.ProcessPoolExecutor`
- Mixed workload: Use `asyncio.to_thread()` or `loop.run_in_executor()`

**Note**: Detailed concurrency contracts are in Section 12.

---

## 7. Cross-Cutting Concerns

### 7.1 Observability

**Logging Strategy**:

- Use structured logging (`structlog` or `logging` with JSON formatter)
- Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
- Include correlation IDs for request tracing
- Never log sensitive data (tokens, passwords, PII)

**Metrics**:

- Request count, latency, error rate
- Resource usage (memory, active tasks)
- Use Prometheus client (`prometheus-client`)

**Tracing**:

- OpenTelemetry for distributed tracing
- Trace all external calls (DB, HTTP, message queue)

### 7.2 Security

**Threat Model**:

- Input validation (prevent injection attacks)
- Rate limiting (prevent abuse)
- Authentication/Authorization (see Section 4.3)

**Mitigation**:

- Validate all inputs with Pydantic models
- Use parameterized queries (SQLAlchemy) — never f-strings in SQL
- Implement rate limiting at API gateway or middleware level
- Run `bandit` for security scanning

### 7.3 Reliability

**Error Handling**:

- All exceptions must be caught and handled
- Use exponential backoff for retries
- Circuit breaker for external dependencies

**Retry Strategy**:

- Idempotent operations: retry up to 3 times (via `tenacity`)
- Non-idempotent operations: no retry (or use idempotency keys)

---

## 8. Implementation Constraints

### 8.1 Framework Constraints

**Must use**:

- Python 3.12+ with type annotations throughout
- FastAPI / Django / Flask (specify)
- Pydantic v2 for data validation
- SQLAlchemy 2.0 with async sessions (if using ORM)

**Must NOT use**:

- `type: ignore` without specific error code
- Mutable default arguments
- Global mutable state
- Bare `except:` clauses
- `import *`

### 8.2 Coding Standards

**Must follow**:

- PEP 8 for style (enforced by ruff)
- PEP 484 for type hints (enforced by mypy --strict)
- PEP 257 for docstrings (Google style)
- ruff format for formatting (enforced by CI)
- All public APIs must have docstrings

---

## 9. Alternatives Considered

### Alternative 1: [Option Name]

**Description**: [Brief description of alternative approach]

**Pros**:

- [Advantage 1]
- [Advantage 2]

**Cons**:

- [Disadvantage 1]
- [Disadvantage 2]

**Decision**: Rejected because [specific reason].

### Alternative 2: [Option Name]

**Description**: [Brief description of alternative approach]

**Pros**:

- [Advantage 1]

**Cons**:

- [Disadvantage 1]

**Decision**: Rejected because [specific reason].

---

## 10. API Specification (Detailed)

### 10.1 Interface Definitions

**Complete Python Protocol definitions with docstrings and type annotations:**

```python
"""User service interfaces."""

from typing import Protocol, runtime_checkable

from myproject.models import User


@runtime_checkable
class UserService(Protocol):
    """Provides user management operations.

    Implementations must be safe for concurrent use from multiple
    async tasks or threads.
    """

    async def get_user_by_id(self, user_id: str) -> User:
        """Retrieve a user by their unique identifier.

        Args:
            user_id: User ID (must be non-empty UUID v4).

        Returns:
            User object if found.

        Raises:
            UserNotFoundError: If user doesn't exist.
            InvalidInputError: If user_id is empty or invalid.
            DatabaseError: If database operation fails (retryable).

        Thread-safety: Yes (stateless implementation)
        Idempotent: Yes
        """
        ...

    async def create_user(self, user: "CreateUserRequest") -> User:
        """Create a new user.

        Args:
            user: User creation request with required fields.

        Returns:
            Created user with generated ID.

        Raises:
            DuplicateUserError: If user with same email exists.
            InvalidInputError: If validation fails.
            DatabaseError: If database operation fails.

        Thread-safety: Yes
        Idempotent: No (use idempotency key if retry is needed)
        """
        ...

    async def update_user(
        self, user_id: str, data: "UpdateUserRequest"
    ) -> User:
        """Update an existing user.

        Args:
            user_id: ID of the user to update.
            data: Fields to update.

        Returns:
            Updated user object.

        Raises:
            UserNotFoundError: If user doesn't exist.
            InvalidInputError: If validation fails.
            DatabaseError: If database operation fails.

        Thread-safety: Yes
        Idempotent: Yes
        """
        ...
```

### 10.2 Design Rationale

#### 10.2.1 Contract Precision

**Contract Table** (defines exact behavior for all scenarios):

| Scenario       | Input           | Return Value | Exception                                  | HTTP Status | Retry?   | Pattern                 |
| -------------- | --------------- | ------------ | ------------------------------------------ | ----------- | -------- | ----------------------- |
| Success        | Valid ID        | User         | (none)                                     | 200         | No       | —                       |
| Not Found      | Valid ID        | N/A          | UserNotFoundError(user_id)                 | 404         | No       | Custom exception        |
| Invalid ID     | Empty/malformed | N/A          | InvalidInputError(detail)                  | 400         | No       | Validation exception    |
| DB Timeout     | Valid ID        | N/A          | DatabaseError(msg, cause=e) from e         | 503         | Yes (3x) | Exception chaining      |
| DB Unavailable | Valid ID        | N/A          | DatabaseError(msg) from e                  | 503         | Yes (3x) | Exception chaining      |

**Exception Types** (defined as class hierarchy):

```python
class AppError(Exception):
    """Base exception for all application errors."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class UserNotFoundError(AppError):
    """Raised when the requested user does not exist."""

    def __init__(self, user_id: str) -> None:
        self.user_id = user_id
        super().__init__(f"User not found: {user_id}")


class DuplicateUserError(AppError):
    """Raised when a user with the same email already exists."""

    def __init__(self, email: str) -> None:
        self.email = email
        super().__init__(f"User already exists: {email}")


class InvalidInputError(AppError):
    """Raised when input validation fails."""

    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(f"Invalid input: {detail}")


class DatabaseError(AppError):
    """Raised when database operations fail."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
```

#### 10.2.2 Caller Guidance (Executable Code Example)

**Complete example showing exception handling, retries, logging, and FastAPI integration:**

```python
"""
get_user_with_retry demonstrates proper usage of UserService.get_user_by_id
with exception handling, retries, logging, and HTTP status mapping.
"""

import asyncio

import structlog
from fastapi import FastAPI, HTTPException

from myproject.exceptions import (
    DatabaseError,
    InvalidInputError,
    UserNotFoundError,
)
from myproject.models import User
from myproject.service import UserService

logger = structlog.get_logger()


async def get_user_with_retry(
    svc: UserService,
    user_id: str,
    *,
    max_retries: int = 3,
    initial_delay: float = 0.1,
    backoff_factor: float = 2.0,
) -> User:
    """Get user with automatic retry for infrastructure errors.

    Args:
        svc: User service instance.
        user_id: User ID to look up.
        max_retries: Maximum number of retry attempts.
        initial_delay: Initial delay in seconds between retries.
        backoff_factor: Multiplier for exponential backoff.

    Returns:
        User object if found.

    Raises:
        UserNotFoundError: If user doesn't exist.
        InvalidInputError: If user_id is invalid.
        DatabaseError: If all retries are exhausted.
    """
    # Input validation
    if not user_id:
        raise InvalidInputError("user_id must not be empty")

    delay = initial_delay
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            user = await svc.get_user_by_id(user_id)
            logger.info("user_retrieved", user_id=user_id, attempt=attempt + 1)
            return user
        except (UserNotFoundError, InvalidInputError):
            # Business errors: do NOT retry
            raise
        except DatabaseError as e:
            last_error = e
            if attempt < max_retries:
                logger.warning(
                    "retrying_after_error",
                    error=str(e),
                    user_id=user_id,
                    attempt=attempt + 1,
                    next_delay=delay,
                )
                await asyncio.sleep(delay)
                delay *= backoff_factor
                continue

            logger.error(
                "user_operation_failed",
                error=str(e),
                user_id=user_id,
                attempts=attempt + 1,
            )
            raise

    # Unreachable, but satisfies type checker
    assert last_error is not None
    raise last_error


# FastAPI endpoint with HTTP status code mapping
app = FastAPI()


@app.get("/users/{user_id}")
async def handle_get_user(user_id: str) -> User:
    """HTTP handler with exception-to-status-code mapping."""
    try:
        return await get_user_with_retry(user_service, user_id)
    except UserNotFoundError:
        raise HTTPException(status_code=404, detail="User not found")
    except InvalidInputError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except DatabaseError:
        raise HTTPException(status_code=503, detail="Service unavailable")
```

#### 10.2.3 Rationale

**Why this design was chosen:**

- **Error handling**: Uses custom exception hierarchy; specific types allow precise `except` blocks
- **Exception chaining**: `raise XxxError(msg) from e` preserves original traceback for debugging
- **Retry strategy**: Exponential backoff for infrastructure errors, no retry for business errors
- **Thread-safety**: Stateless service design — no shared mutable state, safe for async concurrency
- **Type safety**: All public APIs fully typed, enforced by mypy --strict

**Trade-offs**:

- Chose custom exceptions over return codes for Pythonic EAFP pattern
- Chose stateless service to maximize concurrency (no locks needed)
- Chose async/await for I/O-bound operations (database, HTTP calls)
- Chose Protocol over ABC for structural typing flexibility

#### 10.2.4 Alternatives Considered

##### Alternative 1: Return `User | None` instead of raising `UserNotFoundError`

**Pros**: Simpler for "not found" case, no try/except needed  
**Cons**: Caller must remember to check None; cannot carry error context  
**Decision**: Rejected; EAFP is Pythonic, and exceptions carry context (user_id)

##### Alternative 2: Use ABC instead of Protocol

**Pros**: Can provide base implementation; enforces inheritance  
**Cons**: Requires explicit inheritance; less flexible for testing  
**Decision**: Rejected; Protocol allows structural typing — any class with matching methods works

### 10.3 Dependency Interfaces

**External dependencies that this module requires:**

```python
"""Repository interfaces for data access."""

from typing import Protocol, runtime_checkable

from myproject.models import User


@runtime_checkable
class UserRepository(Protocol):
    """Provides data access for user persistence.

    Implementations must be safe for concurrent use.
    """

    async def find_by_id(self, user_id: str) -> User | None:
        """Retrieve a user by ID from the data store.

        Args:
            user_id: User ID to look up.

        Returns:
            User object if found, None if not found.

        Raises:
            DatabaseError: If database operation fails.

        Thread-safety: Yes
        """
        ...

    async def save(self, user: User) -> User:
        """Persist a user to the data store.

        Args:
            user: User object to save. ID will be generated if empty.

        Returns:
            Saved user with generated ID.

        Raises:
            DuplicateUserError: If user with same email exists.
            DatabaseError: If database operation fails.

        Thread-safety: Yes
        """
        ...

    async def update(self, user: User) -> User:
        """Update an existing user in the data store.

        Args:
            user: User object with updated fields.

        Returns:
            Updated user object.

        Raises:
            UserNotFoundError: If user doesn't exist.
            DatabaseError: If database operation fails.

        Thread-safety: Yes
        """
        ...
```

---

## 11. Data Model (Detailed)

### 11.1 User Entity

```python
"""User domain models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from uuid import uuid4

from myproject.exceptions import InvalidInputError


class UserStatus(StrEnum):
    """Valid user account statuses."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"


@dataclass
class User:
    """Represents a system user.

    Attributes:
        id: Unique identifier (UUID v4). Auto-generated if not provided.
        email: User's email address. Required, max 255 chars.
        name: User's display name. Optional, max 100 chars.
        status: Account status. Default: ACTIVE.
        created_at: Timestamp when user was created. Auto-set on creation.
        updated_at: Timestamp of last update. Auto-updated on modification.
    """

    email: str
    name: str = ""
    id: str = field(default_factory=lambda: str(uuid4()))
    status: UserStatus = UserStatus.ACTIVE
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    updated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def validate(self) -> None:
        """Validate field values.

        Raises:
            InvalidInputError: If any field is invalid.
        """
        if not self.email:
            raise InvalidInputError("email is required")
        if len(self.email) > 255:
            raise InvalidInputError("email too long (max 255 chars)")
        if len(self.name) > 100:
            raise InvalidInputError("name too long (max 100 chars)")
```

### 11.2 Pydantic Schemas (API layer)

```python
"""API request/response schemas."""

from pydantic import BaseModel, EmailStr, Field


class CreateUserRequest(BaseModel):
    """Request schema for creating a user."""

    email: EmailStr
    name: str = Field(default="", max_length=100)


class UpdateUserRequest(BaseModel):
    """Request schema for updating a user."""

    email: EmailStr | None = None
    name: str | None = Field(default=None, max_length=100)


class UserResponse(BaseModel):
    """Response schema for user data."""

    model_config = {"from_attributes": True}

    id: str
    email: str
    name: str
    status: str
    created_at: str
    updated_at: str
```

### 11.3 Database Schema

**PostgreSQL table definition:**

```sql
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) NOT NULL UNIQUE,
    name VARCHAR(100),
    status VARCHAR(20) NOT NULL DEFAULT 'active',
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_status ON users(status);
```

**SQLAlchemy Model:**

```python
from sqlalchemy import String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class UserRow(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(100), default="")
    status: Mapped[str] = mapped_column(String(20), default="active")
```

---

## 12. Concurrency Requirements (Detailed)

### 12.1 Thread-Safety Contracts

**Per-method concurrency requirements:**

| Method          | Thread-Safe? | Async? | Expected QPS | Response Time Target | Synchronization Strategy           |
| --------------- | ------------ | ------ | ------------ | -------------------- | ---------------------------------- |
| get_user_by_id  | Yes          | Yes    | 500          | p95 < 100ms          | Stateless (no sync needed)         |
| create_user     | Yes          | Yes    | 50           | p95 < 200ms          | Stateless (DB handles concurrency) |
| update_user     | Yes          | Yes    | 100          | p95 < 150ms          | Stateless (DB handles concurrency) |

### 12.2 Concurrency Strategy

**Design Pattern**: Stateless async service

**Why stateless async?**

- No shared mutable state → no locks needed
- Each request is independent
- Async I/O avoids blocking and GIL contention for I/O-bound work
- Horizontally scalable with uvicorn workers

**Implementation Notes**:

- UserService implementation has no instance fields (or only read-only fields set in `__init__`)
- All state is in the database or cache (which handle their own concurrency)
- Use async connection pooling (SQLAlchemy async engine with `pool_size`)

### 12.3 Connection Pooling

**Database connection pool configuration:**

```python
from sqlalchemy.ext.asyncio import create_async_engine

engine = create_async_engine(
    "postgresql+asyncpg://user:pass@localhost/dbname",
    pool_size=10,         # Max persistent connections
    max_overflow=5,       # Extra connections under load
    pool_timeout=30,      # Seconds to wait for available connection
    pool_recycle=1800,    # Recycle connections after 30 minutes
)
```

**Rationale**: 10 connections can support ~500 QPS at 20ms avg query time

---

## 13. Testing Strategy

### 13.1 Unit Tests

**Coverage Target**: Minimum 80% for business logic

**Test Structure**: Use pytest with `@pytest.mark.parametrize` and fixtures

```python
import pytest

from myproject.exceptions import InvalidInputError, UserNotFoundError
from myproject.models import User
from myproject.service import UserServiceImpl


class TestGetUserByID:
    """Tests for UserService.get_user_by_id."""

    @pytest.fixture()
    def mock_repo(self, mocker):
        return mocker.AsyncMock()

    @pytest.fixture()
    def svc(self, mock_repo):
        return UserServiceImpl(repo=mock_repo)

    @pytest.mark.asyncio()
    async def test_returns_user_when_found(self, svc, mock_repo):
        expected = User(email="test@example.com", id="abc-123")
        mock_repo.find_by_id.return_value = expected

        result = await svc.get_user_by_id("abc-123")

        assert result == expected
        mock_repo.find_by_id.assert_called_once_with("abc-123")

    @pytest.mark.asyncio()
    async def test_raises_not_found_when_missing(self, svc, mock_repo):
        mock_repo.find_by_id.return_value = None

        with pytest.raises(UserNotFoundError, match="nonexistent"):
            await svc.get_user_by_id("nonexistent")

    @pytest.mark.asyncio()
    @pytest.mark.parametrize("invalid_id", ["", "   "])
    async def test_raises_invalid_input_for_empty_id(
        self, svc, invalid_id
    ):
        with pytest.raises(InvalidInputError):
            await svc.get_user_by_id(invalid_id)
```

### 13.2 Integration Tests

**Scope**: Test with real database (use testcontainers or Docker fixtures)

**Example**:

```python
import pytest
from testcontainers.postgres import PostgresContainer

from myproject.repository import PostgresUserRepository


@pytest.fixture(scope="module")
def postgres():
    with PostgresContainer("postgres:16") as pg:
        yield pg


@pytest.fixture()
async def repo(postgres):
    engine = create_async_engine(postgres.get_connection_url())
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return PostgresUserRepository(engine)


@pytest.mark.integration()
@pytest.mark.asyncio()
async def test_create_and_retrieve_user(repo):
    user = User(email="test@example.com", name="Test User")
    saved = await repo.save(user)

    retrieved = await repo.find_by_id(saved.id)
    assert retrieved is not None
    assert retrieved.email == "test@example.com"
```

### 13.3 Performance Tests

**Performance validation with pytest-benchmark:**

```python
import pytest


@pytest.mark.benchmark()
def test_user_validation_performance(benchmark):
    user = User(email="test@example.com", name="Test User")
    benchmark(user.validate)
```

---

## 14. Deployment Considerations

### 14.1 Configuration

**Environment variables (via Pydantic BaseSettings):**

```python
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "myapp"
    db_user: str = "postgres"
    db_password: str = ""
    redis_url: str = "redis://localhost:6379"
    log_level: str = "INFO"
    http_port: int = 8000

    model_config = {"env_prefix": "APP_"}
```

### 14.2 Health Checks

**Endpoints**:

- `/healthz`: Liveness probe (always returns 200 if server is running)
- `/readyz`: Readiness probe (checks DB and cache connectivity)

```python
@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/readyz")
async def readyz(db: AsyncSession = Depends(get_db)):
    try:
        await db.execute(text("SELECT 1"))
        return {"status": "ready"}
    except Exception:
        raise HTTPException(status_code=503, detail="Not ready")
```

### 14.3 Graceful Shutdown

**Implementation with FastAPI lifespan:**

```python
from contextlib import asynccontextmanager

from fastapi import FastAPI


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize resources
    engine = create_async_engine(settings.database_url)
    app.state.engine = engine
    yield
    # Shutdown: cleanup resources
    await engine.dispose()


app = FastAPI(lifespan=lifespan)
```

---

## 15. Project Structure

**Recommended Python project layout:**

```text
myproject/
├── src/
│   └── myproject/
│       ├── __init__.py
│       ├── __main__.py           # CLI entry point
│       ├── app.py                # FastAPI application factory
│       ├── settings.py           # Configuration (Pydantic BaseSettings)
│       ├── exceptions.py         # Custom exception hierarchy
│       ├── user/
│       │   ├── __init__.py
│       │   ├── models.py         # User dataclass + Pydantic schemas
│       │   ├── protocols.py      # UserService, UserRepository Protocols
│       │   ├── service.py        # UserServiceImpl
│       │   ├── repository.py     # PostgresUserRepository
│       │   └── router.py         # FastAPI router (HTTP handlers)
│       └── shared/
│           ├── __init__.py
│           ├── logging.py        # structlog configuration
│           └── middleware.py      # FastAPI middleware
├── tests/
│   ├── conftest.py               # Shared fixtures
│   ├── test_user_service.py      # Unit tests
│   └── test_user_integration.py  # Integration tests
├── migrations/
│   ├── env.py                    # Alembic environment
│   └── versions/
│       └── 001_create_users.py
├── pyproject.toml                # Project metadata + tool config
├── Dockerfile
├── Makefile                      # Build and test commands
└── README.md
```

**Package Guidelines**:

- `src/myproject/`: Application code (src layout for proper packaging)
- `src/myproject/*/protocols.py`: Protocol (interface) definitions
- `src/myproject/*/service.py`: Business logic implementations
- `src/myproject/*/repository.py`: Data access implementations
- `src/myproject/*/router.py`: HTTP route handlers
- `tests/`: All tests (mirror source structure)
- `migrations/`: Alembic database migrations

---

## Appendix A: References

- [PEP 8](https://peps.python.org/pep-0008/) - Style Guide
- [PEP 484](https://peps.python.org/pep-0484/) - Type Hints
- [PEP 257](https://peps.python.org/pep-0257/) - Docstring Conventions
- [PEP 544](https://peps.python.org/pep-0544/) - Protocols
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [Google Design Docs Standards](.github/standards/google-design-doc-standards.md)

---

## Appendix B: Review Checklist

**Before submitting for review, verify:**

- [ ] All sections are complete (no "[TODO]" placeholders)
- [ ] API Protocols have complete docstrings with Args, Returns, Raises
- [ ] Contract table covers all exception scenarios
- [ ] Caller Guidance includes 50-100 lines of executable code
- [ ] Concurrency strategy is clearly defined (async/sync, GIL considerations)
- [ ] All public types and functions are documented
- [ ] Exception types are defined as class hierarchy
- [ ] Test strategy is specified (unit, integration, benchmark)
- [ ] Project structure follows src layout
- [ ] No Java/Go-specific patterns (e.g., getters/setters, sentinel errors, goroutines)

---

## End of Template
