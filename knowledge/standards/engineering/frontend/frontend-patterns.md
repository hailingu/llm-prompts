# Frontend Patterns (Reference)

Detailed code examples for frontend patterns. See agent files for DO/DON'T guidelines.

### Request State Model

```ts
export type RequestState<T> =
  | { status: 'idle' }
  | { status: 'loading' }
  | { status: 'success'; data: T }
  | { status: 'error'; message: string; retryable: boolean };
```

### Boundary Normalization

```ts
export function normalizeUser(raw: ApiUser): UserViewModel {
  return {
    id: raw.id,
    name: raw.name?.trim() || 'Unknown',
    joinedAt: new Date(raw.joined_at).toISOString(),
  };
}
```

### Async Cancellation Guard

```ts
let active = true;
runAsync()
  .then((result) => {
    if (!active) return;
    apply(result);
  })
  .catch((err) => {
    if (!active) return;
    fail(err);
  });

active = false;
```

### 4) Error Boundary with Fallback

```tsx
class ErrorBoundary extends React.Component<Props, State> {
  state = { hasError: false, error: null };

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  render() {
    if (this.state.hasError) {
      return <FallbackComponent error={this.state.error} />;
    }
    return this.props.children;
  }
}
```

### 5) Composable Component Pattern

```tsx
// Prefer composition over prop drilling
function Card({ header, body, footer }) {
  return (
    <div className="card">
      {header && <div className="card-header">{header}</div>}
      <div className="card-body">{body}</div>}
      {footer && <div className="card-footer">{footer}</div>}
    </div>
  );
}

// Usage
<Card header={<Title />} body={<Content />} footer={<Actions />} />
```

### 6) Hook for Async Data Fetching

```ts
function useAsyncData<T>(fetchFn: () => Promise<T>, deps: DependencyList) {
  const [state, setState] = useState<RequestState<T>>({ status: 'idle' });

  useEffect(() => {
    let cancelled = false;
    setState({ status: 'loading' });

    fetchFn()
      .then((data) => {
        if (!cancelled) setState({ status: 'success', data });
      })
      .catch((error) => {
        if (!cancelled) setState({ status: 'error', message: error.message, retryable: true });
      });

    return () => { cancelled = true; };
  }, deps);

  return state;
}
```

## Pattern Storage

- Reusable patterns -> `memory/research/frontend_coding.md`
- Bugs/pitfalls -> `memory/research/frontend_coding.md`
- Cross-project insights -> `memory/global.md` "## Patterns"
