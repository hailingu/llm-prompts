"""Constants and configuration for PPT Visual QA."""

# Default runtime profiles for testing
PROFILES_DEFAULT = [
    {"id": "P1", "width": 1280, "height": 720, "dpr": 1},
    {"id": "P2", "width": 1366, "height": 768, "dpr": 1},
    {"id": "P3", "width": 1512, "height": 982, "dpr": 2},
]

# Main content area available height in pixels
MAIN_OUTER_AVAILABLE_PX = 590.0

# Tailwind spacing mappings
TAILWIND_MB_MAP = {2: 8, 3: 12, 4: 16, 6: 24, 8: 32}
TAILWIND_GAP_MAP = {2: 8, 3: 12, 4: 16, 5: 20, 6: 24, 8: 32}

# Semantic color patterns for detection
SEMANTIC_COLORS = [
    "text-red", "text-green", "text-yellow", "text-orange", "text-blue",
    "bg-red", "bg-green", "bg-yellow", "bg-orange", "bg-blue",
    "border-red", "border-green", "border-yellow", "border-orange", "border-blue",
    "brand-primary", "brand-secondary", "brand-accent",
    "badge-primary", "badge-success", "badge-warning", "badge-danger",
    "badge-risk", "badge-warn", "badge-info", "badge-stage",
    "text-success", "text-warning", "text-danger",
    "bg-success", "bg-warning", "bg-danger",
    "border-success", "border-warning", "border-danger",
    "var(--risk)", "var(--warn)", "var(--info)", "var(--success)", "var(--stage)",
    "var(--brand-primary)", "var(--brand-secondary)", "var(--brand-accent)",
]