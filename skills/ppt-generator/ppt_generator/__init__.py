"""Public package surface for ppt_generator.

This package provides a single entrypoint `generate_pptx` that the CLI
wrapper in `bin/generate_pptx.py` imports and calls.

To avoid circular import issues during development, we use lazy imports.
"""

__version__ = '0.2.0-plugin-architecture'

def generate_pptx(*args, **kwargs):
    """Lazy-loaded generate_pptx function."""
    from .cli import generate_pptx as _generate_pptx
    return _generate_pptx(*args, **kwargs)

__all__ = ["generate_pptx", "__version__"]
