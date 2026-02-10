"""Public package surface for ppt_generator.

This package provides a single entrypoint `generate_pptx` that the CLI
wrapper in `bin/generate_pptx.py` imports and calls.
"""
from .cli import generate_pptx

__all__ = ["generate_pptx"]
