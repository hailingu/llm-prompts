"""Visual renderers - plugin architecture."""

from .base import BaseRenderer

# Re-export render_slide from parent renderers.py module for backward compatibility
import sys
import importlib

_parent_renderers = None

def __getattr__(name):
    """Lazy load render_slide and other functions from parent renderers.py"""
    global _parent_renderers
    if _parent_renderers is None:
        # Import the actual renderers.py file (sibling module)
        parent_package = '.'.join(__name__.split('.')[:-1])
        # Import ppt_generator.renderers_legacy (which is renderers.py renamed)
        # Actually, we need to import from the parent module
        parent_module_name = f'{parent_package}.renderers_legacy'
        try:
            _parent_renderers = importlib.import_module(parent_module_name)
        except ModuleNotFoundError:
            # If not renamed, try accessing via parent's __dict__
            raise AttributeError(f"Cannot find legacy renderers module")
    
    if hasattr(_parent_renderers, name):
        return getattr(_parent_renderers, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
