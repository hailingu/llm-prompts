"""Renderer registry - auto-discovers and manages all available renderers."""

import logging
from typing import List, Optional, Dict, Any
from pathlib import Path
import importlib
import inspect

from ..protocols.renderer_interface import IRenderer
from ..protocols.visual_data_protocol import VisualDataProtocol


logger = logging.getLogger(__name__)


class RendererRegistry:
    """Central registry for all visual renderers.
    
    Automatically discovers renderers by scanning the renderers/ directory,
    checks their availability, and selects the best renderer for each request.
    """
    
    def __init__(self):
        """Initialize registry and discover all renderers."""
        self._renderers: List[IRenderer] = []
        self._type_cache: Dict[str, List[IRenderer]] = {}
        self._discover_renderers()
    
    def _discover_renderers(self):
        """Automatically discover all renderer classes in renderers/ directory."""
        try:
            # Dynamically import renderer modules
            renderers_path = Path(__file__).parent.parent / 'renderers'
            
            if not renderers_path.exists():
                logger.warning(f"Renderers directory not found: {renderers_path}")
                return
            
            # Import native renderers
            self._import_renderers_from_module('ppt_generator.renderers.native')
            
            # Import mermaid renderers
            self._import_renderers_from_module('ppt_generator.renderers.mermaid')
            
            # Future: Import matplotlib, etc.
            # self._import_renderers_from_module('ppt_generator.renderers.matplotlib')
            
            # Log discovered renderers
            available = [r for r in self._renderers if r.is_available()]
            logger.info(f"Discovered {len(self._renderers)} renderers, {len(available)} available")
            for r in available:
                logger.debug(f"  ✓ {r.name}: {r.supported_types}")
            
        except Exception as e:
            logger.error(f"Failed to discover renderers: {e}", exc_info=True)
    
    def _import_renderers_from_module(self, module_path: str):
        """Import all renderer classes from a module."""
        try:
            module = importlib.import_module(module_path)
            
            # Find all IRenderer subclasses
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (issubclass(obj, IRenderer) and 
                    obj is not IRenderer and
                    not inspect.isabstract(obj)):
                    try:
                        renderer = obj()
                        self._renderers.append(renderer)
                        logger.debug(f"Registered renderer: {renderer.name}")
                    except Exception as e:
                        logger.warning(f"Failed to instantiate {name}: {e}")
                        
        except ImportError as e:
            logger.debug(f"Module {module_path} not found: {e}")
        except Exception as e:
            logger.error(f"Error importing from {module_path}: {e}", exc_info=True)
    
    def get_renderers_for_type(self, visual_type: str) -> List[IRenderer]:
        """Get all available renderers for a visual type, sorted by quality.
        
        Args:
            visual_type: Visual type identifier (e.g., 'gantt', 'flowchart')
            
        Returns:
            List[IRenderer]: Available renderers sorted by quality (best first)
        """
        # Check cache first
        cache_key = visual_type.lower()
        if cache_key in self._type_cache:
            return self._type_cache[cache_key]
        
        # Find matching renderers
        candidates = [
            r for r in self._renderers
            if r.is_available() and r.can_render(visual_type)
        ]
        
        # Sort by availability and quality (descending)
        # Note: We can't estimate quality without data, so we'll sort later during selection
        self._type_cache[cache_key] = candidates
        return candidates
    
    def select_renderer(
        self,
        visual_data: VisualDataProtocol,
        quality_preference: str = 'best_available'
    ) -> Optional[IRenderer]:
        """Select the best renderer for given visual data.
        
        Args:
            visual_data: Visual data to be rendered
            quality_preference: Selection strategy
                - 'best_available': Highest quality score
                - 'fastest': Prefer native renderers (no external tools)
                - 'native_only': Only native renderers
                
        Returns:
            Optional[IRenderer]: Selected renderer, or None if no renderer available
        """
        candidates = self.get_renderers_for_type(visual_data.type)
        
        if not candidates:
            logger.warning(f"No renderer available for type: {visual_data.type}")
            return None
        
        # Apply preference filter
        if quality_preference == 'native_only':
            candidates = [r for r in candidates if 'native' in r.name.lower()]
            if not candidates:
                logger.warning(f"No native renderer for type: {visual_data.type}")
                return None
        
        # Estimate quality for each candidate
        scored = []
        for renderer in candidates:
            try:
                quality = renderer.estimate_quality(visual_data)
                if quality > 0:
                    scored.append((renderer, quality))
            except Exception as e:
                logger.warning(f"Failed to estimate quality for {renderer.name}: {e}")
        
        if not scored:
            logger.warning(f"All renderers rejected type: {visual_data.type}")
            return None
        
        # Sort by quality (descending) or speed preference
        if quality_preference == 'fastest':
            # Prefer native (fast) over external tools
            scored.sort(key=lambda x: (
                1 if 'native' in x[0].name.lower() else 0,
                x[1]
            ), reverse=True)
        else:  # best_available
            scored.sort(key=lambda x: x[1], reverse=True)
        
        selected = scored[0][0]
        logger.debug(f"Selected renderer: {selected.name} (quality: {scored[0][1]})")
        return selected
    
    def render_with_fallback(
        self,
        slide: Any,
        visual_data: VisualDataProtocol,
        spec: Dict[str, Any],
        left: float,
        top: float,
        width: float,
        height: float
    ) -> bool:
        """Attempt to render with automatic fallback on failure.
        
        Tries renderers in quality order, falling back to next-best on failure.
        
        Args:
            slide: python-pptx Slide object
            visual_data: Visual data to render
            spec: Design specification
            left, top, width, height: Position and size in inches
            
        Returns:
            bool: True if any renderer succeeded, False if all failed
        """
        candidates = self.get_renderers_for_type(visual_data.type)
        
        if not candidates:
            logger.error(f"No renderer available for type: {visual_data.type}")
            return False
        
        # Sort by quality
        scored = []
        for renderer in candidates:
            try:
                quality = renderer.estimate_quality(visual_data)
                if quality > 0:
                    scored.append((renderer, quality))
            except Exception as e:
                logger.warning(f"Quality estimation failed for {renderer.name}: {e}")
        
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Try each renderer in order
        for renderer, quality in scored:
            try:
                logger.debug(f"Attempting render with {renderer.name} (quality: {quality})")
                if renderer.render(slide, visual_data, spec, left, top, width, height):
                    logger.info(f"✓ {visual_data.type} rendered successfully by {renderer.name}")
                    return True
                else:
                    logger.warning(f"✗ {renderer.name} returned False, trying next...")
            except Exception as e:
                logger.warning(f"✗ {renderer.name} failed: {e}, trying next...")
        
        logger.error(f"✗ All renderers failed for type: {visual_data.type}")
        return False
    
    def list_renderers(self) -> List[Dict[str, Any]]:
        """List all registered renderers with their status.
        
        Returns:
            List of renderer info dicts
        """
        return [
            {
                'name': r.name,
                'types': r.supported_types,
                'available': r.is_available(),
                'class': r.__class__.__name__
            }
            for r in self._renderers
        ]


# Global registry instance (singleton pattern)
_global_registry: Optional[RendererRegistry] = None


def get_registry() -> RendererRegistry:
    """Get or create the global renderer registry.
    
    Returns:
        RendererRegistry: Global singleton instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = RendererRegistry()
    return _global_registry
