"""Renderer interface protocol - all renderers must implement this."""

from abc import ABC, abstractmethod
from typing import List, Any, Dict
from .visual_data_protocol import VisualDataProtocol


class IRenderer(ABC):
    """Abstract base interface for all visual renderers.
    
    All renderers (native, mermaid, matplotlib, etc.) must implement this interface
    to be discoverable and usable by the rendering engine.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique renderer identifier (e.g., 'native-gantt', 'mermaid-flowchart').
        
        Returns:
            str: Renderer name used for logging and debugging
        """
        pass
    
    @property
    @abstractmethod
    def supported_types(self) -> List[str]:
        """List of visual types this renderer can handle.
        
        Returns:
            List[str]: Visual type identifiers (e.g., ['gantt', 'gantt_chart'])
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this renderer is available (dependencies satisfied).
        
        For example:
        - Native renderers always return True
        - Mermaid renderer checks if 'mmdc' command exists
        - Matplotlib renderer checks if the library is installed
        
        Returns:
            bool: True if renderer can be used
        """
        pass
    
    @abstractmethod
    def estimate_quality(self, visual_data: VisualDataProtocol) -> int:
        """Estimate rendering quality score for given data.
        
        Used by the registry to select the best renderer when multiple
        renderers support the same visual type.
        
        Args:
            visual_data: The visual data to be rendered
            
        Returns:
            int: Quality score 0-100 (higher is better)
                 0 = cannot render
                 1-50 = low quality (fallback)
                 51-80 = medium quality (acceptable)
                 81-100 = high quality (preferred)
        """
        pass
    
    @abstractmethod
    def render(
        self,
        slide: Any,
        visual_data: VisualDataProtocol,
        spec: Dict[str, Any],
        left: float,
        top: float,
        width: float,
        height: float
    ) -> bool:
        """Render the visual onto a PowerPoint slide.
        
        Args:
            slide: python-pptx Slide object
            visual_data: Validated visual data conforming to VDP
            spec: Design specification (colors, fonts, etc.)
            left: Left position in inches
            top: Top position in inches
            width: Width in inches
            height: Height in inches
            
        Returns:
            bool: True if rendering succeeded, False otherwise
            
        Raises:
            Should not raise exceptions; return False on failure
        """
        pass
    
    def can_render(self, visual_type: str) -> bool:
        """Quick check if this renderer supports a given type.
        
        Args:
            visual_type: Visual type identifier
            
        Returns:
            bool: True if this renderer supports the type
        """
        return visual_type.lower() in [t.lower() for t in self.supported_types]
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"<{self.__class__.__name__} name='{self.name}' types={self.supported_types}>"
