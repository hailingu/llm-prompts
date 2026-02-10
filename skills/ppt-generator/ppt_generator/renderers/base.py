"""Base renderer class with common utilities."""

from abc import abstractmethod
from typing import Any, Dict
import logging

from ..protocols.renderer_interface import IRenderer
from ..protocols.visual_data_protocol import VisualDataProtocol


logger = logging.getLogger(__name__)


class BaseRenderer(IRenderer):
    """Abstract base class providing common functionality for all renderers.
    
    Subclasses only need to implement:
    - name property
    - supported_types property
    - is_available() method
    - estimate_quality() method
    - _do_render() method (instead of render())
    """
    
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
        """Template method with error handling and logging.
        
        Subclasses should override _do_render() instead of this method.
        """
        try:
            logger.debug(f"{self.name}: Starting render for {visual_data.type}")
            result = self._do_render(slide, visual_data, spec, left, top, width, height)
            if result:
                logger.debug(f"{self.name}: Render succeeded")
            else:
                logger.warning(f"{self.name}: Render returned False")
            return result
        except Exception as e:
            logger.error(f"{self.name}: Render failed with exception: {e}", exc_info=True)
            return False
    
    @abstractmethod
    def _do_render(
        self,
        slide: Any,
        visual_data: VisualDataProtocol,
        spec: Dict[str, Any],
        left: float,
        top: float,
        width: float,
        height: float
    ) -> bool:
        """Actual rendering implementation.
        
        Subclasses implement this instead of render().
        Should not raise exceptions (return False on error).
        """
        pass
    
    def _validate_data(self, visual_data: VisualDataProtocol) -> bool:
        """Validate visual data before rendering.
        
        Args:
            visual_data: Data to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        if not visual_data.type:
            logger.error(f"{self.name}: Missing visual type")
            return False
        
        if not self.can_render(visual_data.type):
            logger.error(f"{self.name}: Cannot render type '{visual_data.type}'")
            return False
        
        return True
