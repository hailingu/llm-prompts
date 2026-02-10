"""Visual data protocols and interfaces for PPT generation."""

from .visual_data_protocol import (
    VisualDataProtocol,
    GanttTask,
    GanttData,
    TimelineItem,
    TimelineData,
)
from .renderer_interface import IRenderer

__all__ = [
    'VisualDataProtocol',
    'GanttTask',
    'GanttData',
    'TimelineItem',
    'TimelineData',
    'IRenderer',
]
