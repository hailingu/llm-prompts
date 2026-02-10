"""Visual Data Protocol (VDP) - Unified data models for all visual types.

This module defines the canonical data structures that all renderers must accept.
It uses Pydantic for validation and JSON Schema generation.
"""

from typing import List, Optional, Literal, Dict, Any, Union
from pydantic import BaseModel, Field, validator
from datetime import datetime


class GanttTask(BaseModel):
    """Single task in a Gantt chart."""
    name: str = Field(..., description="Task name")
    start_month: int = Field(..., ge=0, description="Start month (0-indexed from timeline start)")
    duration_months: int = Field(..., gt=0, description="Duration in months")
    status: Literal['completed', 'active', 'planned', 'done', 'in_progress', 'pending'] = Field(
        default='planned',
        description="Task status for color coding"
    )
    dependencies: Optional[List[str]] = Field(
        default=None,
        description="List of task names this depends on"
    )
    owner: Optional[str] = Field(default=None, description="Task owner/assignee")
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "项目立项",
                "start_month": 0,
                "duration_months": 3,
                "status": "active"
            }
        }


class GanttTimeline(BaseModel):
    """Timeline configuration for Gantt chart."""
    start: str = Field(..., description="Start date in YYYY-MM format")
    end: str = Field(..., description="End date in YYYY-MM format")
    unit: Literal['day', 'week', 'month', 'quarter'] = Field(
        default='month',
        description="Time unit for the chart"
    )
    
    @validator('start', 'end')
    def validate_date_format(cls, v):
        """Validate date format is YYYY-MM or YYYY-MM-DD."""
        if not v:
            raise ValueError("Date cannot be empty")
        parts = v.split('-')
        if len(parts) < 2:
            raise ValueError(f"Invalid date format: {v}, expected YYYY-MM or YYYY-MM-DD")
        return v


class GanttData(BaseModel):
    """Complete data for a Gantt chart."""
    timeline: GanttTimeline
    tasks: List[GanttTask] = Field(..., min_items=1, description="List of tasks")
    milestones: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Optional milestone markers"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "timeline": {
                    "start": "2026-02",
                    "end": "2027-02",
                    "unit": "month"
                },
                "tasks": [
                    {
                        "name": "项目立项",
                        "start_month": 0,
                        "duration_months": 3,
                        "status": "active"
                    },
                    {
                        "name": "样机验证",
                        "start_month": 3,
                        "duration_months": 6,
                        "status": "planned"
                    }
                ]
            }
        }


class TimelineItem(BaseModel):
    """Single item in a timeline visualization."""
    label: Optional[str] = None
    milestone: Optional[str] = None
    phase: Optional[str] = None
    title: Optional[str] = None
    date: Optional[str] = None
    period: Optional[str] = None
    time: Optional[str] = None
    description: Optional[str] = None
    status: Literal['completed', 'active', 'planned', 'done', 'in_progress', 'pending'] = 'planned'
    
    @validator('label', 'milestone', 'phase', 'title', always=True)
    def ensure_display_text(cls, v, values):
        """Ensure at least one display field is set."""
        if not any([v, values.get('milestone'), values.get('phase'), values.get('title')]):
            if not values:
                raise ValueError("At least one of label/milestone/phase/title must be set")
        return v


class TimelineData(BaseModel):
    """Data for a simple timeline visualization."""
    items: List[TimelineItem] = Field(..., min_items=1)
    
    class Config:
        json_schema_extra = {
            "example": {
                "items": [
                    {
                        "phase": "0-3个月",
                        "period": "项目立项、示范场地确认",
                        "status": "active"
                    },
                    {
                        "phase": "3-9个月",
                        "period": "样机验证与现场小规模示范",
                        "status": "planned"
                    }
                ]
            }
        }


class VisualDataProtocol(BaseModel):
    """Unified visual data container that all renderers accept.
    
    This is the contract between content planners/designers and renderers.
    Each visual type has its own typed 'data' field.
    """
    type: str = Field(..., description="Visual type identifier (e.g., 'gantt', 'timeline')")
    title: Optional[str] = Field(default=None, description="Visual title")
    data: Union[GanttData, TimelineData, Dict[str, Any]] = Field(
        ...,
        description="Typed data payload (GanttData, TimelineData, or dict for fallback)"
    )
    
    # Legacy support for current system
    placeholder_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Legacy placeholder_data for backward compatibility"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "type": "gantt",
                "title": "项目实施路线图",
                "data": {
                    "timeline": {"start": "2026-02", "end": "2027-02", "unit": "month"},
                    "tasks": [
                        {"name": "项目立项", "start_month": 0, "duration_months": 3, "status": "active"}
                    ]
                }
            }
        }
    
    @validator('data', pre=True)
    def parse_data(cls, v, values):
        """Auto-parse data based on type."""
        vtype = values.get('type', '').lower()
        
        if isinstance(v, dict):
            # Try to parse into typed models
            if vtype in ('gantt', 'gantt_chart'):
                try:
                    return GanttData(**v)
                except Exception:
                    pass
            elif vtype == 'timeline':
                try:
                    return TimelineData(**v)
                except Exception:
                    pass
        
        return v
