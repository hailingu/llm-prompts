#!/usr/bin/env python3
"""Test script for plugin-based renderer architecture."""

import sys
import os

# Add parent directory to path to import ppt_generator modules directly
sys.path.insert(0, os.path.dirname(__file__))

# Import directly from submodules to avoid circular imports from __init__.py
from ppt_generator.core.registry import get_registry
from ppt_generator.protocols.visual_data_protocol import VisualDataProtocol, GanttData

def test_registry():
    """Test renderer registry initialization."""
    print("=" * 60)
    print("Testing Renderer Registry")
    print("=" * 60)
    
    registry = get_registry()
    print(f"✓ Registry initialized")
    print(f"  Discovered {len(registry._renderers)} renderers")
    
    # List all renderers
    print("\nAvailable renderers:")
    for r_info in registry.list_renderers():
        symbol = '✓' if r_info['available'] else '✗'
        print(f"  {symbol} {r_info['name']}: {r_info['types']}")
    
    return registry

def test_gantt_data():
    """Test Gantt data model validation."""
    print("\n" + "=" * 60)
    print("Testing GanttData Model")
    print("=" * 60)
    
    gantt_data = GanttData(
        timeline={'start': '2026-02', 'end': '2027-02', 'unit': 'month'},
        tasks=[
            {'name': '项目立项', 'start_month': 0, 'duration_months': 3, 'status': 'active'},
            {'name': '样机验证', 'start_month': 3, 'duration_months': 6, 'status': 'planned'},
            {'name': '数据分析', 'start_month': 9, 'duration_months': 3, 'status': 'planned'}
        ]
    )
    print(f"✓ GanttData validation passed")
    print(f"  Timeline: {gantt_data.timeline.start} to {gantt_data.timeline.end}")
    print(f"  Tasks: {len(gantt_data.tasks)}")
    
    return gantt_data

def test_vdp(gantt_data):
    """Test Visual Data Protocol."""
    print("\n" + "=" * 60)
    print("Testing VisualDataProtocol")
    print("=" * 60)
    
    vdp = VisualDataProtocol(
        type='gantt',
        title='项目实施路线图',
        data=gantt_data
    )
    print(f"✓ VisualDataProtocol created")
    print(f"  Type: {vdp.type}")
    print(f"  Title: {vdp.title}")
    print(f"  Data type: {type(vdp.data).__name__}")
    
    return vdp

def test_renderer_selection(registry, vdp):
    """Test renderer selection."""
    print("\n" + "=" * 60)
    print("Testing Renderer Selection")
    print("=" * 60)
    
    renderer = registry.select_renderer(vdp)
    if renderer:
        print(f"✓ Selected renderer: {renderer.name}")
        quality = renderer.estimate_quality(vdp)
        print(f"  Quality score: {quality}/100")
        print(f"  Available: {renderer.is_available()}")
        return True
    else:
        print("✗ No renderer found")
        return False

def main():
    """Run all tests."""
    try:
        registry = test_registry()
        gantt_data = test_gantt_data()
        vdp = test_vdp(gantt_data)
        success = test_renderer_selection(registry, vdp)
        
        print("\n" + "=" * 60)
        if success:
            print("✓ All tests passed!")
        else:
            print("✗ Some tests failed")
        print("=" * 60)
        
        return 0 if success else 1
        
    except Exception as e:
        print(f"\n✗ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
