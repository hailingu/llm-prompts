#!/usr/bin/env python3
"""测试 v15 混合渲染架构中的渲染器选择"""

import sys
sys.path.insert(0, 'skills/ppt-generator')

from ppt_generator.core.registry import get_registry
from ppt_generator.protocols.visual_data_protocol import VisualDataProtocol, GanttData, GanttTask

def test_renderer_selection():
    r = get_registry()
    
    print("=" * 60)
    print("🔍 渲染器注册状态")
    print("=" * 60)
    print(f"总渲染器数: {len(r._renderers)}")
    print(f"可用渲染器数: {len([x for x in r._renderers if x.is_available()])}")
    print()
    
    for renderer in r._renderers:
        if renderer.is_available():
            print(f"✅ {renderer.name}")
            print(f"   支持类型: {renderer.supported_types[:5]}...")
    print()
    
    print("=" * 60)
    print("🎯 渲染器选择测试")
    print("=" * 60)
    
    # Test 1: Flowchart (slide 16)
    print("📊 Slide 16 - Flowchart")
    flowchart_data = VisualDataProtocol(
        type='flowchart',
        data={},
        placeholder_data={
            'mermaid_code': '''flowchart TD
  Start[开始] --> Wind[绕组制造]
  Wind --> Check1{SPC 检查}
  Check1 -->|Pass| Varnish[浸漆处理]
  Check1 -->|Fail| Rework1[返工]'''
        }
    )
    
    selected = r.select_renderer(flowchart_data)
    if selected:
        quality = selected.estimate_quality(flowchart_data)
        print(f"   选中: {selected.name} (质量: {quality}/100)")
    else:
        print("   ❌ 无可用渲染器")
    print()
    
    # Test 2: Gantt with structured data (slide 28)
    print("📊 Slide 28 - Gantt (有结构化数据)")
    gantt_data = VisualDataProtocol(
        type='gantt',
        data=GanttData(
            timeline={'start': '2026-02', 'end': '2027-02', 'unit': 'month'},
            tasks=[
                GanttTask(name='项目立项', start_month=0, duration_months=3, status='active'),
                GanttTask(name='样机验证', start_month=3, duration_months=6, status='planned'),
            ]
        ),
        placeholder_data={'mermaid_code': 'gantt\n  title Project'}
    )
    
    selected2 = r.select_renderer(gantt_data)
    if selected2:
        quality2 = selected2.estimate_quality(gantt_data)
        print(f"   选中: {selected2.name} (质量: {quality2}/100)")
    else:
        print("   ❌ 无可用渲染器")
    print()
    
    # Test 3: Gantt with only mermaid_code (no structured data)
    print("📊 Gantt (仅 mermaid_code)")
    gantt_mermaid_only = VisualDataProtocol(
        type='gantt',
        data={},
        placeholder_data={'mermaid_code': 'gantt\n  title Project\n  section Section\n  Task1 :a1, 2026-02-01, 30d'}
    )
    
    selected3 = r.select_renderer(gantt_mermaid_only)
    if selected3:
        quality3 = selected3.estimate_quality(gantt_mermaid_only)
        print(f"   选中: {selected3.name} (质量: {quality3}/100)")
    else:
        print("   ❌ 无可用渲染器")
    print()
    
    print("=" * 60)
    print("📦 v15 PPT 生成结果")
    print("=" * 60)
    import subprocess
    result = subprocess.run(['ls', '-lh', '/private/tmp/MFT-20260210.v15-mermaid-hybrid.pptx'], 
                          capture_output=True, text=True)
    print(result.stdout)

if __name__ == '__main__':
    test_renderer_selection()
