#!/usr/bin/env python3
"""诊断 v15 渲染问题"""

import sys
sys.path.insert(0, 'skills/ppt-generator')

import json
from ppt_generator.protocols.visual_data_protocol import VisualDataProtocol, GanttData
from ppt_generator.core.registry import get_registry

# 加载数据
with open('docs/presentations/MFT-20260210/slides_semantic.json') as f:
    data = json.load(f)

print("=" * 70)
print("🔍 诊断 Slide 16 (flowchart) 和 Slide 29 (gantt) 的渲染问题")
print("=" * 70)

# Slide 16 - Flowchart
slide16 = [s for s in data['slides'] if s['id'] == 16][0]
print("\n📊 Slide 16 - 制造及质量一致性 (flowchart)")
print(f"   slide_type: {slide16['slide_type']}")
print(f"   visual.type: {slide16['visual']['type']}")
print(f"   Has mermaid_code: {'mermaid_code' in slide16['visual'].get('placeholder_data', {})}")

try:
    visual16 = slide16['visual']
    vdp16 = VisualDataProtocol(**visual16)
    registry = get_registry()
    renderer16 = registry.select_renderer(vdp16)
    if renderer16:
        quality16 = renderer16.estimate_quality(vdp16)
        print(f"   ✅ 选中渲染器: {renderer16.name} (质量: {quality16})")
    else:
        print("   ❌ 未选中渲染器")
except Exception as e:
    print(f"   ❌ 错误: {e}")

# Slide 29 - Gantt
slide29 = [s for s in data['slides'] if s['id'] == 29][0]
print("\n📊 Slide 29 - 12-18个月扩展计划 (gantt)")
print(f"   slide_type: {slide29['slide_type']}")
print(f"   visual.type: {slide29['visual']['type']}")
print(f"   Has gantt_data: {'gantt_data' in slide29['visual'].get('placeholder_data', {})}")
print(f"   Has mermaid_code: {'mermaid_code' in slide29['visual'].get('placeholder_data', {})}")

try:
    visual29 = slide29['visual']
    vdp29 = VisualDataProtocol(**visual29)
    print(f"   VDP data type: {type(vdp29.data).__name__ if vdp29.data else 'None'}")
    
    renderer29 = registry.select_renderer(vdp29)
    if renderer29:
        quality29 = renderer29.estimate_quality(vdp29)
        print(f"   ✅ 选中渲染器: {renderer29.name} (质量: {quality29})")
    else:
        print("   ❌ 未选中渲染器")
except Exception as e:
    print(f"   ❌ 错误: {e}")
    import traceback
    traceback.print_exc()

# 检查渲染器可用性
print("\n" + "=" * 70)
print("📋 当前可用渲染器")
print("=" * 70)
for r in registry._renderers:
    if r.is_available():
        print(f"✅ {r.name}: {r.supported_types[:5]}...")
