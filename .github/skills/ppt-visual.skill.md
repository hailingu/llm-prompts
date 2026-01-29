---
name: ppt-visual
version: 1.2.0
description: "提供通用视觉设计原则（Presentation Zen, Apple Keynote风格）和 Material Design 集成指南，包括视觉层次、图标系统、图像处理、Material Type Scale、Material Motion 和组件规范。注：完整的 Material Design 系统（Design Tokens、品牌一致性）由 ppt-theme-manager.skill 负责。"
category: presentation
dependencies:
  libraries:
    - heroicons  # MIT License - Icon system
    - lucide-icons  # ISC License - Icon fallback
    - material-design-3  # Design tokens and components
    - mermaid.js  # Diagram rendering
  python_packages:
    - PyYAML  # VISUAL block parsing
    - Pillow  # Image processing
tags:
  - visual-hierarchy
  - icon-system
  - image-processing
  - color-psychology
  - layout-composition
  - visual-annotation
  - diagram-generation
  - material-design
  - material-type-scale
  - material-motion
standards:
  - Presentation Zen (Garr Reynolds, 2008)
  - Apple Human Interface Guidelines
  - Material Design 3 (Google, 2021)
  - Material Motion Guidelines
  - Swiss Design Grid Systems (Josef Müller-Brockmann)
integration:
  agents:
    - ppt-specialist  # Primary consumer for VISUAL processing
    - ppt-visual-designer  # Uses for visual principles and Material specs
    - ppt-content-planner  # Generates VISUAL blocks
  skills:
    - ppt-theme-manager  # Provides Material Design Tokens and brand system
    - ppt-layout  # Provides grid system and layout templates
    - ppt-chart  # Provides data visualization specs
last_updated: 2026-01-28
---

# ppt-visual Skill

**功能**：提供通用视觉设计原则和 Material Design 集成指南，包括视觉层次设计、图标系统、图像处理、布局构图、Material Type Scale、Material Motion 和组件规范。

**职责边界**：
- ✅ **本 skill 负责**：通用视觉原则（Presentation Zen、Apple Keynote）、Material Design 应用指南、VISUAL block 处理
- 🔗 **协作 skill**：
  - `ppt-theme-manager.skill`：Material Design Tokens、品牌色彩系统、WCAG 验证
  - `ppt-layout.skill`：网格系统、布局模板
  - `ppt-chart.skill`：数据可视化、Cleveland Hierarchy

---

## 1. 核心设计原则

### 1.1 Visual Hierarchy Design（视觉层次设计）

**Garr Reynolds - Presentation Zen原则**：
- **Big, Bold, Beautiful**: 大标题、粗体强调、美观图片
- **Signal vs Noise**: 信号（关键信息）最大化，噪音（装饰）最小化
- **Restraint**: 克制使用效果，简约至上

**层次设计公式**：
```
Z-Index优先级（从高到低）：
1. 关键数据/结论（最大字号，强对比色）
2. 标题/主题（大字号，品牌色）
3. 支撑数据/图表（中字号，中性色）
4. 注释/来源（小字号，浅色）
5. 背景/装饰（最低优先级）
```

**实现示例**：
```python
def apply_visual_hierarchy(elements):
    """应用视觉层次原则"""
    hierarchy = {
        'key_message': {
            'font_size': 48,
            'font_weight': 'bold',
            'color': '#1E293B',  # 最深
            'position': 'center-top'
        },
        'title': {
            'font_size': 36,
            'font_weight': 'semibold',
            'color': '#2563EB',  # 品牌色
            'position': 'top-left'
        },
        'content': {
            'font_size': 18,
            'font_weight': 'normal',
            'color': '#475569',  # 中性
            'position': 'body'
        },
        'annotation': {
            'font_size': 12,
            'font_weight': 'light',
            'color': '#94A3B8',  # 浅色
            'position': 'bottom'
        }
    }
    return hierarchy
```

---

### 1.2 Color Psychology（色彩心理学）

**情感色彩映射**：
```python
COLOR_EMOTIONS = {
    'trust': '#2563EB',      # 蓝色 - 专业、可靠
    'growth': '#10B981',     # 绿色 - 增长、成功
    'energy': '#F59E0B',     # 橙色 - 活力、创新
    'urgency': '#EF4444',    # 红色 - 紧急、警告
    'stability': '#6366F1',  # 靛蓝 - 稳定、传统
    'creativity': '#8B5CF6', # 紫色 - 创意、想象
    'neutral': '#64748B'     # 灰色 - 中性、平衡
}

def choose_color_by_message(message_tone):
    """根据消息基调选择颜色"""
    if 'risk' in message_tone or 'problem' in message_tone:
        return COLOR_EMOTIONS['urgency']
    elif 'success' in message_tone or 'achievement' in message_tone:
        return COLOR_EMOTIONS['growth']
    elif 'innovation' in message_tone:
        return COLOR_EMOTIONS['creativity']
    else:
        return COLOR_EMOTIONS['trust']  # 默认专业蓝
```

---

### 1.3 Layout Composition（布局构图）

**网格系统**（瑞士设计）：
```
12列网格布局：
┌─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┐
│ │ │ │ │ │ │ │ │ │ │ │ │
└─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┘

常用分割：
- 标题：1-12列（全宽）
- 单栏内容：2-11列（留边距）
- 双栏布局：1-6 | 7-12
- 三栏布局：1-4 | 5-8 | 9-12
- 重点内容：4-9列（居中）
```

**黄金比例应用**：
```python
GOLDEN_RATIO = 1.618

def apply_golden_ratio(width):
    """应用黄金比例分割"""
    return {
        'major': width / GOLDEN_RATIO,  # 约62%
        'minor': width - (width / GOLDEN_RATIO)  # 约38%
    }

# 示例：1920px宽度
layout = apply_golden_ratio(1920)
# major: 1186px, minor: 734px
```

**三分法构图**：
```
视觉焦点位置（Rule of Thirds）：
┌───┬───┬───┐
│ · │   │ · │  ← 上方交点
├───┼───┼───┤
│   │   │   │
├───┼───┼───┤
│ · │   │ · │  ← 下方交点
└───┴───┴───┘

关键元素放置在4个交点附近
标题/图表中心对齐到交点
```

---

### 1.4 Material Type Scale（Material Design 字体等级）

**功能**：Material Design 3 的字体等级系统，针对幻灯片场景优化。

**完整 Type Scale**（适配演示场景）：
```python
MATERIAL_TYPE_SCALE = {
    # Display 等级 - 用于超大标题/封面页
    'display_large': {
        'size': 96,          # 原96sp → 96pt (slides)
        'weight': 'regular', # 400
        'line_height': 1.1,
        'usage': '封面页主标题、章节分隔超大标题'
    },
    'display_medium': {
        'size': 72,
        'weight': 'regular',
        'line_height': 1.1,
        'usage': '章节页标题'
    },
    'display_small': {
        'size': 60,
        'weight': 'regular',
        'line_height': 1.15,
        'usage': '强调性标题'
    },
    
    # Headline 等级 - 用于幻灯片标题
    'headline_large': {
        'size': 48,
        'weight': 'semibold',  # 600
        'line_height': 1.2,
        'usage': '主幻灯片标题（标准场景）'
    },
    'headline_medium': {
        'size': 36,
        'weight': 'semibold',
        'line_height': 1.25,
        'usage': '次级标题、数据图表标题'
    },
    'headline_small': {
        'size': 28,
        'weight': 'semibold',
        'line_height': 1.3,
        'usage': '子标题、卡片标题'
    },
    
    # Title 等级 - 用于内容区块标题
    'title_large': {
        'size': 24,
        'weight': 'medium',  # 500
        'line_height': 1.3,
        'usage': '内容区块标题、列表标题'
    },
    'title_medium': {
        'size': 20,
        'weight': 'medium',
        'line_height': 1.35,
        'usage': '小节标题、表格标题'
    },
    'title_small': {
        'size': 18,
        'weight': 'medium',
        'line_height': 1.4,
        'usage': '强调性正文、引用标题'
    },
    
    # Body 等级 - 用于正文内容
    'body_large': {
        'size': 18,
        'weight': 'regular',
        'line_height': 1.5,
        'usage': '标准正文、列表项（大场景）'
    },
    'body_medium': {
        'size': 16,
        'weight': 'regular',
        'line_height': 1.5,
        'usage': '标准正文（中小场景）'
    },
    'body_small': {
        'size': 14,
        'weight': 'regular',
        'line_height': 1.55,
        'usage': '次要正文、表格内容'
    },
    
    # Label 等级 - 用于标签/注释
    'label_large': {
        'size': 14,
        'weight': 'medium',
        'line_height': 1.4,
        'usage': '按钮、标签、标注'
    },
    'label_medium': {
        'size': 12,
        'weight': 'medium',
        'line_height': 1.4,
        'usage': '图表标签、数据标注'
    },
    'label_small': {
        'size': 10,
        'weight': 'medium',
        'line_height': 1.5,
        'usage': '版权信息、引用来源'
    }
}

def apply_material_type_scale(element_type):
    """根据元素类型应用 Material Type Scale"""
    return MATERIAL_TYPE_SCALE.get(element_type, MATERIAL_TYPE_SCALE['body_medium'])
```

**使用示例**：
```python
# 幻灯片标题
title_style = apply_material_type_scale('headline_large')
# {'size': 48, 'weight': 'semibold', 'line_height': 1.2, 'usage': '主幻灯片标题'}

# 正文内容
body_style = apply_material_type_scale('body_large')
# {'size': 18, 'weight': 'regular', 'line_height': 1.5, 'usage': '标准正文'}
```

**与 ppt-theme-manager 协作**：
- `ppt-theme-manager.skill` 定义字体家族（Roboto, Noto Sans CJK）
- 本 skill 提供 Type Scale 等级系统
- `ppt-visual-designer.agent` 整合两者生成完整字体规范

---

### 1.5 Material Motion（Material Design 动画规范）

**功能**：Material Design 动画时长和缓动函数规范，用于定义幻灯片切换和元素动画。

**Motion Tokens**：
```python
MATERIAL_MOTION = {
    # Duration 时长（毫秒）
    'duration': {
        'short1': 50,      # 微交互（状态变化）
        'short2': 100,     # 简单淡入淡出
        'medium1': 200,    # 标准入场动画
        'medium2': 300,    # 标准动画
        'long1': 400,      # 复杂动画
        'long2': 500,      # 大型转场
        'extra_long': 700  # 全屏转场（慎用）
    },
    
    # Easing 缓动函数
    'easing': {
        'standard': 'cubic-bezier(0.4, 0.0, 0.2, 1)',      # 标准曲线
        'decelerate': 'cubic-bezier(0.0, 0.0, 0.2, 1)',    # 减速（入场）
        'accelerate': 'cubic-bezier(0.4, 0.0, 1, 1)',      # 加速（退场）
        'linear': 'linear'                                  # 线性（进度条）
    },
    
    # Animation Types 动画类型
    'patterns': {
        'fade_in': {
            'duration': 200,
            'easing': 'decelerate',
            'properties': ['opacity'],
            'from': 0,
            'to': 1
        },
        'slide_up': {
            'duration': 300,
            'easing': 'decelerate',
            'properties': ['transform'],
            'from': 'translateY(40px)',
            'to': 'translateY(0)'
        },
        'scale_emphasis': {
            'duration': 200,
            'easing': 'standard',
            'properties': ['transform'],
            'from': 'scale(1.0)',
            'to': 'scale(1.05)'
        },
        'fade_out': {
            'duration': 150,
            'easing': 'accelerate',
            'properties': ['opacity'],
            'from': 1,
            'to': 0
        }
    }
}

def create_entrance_animation(element_type='standard'):
    """创建 Material Design 风格的入场动画"""
    if element_type == 'hero':
        # 封面元素：淡入 + 轻微上滑
        return {
            'fade_in': MATERIAL_MOTION['patterns']['fade_in'],
            'slide_up': MATERIAL_MOTION['patterns']['slide_up'],
            'total_duration': 300
        }
    elif element_type == 'content':
        # 内容元素：快速淡入
        return {
            'fade_in': {
                **MATERIAL_MOTION['patterns']['fade_in'],
                'duration': 200
            },
            'total_duration': 200
        }
    else:
        return MATERIAL_MOTION['patterns']['fade_in']
```

**幻灯片动画推荐**：
```yaml
# 推荐使用场景
slide_transitions:
  default: 
    type: fade
    duration: 300ms
    easing: standard
  
  section_divider:
    type: slide_left
    duration: 400ms
    easing: decelerate

element_animations:
  bullet_points:
    type: fade_in + slide_up
    duration: 200ms
    stagger: 100ms  # 逐项延迟
  
  charts:
    type: fade_in
    duration: 300ms
    data_animation: true  # 数据增长动画
  
  diagrams:
    type: fade_in
    duration: 400ms
    sequential: true  # 组件逐个出现
```

**注意事项**：
- ❌ **避免过度动画**：演示文稿应克制使用动画（Presentation Zen 原则）
- ✅ **仅在必要时使用**：强调关键信息、引导视线、章节转场
- ⚠️ **性能限制**：避免同时动画 >5 个元素，避免 width/height 动画（非 GPU 加速）

---

### 1.6 Material Components（Material Design 组件规范）

**功能**：Material Design 组件在幻灯片中的应用规范。

**核心组件**：

#### 1.6.1 Cards（卡片）
```python
MATERIAL_CARD = {
    'elevation': 1,           # Material elevation level (0-5)
    'corner_radius': 8,       # 圆角（pt）
    'padding': 16,            # 内边距（pt）
    'background': 'surface',  # 使用 surface token（由 theme-manager 提供）
    'border': None,           # 默认无边框（用 elevation 区分）
    'usage': '内容分组、数据展示、引用块'
}

def create_card_spec(content_type):
    """创建卡片规范"""
    base = MATERIAL_CARD.copy()
    
    if content_type == 'data':
        # 数据卡片：添加头部分隔线
        base['header_divider'] = True
        base['padding_header'] = 12
        base['padding_content'] = 16
    elif content_type == 'quote':
        # 引用卡片：增加左侧强调条
        base['accent_bar'] = {
            'width': 4,
            'color': 'primary',  # 使用主题色
            'position': 'left'
        }
        base['background_tint'] = 0.05  # 轻微背景色
    
    return base
```

#### 1.6.2 Chips（标签/徽章）
```python
MATERIAL_CHIP = {
    'height': 32,             # 固定高度（pt）
    'padding_horizontal': 12, # 水平内边距
    'corner_radius': 16,      # 半圆角
    'font_size': 14,          # label_large
    'font_weight': 'medium',
    'usage': '标签、分类、状态指示'
}

CHIP_VARIANTS = {
    'assist': {              # 辅助操作
        'background': 'transparent',
        'border': '1px solid outline',
        'color': 'on_surface'
    },
    'filter': {              # 筛选标签
        'background': 'secondary_container',
        'color': 'on_secondary_container',
        'selected_background': 'secondary'
    },
    'input': {               # 输入标签（可删除）
        'background': 'surface_variant',
        'color': 'on_surface_variant',
        'close_icon': True
    },
    'suggestion': {          # 建议标签
        'background': 'surface_variant',
        'color': 'on_surface_variant',
        'elevation': 0
    }
}
```

#### 1.6.3 Data Tables（数据表格）
```python
MATERIAL_TABLE = {
    'header': {
        'font_size': 14,           # label_large
        'font_weight': 'medium',
        'color': 'on_surface_variant',
        'padding_vertical': 12,
        'background': 'surface_variant',
        'border_bottom': '2px solid outline_variant'
    },
    'row': {
        'font_size': 14,           # body_medium
        'font_weight': 'regular',
        'padding_vertical': 16,
        'min_height': 52,
        'border_bottom': '1px solid outline_variant'
    },
    'cell': {
        'padding_horizontal': 16,
        'alignment': {
            'text': 'left',
            'number': 'right',
            'icon': 'center'
        }
    },
    'zebra_striping': {        # 斑马纹（可选）
        'enabled': True,
        'odd_background': 'transparent',
        'even_background': 'surface_variant',
        'opacity': 0.3
    },
    'usage': '数据对比、规格说明、时间线'
}
```

#### 1.6.4 Callouts（提示框/强调块）
```python
MATERIAL_CALLOUT = {
    'accent_bar': {
        'width': 4,
        'position': 'left',
        'color': 'primary'  # 根据类型变化
    },
    'background': {
        'base': 'surface',
        'tint': 0.05  # 轻微主题色着色
    },
    'padding': 16,
    'corner_radius': 4,
    'icon': {              # 可选图标
        'size': 24,
        'position': 'top-left',
        'margin_right': 12
    }
}

CALLOUT_TYPES = {
    'info': {
        'accent_color': 'primary',
        'icon': 'information-circle',
        'background_tint': 'primary'
    },
    'success': {
        'accent_color': 'success',  # 需要 theme-manager 提供
        'icon': 'check-circle',
        'background_tint': 'success'
    },
    'warning': {
        'accent_color': 'warning',
        'icon': 'exclamation-triangle',
        'background_tint': 'warning'
    },
    'error': {
        'accent_color': 'error',
        'icon': 'x-circle',
        'background_tint': 'error'
    }
}
```

**组件使用决策树**：
```python
def select_component(content_intent):
    """根据内容意图选择 Material 组件"""
    if '数据对比' in content_intent or '规格' in content_intent:
        return 'data_table'
    elif '分组内容' in content_intent or '独立信息块' in content_intent:
        return 'card'
    elif '标签' in content_intent or '分类' in content_intent:
        return 'chip'
    elif '重要提示' in content_intent or '警告' in content_intent:
        return 'callout'
    else:
        return 'default_layout'  # 使用标准布局（无组件包装）
```

**与 ppt-theme-manager 协作**：
- `ppt-theme-manager.skill` 提供颜色 tokens（surface, primary, error 等）
- 本 skill 定义组件结构和规范
- `ppt-visual-designer.agent` 整合生成完整组件 specs

---

## 2. 视觉元素系统

### 2.1 Icon System（图标系统）

**设计原则**（Material Design + Apple SF Symbols）：
- **一致性**：统一风格（线性/填充/双色）
- **可识别性**：3秒内理解含义
- **可缩放性**：16px-128px清晰
- **无障碍**：配合文字标签使用

**图标库推荐**：
```yaml
primary_library: heroicons  # MIT License, 清晰现代
fallback: lucide-icons       # ISC License, 轻量级
custom: tabler-icons         # MIT License, 一致性好

style_guide:
  stroke_width: 2px
  corner_radius: 2px
  color: 继承文字颜色
  size: [16, 24, 32, 48]  # 4的倍数
```

**使用场景映射**：
```python
ICON_MAP = {
    # 状态
    'success': 'check-circle',
    'warning': 'exclamation-triangle',
    'error': 'x-circle',
    'info': 'information-circle',
    
    # 动作
    'download': 'arrow-down-tray',
    'upload': 'arrow-up-tray',
    'search': 'magnifying-glass',
    'settings': 'cog',
    
    # 业务
    'performance': 'chart-bar',
    'security': 'shield-check',
    'scalability': 'arrows-pointing-out',
    'cost': 'currency-dollar'
}

def select_icon(concept):
    """智能选择图标"""
    return ICON_MAP.get(concept.lower(), 'document')  # 默认文档图标
```

---

### 2.2 Image Treatment（图像处理）

**高质量图像标准**：
- **分辨率**：≥200 DPI（演示）, ≥300 DPI（打印）
- **格式**：PNG（透明）, JPG（照片）, SVG（图标/图表）
- **尺寸**：全屏1920x1080，半屏960x1080
- **优化**：压缩后≤500KB/张

**处理技巧**（Apple Keynote风格）：
```python
def apply_image_treatment(image, style='keynote'):
    """应用Apple Keynote风格的图像处理"""
    treatments = {
        'keynote': {
            'overlay': 'gradient',      # 渐变遮罩（增强文字可读性）
            'gradient_direction': 'bottom-to-top',
            'gradient_colors': ['rgba(0,0,0,0.6)', 'transparent'],
            'blur_background': False,
            'saturation': 1.1,          # 轻微提升饱和度
            'contrast': 1.05            # 轻微提升对比度
        },
        'minimal': {
            'overlay': 'solid',
            'overlay_color': 'rgba(255,255,255,0.9)',
            'blur_background': True,
            'blur_radius': 20
        }
    }
    return apply_effects(image, treatments[style])
```

**图文结合规则**：
```yaml
# 文字在图片上方
text_on_image:
  overlay_required: true
  min_contrast: 4.5
  safe_zones:
    - left-third      # 左三分之一
    - bottom-quarter  # 底部四分之一
  
# 图片作为背景
image_as_background:
  opacity: 0.15-0.3  # 高度透明
  blur: 10-20px
  position: right or full-bleed
```

---

## 3. Visual Annotation Processing（视觉标注处理）

**功能**：解析slides.md中的VISUAL block，验证diagram可用性，生成diagram specifications。

**核心职责**（与ppt-specialist协作）：
1. Parse VISUAL block schema from slides.md
2. Validate diagram availability (mermaid code or file from visual-designer)
3. Generate diagram specifications for specialist rendering
4. (Optional) Generate basic mermaid code for missing diagrams

### 3.1 VISUAL Block Schema

**标准格式**（由ppt-content-planner生成）：
```yaml
VISUAL:
  type: "sequence"                      # architecture|flowchart|sequence|state_machine|comparison|timeline|gantt|matrix|heatmap|scatter
  title: "用户交互流程（Browser → WASM → Backend AI）"
  priority: "critical"                  # critical|high|medium|low|optional
  data_source: "Slide 5 architecture description + Speaker notes"
  content_requirements:
    - "Show real-time interaction path with <50ms latency requirement"
    - "Show async AI task path with <2s target latency"
    - "Label key components: Browser UI, WASM Worker, Backend API, Model Service"
  notes: "Emphasize latency tradeoffs between client-side and server-side processing"
```

### 3.2 Parse and Validate

```python
import yaml
import re

def parse_visual_annotation(slide_text):
    """
    从slide text中提取VISUAL block
    
    Returns:
        visual_spec: 解析后的视觉标注对象
        validation: 验证结果
    """
    # 提取VISUAL block（YAML格式）
    visual_match = re.search(r'VISUAL:\s*\n((?:  .+\n)+)', slide_text, re.MULTILINE)
    
    if not visual_match:
        return None, {'status': 'no_visual', 'message': 'No VISUAL block found'}
    
    try:
        visual_yaml = visual_match.group(1)
        visual_spec = yaml.safe_load(visual_yaml)
        
        # 验证必填字段
        required_fields = ['type', 'title', 'priority', 'content_requirements']
        missing = [f for f in required_fields if f not in visual_spec]
        
        if missing:
            return visual_spec, {
                'status': 'invalid',
                'missing_fields': missing,
                'message': f'Missing required fields: {missing}'
            }
        
        # 验证type有效性
        valid_types = [
            'architecture', 'flowchart', 'sequence', 'state_machine',
            'comparison', 'timeline', 'gantt', 'matrix', 'heatmap', 'scatter'
        ]
        
        if visual_spec['type'] not in valid_types:
            return visual_spec, {
                'status': 'invalid',
                'message': f"Invalid type '{visual_spec['type']}'. Must be one of: {valid_types}"
            }
        
        return visual_spec, {'status': 'valid'}
    
    except yaml.YAMLError as e:
        return None, {'status': 'parse_error', 'message': str(e)}


def validate_diagram_availability(visual_spec, mermaid_code=None, diagram_file=None):
    """
    验证diagram是否可用
    
    Args:
        visual_spec: 解析后的VISUAL标注
        mermaid_code: slides.md中的mermaid代码块（可选）
        diagram_file: visual-designer提供的图片文件（可选）
    
    Returns:
        availability_report: {
            'status': 'available' | 'missing' | 'partial',
            'source': 'mermaid' | 'file' | 'none',
            'action_required': 'none' | 'generate_mermaid' | 'escalate_to_designer'
        }
    """
    if diagram_file:
        # visual-designer已提供图片
        return {
            'status': 'available',
            'source': 'file',
            'file_path': diagram_file,
            'action_required': 'none'
        }
    
    if mermaid_code:
        # slides.md包含mermaid代码
        return {
            'status': 'available',
            'source': 'mermaid',
            'mermaid_code': mermaid_code,
            'action_required': 'none'
        }
    
    # 两者都缺失
    priority = visual_spec.get('priority', 'medium')
    
    if priority == 'critical':
        # 关键diagram缺失：阻塞并上报
        return {
            'status': 'missing',
            'source': 'none',
            'action_required': 'escalate_to_creative_director',
            'message': f"Critical diagram missing: {visual_spec['title']}"
        }
    elif priority in ['high', 'medium']:
        # 尝试生成基础mermaid code
        return {
            'status': 'partial',
            'source': 'none',
            'action_required': 'generate_basic_mermaid',
            'message': f"Diagram missing, will attempt auto-generation"
        }
    else:
        # 可选diagram：标记为warning
        return {
            'status': 'missing',
            'source': 'none',
            'action_required': 'warn_only',
            'message': f"Optional diagram missing: {visual_spec['title']}"
        }
```

### 3.3 Generate Basic Mermaid Code (Optional Helper)

**注意**：此功能为辅助性质，生成的是**基础结构**，specialist需要根据Material Design tokens进一步渲染。

```python
def generate_basic_mermaid(visual_spec):
    """
    根据content_requirements生成基础mermaid code
    
    仅生成结构，specialist负责应用Material Design样式
    """
    diagram_type = visual_spec['type']
    title = visual_spec['title']
    requirements = visual_spec.get('content_requirements', [])
    
    generators = {
        'sequence': generate_sequence_diagram,
        'flowchart': generate_flowchart,
        'architecture': generate_architecture_diagram,
        'timeline': generate_timeline
    }
    
    generator = generators.get(diagram_type)
    
    if not generator:
        return {
            'status': 'unsupported',
            'message': f'Auto-generation not supported for type: {diagram_type}'
        }
    
    mermaid_code = generator(title, requirements)
    
    return {
        'status': 'generated',
        'mermaid_code': mermaid_code,
        'note': 'Basic structure only. Specialist will apply Material Design styling.'
    }


def generate_sequence_diagram(title, requirements):
    """
    生成sequence diagram基础结构
    
    示例输入 (content_requirements):
      - "Show Browser → WASM → Backend AI path"
      - "Label <50ms latency requirement"
      - "Show async task with <2s latency"
    
    示例输出 (mermaid code):
      sequenceDiagram
        participant Browser
        participant WASM
        participant Backend_AI
        Browser->>WASM: User input (<50ms)
        WASM->>Backend_AI: AI task request
        Backend_AI-->>WASM: Result (<2s)
    """
    # 提取参与者（简单正则提取）
    participants = []
    for req in requirements:
        # 匹配 "A → B" 或 "A to B" 模式
        matches = re.findall(r'([A-Z][A-Za-z0-9_\s]*?)(?:\s*(?:→|->|to)\s*([A-Z][A-Za-z0-9_\s]*?))', req)
        for match in matches:
            participants.extend([m.strip().replace(' ', '_') for m in match])
    
    participants = list(dict.fromkeys(participants))  # 去重保持顺序
    
    # 生成mermaid代码
    lines = ['sequenceDiagram']
    
    # 声明参与者
    for p in participants:
        lines.append(f'    participant {p}')
    
    # 生成交互（简化版）
    if len(participants) >= 2:
        for i in range(len(participants) - 1):
            latency = extract_latency(requirements[i] if i < len(requirements) else '')
            label = f'Request{" " + latency if latency else ""}'
            lines.append(f'    {participants[i]}->>{participants[i+1]}: {label}')
    
    return '\n'.join(lines)


def generate_flowchart(title, requirements):
    """生成flowchart基础结构"""
    lines = ['flowchart LR']
    
    # 简单解析步骤
    steps = []
    for i, req in enumerate(requirements):
        # 提取步骤名称（简化）
        step_name = req.split(':')[0] if ':' in req else req[:30]
        step_id = f'step{i+1}'
        steps.append((step_id, step_name))
    
    # 生成节点
    for step_id, step_name in steps:
        lines.append(f'    {step_id}["{step_name}"]')
    
    # 生成连接
    for i in range(len(steps) - 1):
        lines.append(f'    {steps[i][0]} --> {steps[i+1][0]}')
    
    return '\n'.join(lines)


def extract_latency(text):
    """从文本中提取延迟要求"""
    match = re.search(r'<(\d+(?:\.\d+)?)\s*(ms|s)', text)
    return match.group(0) if match else ''
```

### 3.4 Integration with ppt-specialist

**Workflow**：
```
1. ppt-content-planner生成slides.md
   └─> 包含VISUAL blocks

2. ppt-visual (本skill)处理annotations
   ├─> Parse VISUAL blocks
   ├─> Validate diagram availability
   ├─> If missing + priority=high: generate_basic_mermaid()
   └─> If missing + priority=critical: escalate to creative-director

3. ppt-specialist接收处理结果
   ├─> If source='file': 直接embed图片
   ├─> If source='mermaid': render with Material Design tokens
   └─> If status='missing' + critical: reject & escalate
```

**输出格式**（传递给specialist）：
```json
{
  "slide_number": 5,
  "visual_annotation": {
    "type": "sequence",
    "title": "用户交互流程",
    "priority": "critical",
    "content_requirements": [...]
  },
  "diagram_source": {
    "status": "available",
    "source": "mermaid",
    "mermaid_code": "sequenceDiagram\n  participant Browser\n  ...",
    "styling_requirements": {
      "apply_material_design": true,
      "color_mapping": {
        "Browser": "semantic.ui_layer",
        "WASM": "semantic.compute_layer",
        "Backend_AI": "semantic.api_layer"
      },
      "emphasis": ["latency labels"]
    }
  }
}
```

---

## 4. 集成接口规范

### 4.1 输入格式

```json
{
  "slide_intent": "强调系统性能提升",
  "content_type": "comparison",
  "emphasis": "speed",
  "brand_colors": {
    "primary": "#2563EB",
    "secondary": "#10B981"
  }
}
```

### 4.2 输出格式

**设计规范输出**：
```json
{
  "visual_design": {
    "layout": "two-column",
    "grid": {
      "columns": 12,
      "content_columns": [2, 11],
      "gutter": 24
    },
    "hierarchy": {
      "primary_message": {
        "text": "性能提升3倍",
        "size": 48,
        "weight": "bold",
        "color": "#10B981",
        "position": "center-top"
      },
      "supporting_data": {
        "size": 18,
        "color": "#475569"
      }
    },
    "icons": [
      {
        "concept": "performance",
        "icon": "bolt",
        "color": "#F59E0B",
        "size": 32
      }
    ],
    "image": {
      "url": "assets/speed-graphic.svg",
      "treatment": "keynote",
      "position": "right-half"
    }
  },
  "color_palette": {
    "primary": "#2563EB",
    "accent": "#10B981",
    "text": "#1E293B",
    "background": "#FFFFFF"
  },
  "assets": [
    {
      "type": "icon",
      "name": "bolt",
      "format": "svg",
      "license": "MIT"
    },
    {
      "type": "image",
      "source": "unsplash",
      "license": "free",
      "attribution": "Photo by XXX on Unsplash"
    }
  ]
}
```

**Diagram处理输出**（传递给specialist）：
```json
{
  "slide_number": 5,
  "visual_annotation": {
    "type": "sequence",
    "title": "用户交互流程",
    "priority": "critical",
    "content_requirements": [...]
  },
  "diagram_source": {
    "status": "available",
    "source": "mermaid",
    "mermaid_code": "sequenceDiagram\n  participant Browser\n  ...",
    "styling_requirements": {
      "apply_material_design": true,
      "color_mapping": {
        "Browser": "semantic.ui_layer",
        "WASM": "semantic.compute_layer",
        "Backend_AI": "semantic.api_layer"
      },
      "emphasis": ["latency labels"]
    }
  }
}
```

---

## 5. 最佳实践

**DO**：
- ✅ 使用高质量、高分辨率图像（≥200 DPI）
- ✅ 统一图标风格（全部线性或全部填充）
- ✅ 应用视觉层次（大小、粗细、颜色）
- ✅ 为图片添加遮罩提升文字可读性
- ✅ 保持色彩一致性（使用品牌色板）
- ✅ 利用留白引导视线

**DON'T**：
- ❌ 使用低质量、模糊的图片
- ❌ 混用多种图标风格
- ❌ 过度使用装饰元素
- ❌ 忽视图片版权和归属
- ❌ 使用分散注意力的动画
- ❌ 在深色图片上使用深色文字

---

## 6. 资源库

### 6.1 免费资源

**图标**：
- Heroicons - https://heroicons.com (MIT)
- Lucide - https://lucide.dev (ISC)
- Tabler Icons - https://tabler-icons.io (MIT)

**图片**：
- Unsplash - https://unsplash.com (Free for commercial)
- Pexels - https://pexels.com (Free)
- Pixabay - https://pixabay.com (Free)

**插图**：
- unDraw - https://undraw.co (Open source)
- Storyset - https://storyset.com (Free with attribution)

**渐变**：
- WebGradients - https://webgradients.com (MIT)
- Gradient Hunt - https://gradienthunt.com (Free)

### 6.2 参考资料

- Garr Reynolds. *Presentation Zen* (2008)
- Apple. *Human Interface Guidelines*
- Josef Müller-Brockmann. *Grid Systems in Graphic Design* (1981)
- Material Design. *Design System*
- Edward Tufte. *Envisioning Information* (1990)
- Mermaid.js Documentation - https://mermaid.js.org

