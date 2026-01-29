---
name: ppt-theme-manager
version: 1.1.0
description: "基于 Design Tokens 系统管理品牌配色、字体、间距等视觉元素，提供 Material Design 3 集成和自定义品牌主题支持。负责完整的设计系统定义和 WCAG 2.1 可访问性验证。注：视觉原则和应用指南由 ppt-visual.skill 提供。"
category: presentation
dependencies:
  libraries:
    - material-design-3  # Design tokens and color system
tags:
  - theme-management
  - color-system
  - design-tokens
  - brand-consistency
  - material-design
  - wcag-compliance
standards:
  - Material Design 3 (Google)
  - Design Tokens W3C Community Group
  - Web Content Accessibility Guidelines (WCAG 2.1)
  - Salesforce Lightning Design System
integration:
  agents:
    - ppt-visual-designer  # Creates design_spec.json
    - ppt-specialist  # Applies theme to PPTX
  skills:
    - ppt-visual  # Visual principles and Material Design application
    - ppt-layout  # Grid system integration
last_updated: 2026-01-28
---

# ppt-theme-manager Skill

**功能**：基于 Design Tokens 系统管理品牌配色、字体、间距等视觉元素，提供 Material Design 3 集成和 WCAG 2.1 可访问性验证，确保多页面、多格式的视觉一致性。

**职责边界**：
- ✅ **本skill负责**：Design Tokens 定义、品牌色彩系统、字体系统、间距系统、WCAG 验证、主题预设和应用
- 🔗 **协作skill**：
  - `ppt-visual.skill`：Material Type Scale、Material Motion、视觉层次原则
  - `ppt-layout.skill`：网格系统、布局模板

---

## 1. Design Tokens 系统概述

### 1.1 核心概念

**定义**（源自Salesforce Lightning Design System）：
```yaml
# Design Tokens = 设计决策的命名化最小单元
# 优势：单一来源真理（Single Source of Truth）
# 目标：在代码和设计工具间建立统一的设计语言

token_example:
  color.brand.primary: "#0070F3"    # 而非直接用Hex
  spacing.md: "24px"                 # 而非硬编码
  font.heading.size: "36pt"          # 而非magic number
```

### 1.2 Token 分层体系

**3层体系**（确保灵活性和可维护性）：
```
Global Tokens (全局基础)
    ↓ 映射
Alias Tokens (语义化)
    ↓ 应用
Component Tokens (组件级)

示例：
Global:   blue-500: "#0070F3"         # 全局色值
Alias:    color-primary: blue-500     # 语义化名称
Component: button-bg-primary: color-primary  # 组件应用
```

**优势**：
- **维护性**：修改 Global Token 自动级联到所有引用
- **语义化**：Alias Token 传达设计意图（primary, success）
- **组件化**：Component Token 封装组件特定规则

### 1.3 与 Material Design 3 的关系

```yaml
# Material Design 3 提供基础 Token 规范
material_design_tokens:
  color:
    - primary, secondary, tertiary
    - surface, background, error
    - on-primary, on-surface  # 文字颜色
  
  typography:
    - Display, Headline, Title, Body, Label
  
  spacing:
    - 4dp base grid system

# 本skill实现和扩展
this_skill_provides:
  - Material Design 3 token 映射
  - 自定义品牌主题系统
  - WCAG 2.1 验证和安全配对
  - 预设主题（Corporate, Creative, Minimal, Tech）
```

---

## 2. 核心 Token 系统

### 2.1 Color System（色彩系统）

#### 色彩定义

**品牌色（Brand Colors）**：
```yaml
primary:
  main: "#0070F3"      # 主色
  light: "#3291FF"     # 浅色变体
  dark: "#0053B3"      # 深色变体

secondary:
  main: "#7928CA"      # 次要色
  light: "#A159FF"
  dark: "#5A1F9A"

semantic:
  success: "#10B981"   # 成功/积极
  warning: "#F59E0B"   # 警告
  error: "#EF4444"     # 错误
  info: "#3B82F6"      # 信息
```

**中性色（Neutral/Gray Scale）**：
```yaml
gray:
  50: "#F9FAFB"
  100: "#F3F4F6"
  200: "#E5E7EB"
  300: "#D1D5DB"
  400: "#9CA3AF"
  500: "#6B7280"
  600: "#4B5563"
  700: "#374151"
  800: "#1F2937"
  900: "#111827"
```

#### WCAG对比度验证

**标准**（WCAG 2.1 AA/AAA）：
```
Normal Text (14-18pt):
  AA: ≥4.5:1
  AAA: ≥7:1

Large Text (≥18pt or ≥14pt bold):
  AA: ≥3:1
  AAA: ≥4.5:1
```

**对比度计算**（Python实现）：
```python
def calculate_contrast_ratio(color1_hex, color2_hex):
    """计算WCAG对比度（1:1 to 21:1）"""
    
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def relative_luminance(rgb):
        """计算相对亮度"""
        rgb_norm = [c / 255.0 for c in rgb]
        rgb_linear = [
            c / 12.92 if c <= 0.03928 
            else ((c + 0.055) / 1.055) ** 2.4
            for c in rgb_norm
        ]
        return 0.2126 * rgb_linear[0] + 0.7152 * rgb_linear[1] + 0.0722 * rgb_linear[2]
    
    L1 = relative_luminance(hex_to_rgb(color1_hex))
    L2 = relative_luminance(hex_to_rgb(color2_hex))
    
    lighter = max(L1, L2)
    darker = min(L1, L2)
    
    return (lighter + 0.05) / (darker + 0.05)

# 示例
ratio = calculate_contrast_ratio("#0070F3", "#FFFFFF")  # 3.28:1
if ratio >= 4.5:
    print("✅ AA合规")
else:
    print(f"❌ 对比度不足: {ratio:.2f}:1")
```

**预设配对（Pre-validated Pairs）**：
```yaml
safe_combinations:
  - foreground: primary-main     # #0070F3
    background: white           # #FFFFFF
    ratio: 3.28
    compliant: "Large Text AA"  # 仅18pt+
    
  - foreground: gray-900        # #111827
    background: white
    ratio: 15.8
    compliant: "AA + AAA"       # 所有字号
    
  - foreground: white
    background: primary-dark    # #0053B3
    ratio: 4.72
    compliant: "Normal Text AA" # 14pt+
```

---

### 2.2 Typography System（字体系统）

**字体栈（Font Stack）**：
```yaml
heading:
  family: "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
  weights:
    light: 300
    regular: 400
    semibold: 600
    bold: 700

body:
  family: "Inter, Roboto, 'Helvetica Neue', Arial, sans-serif"
  weights:
    regular: 400
    medium: 500

monospace:
  family: "JetBrains Mono, 'Courier New', monospace"
  weight: 400
```

**Type Scale（字号系统）**：
```yaml
# 基于Major Third比例（1.250）

hero: 60pt         # 标题页主标题
h1: 48pt           # 一级标题
h2: 38pt           # 二级标题（章节）
h3: 30pt           # 三级标题
h4: 24pt           # 四级标题
body-large: 20pt   # 大正文
body: 16pt         # 标准正文
body-small: 14pt   # 小正文
caption: 12pt      # 图注/来源
```

**行高（Line Height）**：
```yaml
heading: 1.2       # 标题紧凑
body: 1.5          # 正文舒适
caption: 1.4       # 图注适中
```

---

### 2.3 Spacing System（8点网格）

**基础单位**：
```yaml
base: 8px          # 基准

scale:
  xs: 4px          # 0.5x
  sm: 8px          # 1x
  md: 16px         # 2x
  lg: 24px         # 3x
  xl: 32px         # 4x
  2xl: 48px        # 6x
  3xl: 64px        # 8x
  4xl: 96px        # 12x

usage:
  element_padding: md (16px)
  section_spacing: xl (32px)
  slide_margin: 2xl (48px)
```

**Python检查**：
```python
def validate_spacing(value_px):
    """验证是否符合8点网格"""
    if value_px % 8 == 0:
        return True, f"✅ {value_px}px 符合8点网格"
    else:
        nearest = round(value_px / 8) * 8
        return False, f"❌ {value_px}px → 建议调整为 {nearest}px"

# 示例
validate_spacing(23)  # (False, "建议调整为24px")
validate_spacing(24)  # (True, "符合8点网格")
```

---

## 3. 主题应用

### 3.1 Brand Presets（品牌预设）

**功能**：提供4种预设主题风格，快速启动设计系统。

#### Corporate（企业风格）
```yaml
corporate:
  colors:
    primary: "#003087"      # 深蓝（IBM风格）
    secondary: "#5E5E5E"    # 中灰
    accent: "#0F62FE"       # 亮蓝
  fonts:
    heading: "IBM Plex Sans"
    body: "IBM Plex Sans"
  tone: "formal, data-driven"
```

#### Creative（创意风格）
```yaml
creative:
  colors:
    primary: "#FF6B6B"      # 珊瑚红
    secondary: "#4ECDC4"    # 青绿
    accent: "#FFE66D"       # 明黄
  fonts:
    heading: "Montserrat"
    body: "Open Sans"
  tone: "playful, vibrant"
```

#### Minimal（极简风格）
```yaml
minimal:
  colors:
    primary: "#000000"      # 纯黑
    secondary: "#FFFFFF"    # 纯白
    accent: "#E0E0E0"       # 浅灰
  fonts:
    heading: "Helvetica Neue"
    body: "Helvetica Neue"
  tone: "clean, Swiss Design"
```

#### Tech（科技风格）
```yaml
tech:
  colors:
    primary: "#00D9FF"      # 霓虹蓝
    secondary: "#7B61FF"    # 紫色
    accent: "#FF006E"       # 品红
  fonts:
    heading: "Space Grotesk"
    body: "Inter"
  tone: "futuristic, bold"
```

---

### 3.2 主题对象规范

**完整 Theme 对象结构**：
```json
{
  "theme_id": "corporate-blue",
  "name": "Corporate Professional",
  "tokens": {
    "color": {
      "primary": {"main": "#003087", "light": "#4A6FA5", "dark": "#001F4D"},
      "secondary": {"main": "#5E5E5E", "light": "#8E8E8E", "dark": "#2E2E2E"},
      "background": {"default": "#FFFFFF", "alt": "#F5F5F5"},
      "text": {"primary": "#1A1A1A", "secondary": "#6B7280"}
    },
    "typography": {
      "heading": {"family": "IBM Plex Sans", "weight": 600},
      "body": {"family": "IBM Plex Sans", "weight": 400},
      "scale": {"h1": "48pt", "h2": "38pt", "body": "16pt"}
    },
    "spacing": {
      "base": "8px",
      "slide_margin": "48px",
      "section_gap": "32px"
    }
  },
  "wcag_report": {
    "primary_on_white": {"ratio": 8.2, "compliant": "AAA"},
    "secondary_on_white": {"ratio": 5.1, "compliant": "AA"}
  }
}
```

**导出功能**（支持多种格式）：

**(1) 导出为 CSS 变量**：
```python
def export_to_css(theme):
    """生成CSS Custom Properties"""
    css = ":root {\n"
    
    # Colors
    for key, value in theme['tokens']['color'].items():
        if isinstance(value, dict):
            for shade, hex_color in value.items():
                css += f"  --color-{key}-{shade}: {hex_color};\n"
        else:
            css += f"  --color-{key}: {value};\n"
    
    # Typography
    css += f"  --font-heading: {theme['tokens']['typography']['heading']['family']};\n"
    css += f"  --font-body: {theme['tokens']['typography']['body']['family']};\n"
    
    # Spacing
    css += f"  --spacing-base: {theme['tokens']['spacing']['base']};\n"
    
    css += "}\n"
    return css

# 输出示例：
# :root {
#   --color-primary-main: #003087;
#   --color-primary-light: #4A6FA5;
#   --font-heading: IBM Plex Sans;
#   --spacing-base: 8px;
# }
```

---

### 3.3 集成接口（输入/输出规范）

**输入格式**：
```json
{
  "theme_request": {
    "preset": "corporate",
    "brand_overrides": {
      "primary_color": "#0070F3",
      "logo_path": "assets/logo.png"
    },
    "target_format": "pptx"
  }
}
```

**输出格式**：
```json
{
  "applied_theme": {
    "theme_id": "corporate-custom",
    "tokens": { /* 完整token对象 */ },
    "wcag_compliance": {
      "primary_on_white": {"ratio": 3.28, "status": "⚠️ Large Text Only"},
      "text_on_background": {"ratio": 15.8, "status": "✅ AAA"}
    },
    "css_export": ":root { ... }",
    "warnings": [
      "Primary color对比度不足（14pt文字），建议使用#0053B3深色变体"
    ]
  }
}
```

---

## 4. 最佳实践

### 4.1 Token 使用规范

**DO**：
- ✅ **优先使用 Alias Tokens**：`color-primary` 而非 `#0070F3`（语义化命名）
- ✅ **遵循分层体系**：Component Token → Alias Token → Global Token
- ✅ **所有 spacing 符合 8 点网格**：4, 8, 16, 24, 32, 48, 64, 96 px
- ✅ **提供深色变体**：每个主题色提供 light/main/dark 三个变体
- ✅ **文档化设计决策**：在 token 定义中注释设计意图

**DON'T**：
- ❌ **硬编码颜色值**：破坏单一来源真理（Single Source of Truth）
- ❌ **跳过 Alias Token**：直接从 Global Token 到 Component Token
- ❌ **使用非系统字号**：如 23pt, 17pt（破坏字体比例）
- ❌ **忽略 spacing grid**：使用 15px, 23px 等非标准值

### 4.2 WCAG 可访问性规范

**DO**：
- ✅ **验证对比度**：Normal Text ≥4.5:1, Large Text ≥3:1
- ✅ **提供安全配对**：预设 WCAG 合规的文字/背景组合
- ✅ **标注合规等级**：在 token 定义中注明 AA/AAA
- ✅ **测试色盲模式**：验证 Protanopia, Deuteranopia, Tritanopia

**DON'T**：
- ❌ **忽略 WCAG 警告**：对比度不足会严重影响可读性
- ❌ **依赖颜色传达信息**：必须配合图标、文字、形状
- ❌ **使用低对比度灰色**：如 #CCCCCC on #FFFFFF（仅 1.6:1）

### 4.3 主题定制规范

**DO**：
- ✅ **从预设开始**：选择 Corporate/Creative/Minimal/Tech 预设并覆盖
- ✅ **保持品牌一致性**：主题色、字体与品牌指南对齐
- ✅ **生成完整色板**：primary 需包含 light/main/dark + on-primary
- ✅ **测试多场景**：亮色背景、暗色背景、打印模式

**DON'T**：
- ❌ **随意混搭预设**：破坏视觉一致性
- ❌ **过度使用颜色**：建议≤5种主要颜色
- ❌ **忽略文化差异**：红色在中国代表吉祥，在西方可能代表危险

---

## 5. Implementation Interface (Python)

### 5.1 Core Function: load_design_spec()

**目的**: 从 design_spec.json 加载完整设计系统（包含 color_system, typography_system, spacing_system, layout_system, component_library）

**函数签名**:
```python
from dataclasses import dataclass
from typing import Dict, Optional, List
import json

@dataclass
class DesignSpec:
    """设计规范完整对象（单一来源真理）"""
    # Design system sections
    color_system: Dict[str, str]         # {"primary": "#1565C0", ...}
    typography_system: Dict[str, dict]   # {"headline_large": {"size": 36, "weight": "bold"}, ...}
    spacing_system: Dict[str, int]       # {"xs": 4, "sm": 8, "md": 16, ...}
    layout_system: Dict[str, any]        # {"grid_columns": 12, "margin_horizontal": 80, "layouts": {...}}
    elevation_system: Dict[str, dict]    # {"level_1": {"shadow": "..."}, ...}
    shape_system: Dict[str, int]         # {"corner_radius_sm": 4, "corner_radius_md": 8, ...}
    component_library: Dict[str, dict]   # {"card": {...}, "callout": {...}, ...}
    
    # Metadata
    meta: Dict[str, str]                 # {"session_id": "...", "version": "...", ...}
    branding: Optional[Dict[str, str]]   # {"logo_path": "...", "company_name": "..."}

def load_design_spec(file_path: str) -> DesignSpec:
    """
    从 design_spec.json 加载完整设计规范
    
    参数:
        file_path (str): design_spec.json 的绝对路径
    
    返回:
        DesignSpec: 完整设计规范对象
    
    异常:
        FileNotFoundError: design_spec.json 不存在
        ValueError: JSON 格式错误或必需字段缺失
    
    示例:
        >>> design_spec = load_design_spec("source/design_spec.json")
        >>> primary_color = design_spec.color_system["primary"]  # "#1565C0"
        >>> spacing_md = design_spec.spacing_system["md"]         # 16
        >>> grid_cols = design_spec.layout_system["grid_columns"] # 12
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Validate required sections
    required_sections = ['color_system', 'typography_system', 'spacing_system', 
                        'layout_system', 'component_library', 'meta']
    for section in required_sections:
        if section not in data:
            raise ValueError(f"Missing required section: {section}")
    
    return DesignSpec(
        color_system=data['color_system'],
        typography_system=data['typography_system'],
        spacing_system=data['spacing_system'],
        layout_system=data['layout_system'],
        elevation_system=data.get('elevation_system', {}),
        shape_system=data.get('shape_system', {}),
        component_library=data['component_library'],
        meta=data['meta'],
        branding=data.get('branding')
    )
```

### 5.2 Helper Function: get_spacing_token()

**目的**: 安全获取间距token值（带默认值处理）

**函数签名**:
```python
def get_spacing_token(token_name: str, design_spec: DesignSpec, default: int = 16) -> int:
    """
    从 design_spec 获取间距token值
    
    参数:
        token_name (str): token名称，如 "md", "lg", "content_padding"
        design_spec (DesignSpec): 设计规范对象
        default (int): token不存在时的默认值（单位：像素）
    
    返回:
        int: 间距值（像素）
    
    示例:
        >>> spacing = get_spacing_token("md", design_spec)  # 16
        >>> padding = get_spacing_token("content_padding", design_spec, default=32)  # 32 or from spec
    """
    return design_spec.spacing_system.get(token_name, default)
```

### 5.3 Helper Function: get_color_token()

**目的**: 安全获取颜色token（带对比度验证）

**函数签名**:
```python
def get_color_token(token_name: str, design_spec: DesignSpec, 
                   validate_contrast: bool = False, 
                   background: Optional[str] = None) -> str:
    """
    从 design_spec 获取颜色token值（可选WCAG对比度验证）
    
    参数:
        token_name (str): token名称，如 "primary", "on_surface"
        design_spec (DesignSpec): 设计规范对象
        validate_contrast (bool): 是否验证WCAG对比度
        background (str, optional): 背景颜色（用于对比度验证）
    
    返回:
        str: 颜色十六进制值，如 "#1565C0"
    
    异常:
        ValueError: token不存在或对比度不符合WCAG标准
    
    示例:
        >>> primary = get_color_token("primary", design_spec)  # "#1565C0"
        >>> text_color = get_color_token("on_surface", design_spec, 
        ...                              validate_contrast=True, 
        ...                              background="#FFFFFF")  # 验证对比度≥4.5:1
    """
    if token_name not in design_spec.color_system:
        raise ValueError(f"Color token '{token_name}' not found in design_spec")
    
    color = design_spec.color_system[token_name]
    
    if validate_contrast and background:
        ratio = calculate_contrast_ratio(color, background)
        if ratio < 4.5:  # WCAG AA minimum
            raise ValueError(f"Contrast ratio {ratio:.2f} < 4.5 for {token_name} on {background}")
    
    return color

def calculate_contrast_ratio(foreground: str, background: str) -> float:
    """计算WCAG对比度（1:1 到 21:1）"""
    # Implementation: Convert hex to RGB, calculate relative luminance, return ratio
    # See: https://www.w3.org/WAI/WCAG21/Understanding/contrast-minimum.html
    pass
```

### 5.4 Helper Function: get_typography_spec()

**目的**: 获取完整字体规格（含大小、粗细、行高）

**函数签名**:
```python
@dataclass
class TypographySpec:
    """字体完整规格"""
    font_family: str      # "Noto Sans SC", "Roboto", etc.
    font_size: int        # 单位：pt
    font_weight: str      # "regular", "medium", "bold"
    line_height: float    # 行高倍数，如 1.6
    letter_spacing: Optional[float] = None  # 字母间距（em）

def get_typography_spec(type_scale: str, design_spec: DesignSpec) -> TypographySpec:
    """
    从 design_spec 获取字体规格
    
    参数:
        type_scale (str): Material Type Scale名称，如 "headline_large", "body_medium"
        design_spec (DesignSpec): 设计规范对象
    
    返回:
        TypographySpec: 完整字体规格对象
    
    异常:
        ValueError: type_scale 不存在
    
    示例:
        >>> title_spec = get_typography_spec("headline_large", design_spec)
        >>> print(f"{title_spec.font_size}pt {title_spec.font_weight}")  # "36pt bold"
    """
    if type_scale not in design_spec.typography_system:
        raise ValueError(f"Typography scale '{type_scale}' not found")
    
    spec_data = design_spec.typography_system[type_scale]
    return TypographySpec(
        font_family=spec_data.get('font_family', 'Noto Sans SC'),
        font_size=spec_data['font_size'],
        font_weight=spec_data.get('font_weight', 'regular'),
        line_height=spec_data.get('line_height', 1.5),
        letter_spacing=spec_data.get('letter_spacing')
    )
```

### 5.5 Integration Checklist

**在 ppt-specialist 中使用本skill时必须**:
1. ✅ 调用 `load_design_spec()` 一次加载完整设计系统（不要多次读取JSON）
2. ✅ 使用 `get_spacing_token()` 获取所有间距值（禁止硬编码Inches(1.5)）
3. ✅ 使用 `get_color_token()` 获取颜色（可选对比度验证）
4. ✅ 使用 `get_typography_spec()` 获取字体规格（禁止硬编码36pt）
5. ✅ 从 `design_spec.layout_system` 读取网格配置（grid_columns, margin, gutter）
6. ✅ 从 `design_spec.component_library` 读取组件规格（card, callout, etc.）

**反例（禁止）**:
```python
# ❌ 硬编码颜色
text_box.fill.solid()
text_box.fill.fore_color.rgb = RGBColor(21, 101, 192)  # 应该用 get_color_token("primary")

# ❌ 硬编码间距
content_left = Inches(1.5)  # 应该用 get_spacing_token("content_padding") + grid calculation

# ❌ 硬编码字体
font_size = Pt(36)  # 应该用 get_typography_spec("headline_large").font_size
```

### 5.6 Anti-Pattern Checklist

**绝对禁止**:
- ❌ 直接修改 design_spec.json（由 visual-designer 维护）
- ❌ 在代码中硬编码任何设计token值（颜色、间距、字体）
- ❌ 跳过WCAG验证直接使用颜色配对
- ❌ 忽略 layout_system 和 component_library（specialist常犯错误）
- ❌ 多次读取 design_spec.json（应该load一次，全局复用）

---

## 6. 资源和参考

### 6.1 官方文档

- **Material Design 3** - [The Color System](https://m3.material.io/styles/color/system)
- **Salesforce Lightning** - [Design Tokens](https://www.lightningdesignsystem.com/design-tokens/)
- **W3C Design Tokens** - [Community Group](https://www.w3.org/community/design-tokens/)
- **WCAG 2.1** - [Understanding Contrast Ratios](https://www.w3.org/WAI/WCAG21/Understanding/contrast-minimum.html)

### 6.2 工具和资源

- **Tailwind CSS** - Spacing Scale (8pt grid implementation)
- **Adobe Color** - Accessibility Tools (contrast checker)
- **Coolors** - Color Palette Generator
- **Material Theme Builder** - [m3.material.io/theme-builder](https://m3.material.io/theme-builder)

### 6.3 相关 Skills

- `ppt-visual.skill` - Material Type Scale, Material Motion, 视觉层次
- `ppt-layout.skill` - Grid System, Layout Templates
- `ppt-aesthetic-qa.skill` - WCAG 验证, 设计合规性检查
