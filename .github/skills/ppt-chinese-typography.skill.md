---
name: ppt-chinese-typography
version: 1.1.0
description: "处理中文字体嵌入、字符覆盖验证、排版规则应用。核心功能包括：Noto Sans SC 字体子集生成（压缩至500KB）、字符覆盖率验证（自动检测缺字）、中文排版规范应用（行高1.6-1.8、最小20pt字号、baseline对齐）、跨平台兼容性保证（Windows/macOS/WPS/Google Slides）、字体嵌入策略（subset/full/system模式）、fallback字体链配置。"
category: presentation
dependencies:
  libraries:
    - Noto Sans SC  # Google Fonts - 中文字体（2.004版本）
  python_packages:
    - fonttools  # Font subsetting and manipulation
    - brotli  # WOFF2 compression
    - lxml  # XML manipulation for PPTX structure
tags:
  - chinese-typography
  - font-subsetting
  - cross-platform
  - noto-sans-sc
  - character-coverage
  - gb2312
  - cjk-layout
  - baseline-alignment
  - orphan-fix
standards:
  - W3C中文排版需求 (Requirements for Chinese Text Layout)
  - GB/T 18358-2009 (信息技术 中文Linux系统字体配置规范)
  - Apple HIG Chinese Typography Guidelines
  - Microsoft Typography Guidelines for CJK
  - GB/T 18358-2009 (中文出版物夹用英文的编辑规范)
integration:
  agents:
    - ppt-specialist  # Primary consumer for font embedding
  skills:
    - ppt-export  # Uses font subset in PPTX generation
    - ppt-theme-manager  # Typography system integration
    - ppt-markdown-parser  # Extract used characters from slides.md
last_updated: 2026-01-28
---

# ppt-chinese-typography Skill

**功能**：处理中文字体嵌入、字符覆盖验证、排版规则应用，确保PPT在跨平台环境下正确显示中文内容。

**职责边界**：
- ✅ **本skill负责**：字体子集生成（fonttools）、字符覆盖验证、中文排版规则（行高、字距、baseline对齐）、字体嵌入到PPTX、跨平台兼容性测试
- 🔗 **协作skill**：
  - `ppt-export.skill`：调用字体子集嵌入到PPTX最终交付物
  - `ppt-theme-manager.skill`：提供typography system规范（font-family、font-size）
  - `ppt-markdown-parser.skill`：提供slides.md内容用于字符集提取

---

## 1. 核心字体处理功能

### 1.1 Noto Sans SC Subset Generation（字体子集生成）

**目标**：将完整的Noto Sans SC字体（~20MB）压缩为项目专用子集（~500KB），包含slides.md中实际使用的字符。

**标准字符集**（基础模式）：
```yaml
basic_mode:
  chinese_chars: GB2312 常用3500字
  latin_chars: A-Z, a-z, 0-9, 基础标点
  coverage: ~95% 常见业务场景
  file_size: ~500KB

advanced_mode:
  chinese_chars: 动态检测slides.md使用的字符
  latin_chars: 完整ASCII + 扩展符号
  coverage: 100% 当前项目
  file_size: 200KB-800KB（取决于内容）
```

**fonttools subset命令**：
```bash
# 基础模式：预定义字符集
pyftsubset NotoSansSC-Regular.otf \
  --unicodes="U+4E00-U+9FA5" \
  --unicodes="U+0020-U+007E" \
  --layout-features="*" \
  --flavor=woff2 \
  --output-file=NotoSansSC-Subset.woff2

# 高级模式：动态字符检测
python3 << 'EOF'
import re
from fontTools import subset

# 提取slides.md中所有中文字符
with open('slides.md', 'r', encoding='utf-8') as f:
    content = f.read()
    chinese_chars = set(re.findall(r'[\u4e00-\u9fff]', content))
    latin_chars = set(re.findall(r'[A-Za-z0-9]', content))

# 生成unicode范围
unicodes = [f"U+{ord(c):04X}" for c in chinese_chars | latin_chars]

# Subset字体
options = subset.Options()
options.flavor = 'woff2'
options.layout_features = ['*']

font = subset.load_font('NotoSansSC-Regular.otf', options)
subsetter = subset.Subsetter(options=options)
subsetter.populate(unicodes=unicodes)
subsetter.subset(font)
subset.save_font(font, 'NotoSansSC-Custom.woff2', options)
EOF
```

**字符集检测算法**：
```python
def extract_used_characters(slides_md_path):
    """
    提取slides.md中使用的所有字符
    返回: 按类型分类的字符集
    """
    with open(slides_md_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    char_sets = {
        'chinese': set(re.findall(r'[\u4e00-\u9fff]', content)),
        'latin': set(re.findall(r'[A-Za-z]', content)),
        'digits': set(re.findall(r'[0-9]', content)),
        'punctuation': set(re.findall(r'[，。！？；：""''（）、—…《》]', content)),
        'ascii_punct': set(re.findall(r'[.,!?;:\'"()\-]', content))
    }
    
    # 统计信息
    stats = {
        'total_chars': sum(len(s) for s in char_sets.values()),
        'chinese_count': len(char_sets['chinese']),
        'latin_count': len(char_sets['latin']),
        'estimated_subset_size': len(char_sets['chinese']) * 0.15  # KB
    }
    
    return char_sets, stats

# 示例输出
# char_sets = {
#     'chinese': {'在', '线', 'P', 'S', '算', '法', ...},
#     'latin': {'O', 'n', 'l', 'i', 'n', 'e', ...},
#     ...
# }
# stats = {'total_chars': 1280, 'chinese_count': 850, 'estimated_subset_size': 127.5}
```

---

### 1.2 Font Coverage Validation（字符覆盖验证）

**功能**：确保嵌入的字体包含所有使用的字符，避免PPTX中出现"□"（缺字）。

**验证流程**：
```python
from fontTools.ttLib import TTFont

def validate_font_coverage(font_path, slides_md_path):
    """
    验证字体覆盖率
    
    Args:
        font_path: 字体文件路径（.otf/.woff2）
        slides_md_path: slides.md路径
    
    Returns:
        coverage_report: {
            'status': 'pass' | 'fail',
            'coverage_rate': 0.98,
            'total_chars': 1280,
            'covered_chars': 1254,
            'missing_chars': ['𠮷', '𣎴'],  # 罕见字
            'missing_details': [
                {'char': '𠮷', 'unicode': 'U+20BB7', 'location': 'slide 5, line 3'}
            ]
        }
    """
    # 加载字体
    font = TTFont(font_path)
    cmap = font.getBestCmap()  # Unicode -> Glyph映射
    supported_chars = set(cmap.keys())
    
    # 提取使用的字符
    used_chars, _ = extract_used_characters(slides_md_path)
    all_used = set()
    for char_set in used_chars.values():
        all_used.update(ord(c) for c in char_set)
    
    # 检查覆盖
    missing = all_used - supported_chars
    coverage_rate = (len(all_used) - len(missing)) / len(all_used)
    
    if missing:
        # 定位缺失字符位置
        with open(slides_md_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        missing_details = []
        for unicode_val in missing:
            char = chr(unicode_val)
            for i, line in enumerate(lines, 1):
                if char in line:
                    missing_details.append({
                        'char': char,
                        'unicode': f'U+{unicode_val:04X}',
                        'location': f'line {i}: {line.strip()[:50]}'
                    })
                    break
    
    return {
        'status': 'pass' if not missing else 'fail',
        'coverage_rate': coverage_rate,
        'total_chars': len(all_used),
        'covered_chars': len(all_used) - len(missing),
        'missing_chars': [chr(u) for u in missing],
        'missing_details': missing_details if missing else []
    }

# 示例输出
# {
#   'status': 'fail',
#   'coverage_rate': 0.998,
#   'total_chars': 1280,
#   'covered_chars': 1278,
#   'missing_chars': ['𠮷', '𣎴'],
#   'missing_details': [
#       {'char': '𠮷', 'unicode': 'U+20BB7', 'location': 'line 45: 用户姓名：吉𠮷'}
#   ]
# }
```

**自动修复策略**：
```python
def auto_fix_missing_chars(missing_chars, mode='fallback'):
    """
    处理缺失字符
    
    Modes:
        fallback: 使用形近字替换（𠮷 → 吉）
        expand: 扩展字体子集包含缺失字符
        warn: 仅警告，不修复
    """
    if mode == 'fallback':
        # 形近字映射表
        fallback_map = {
            '𠮷': '吉',  # CJK扩展A
            '𣎴': '木',
            '囍': '喜喜',
        }
        return fallback_map.get(missing_chars[0], '?')
    
    elif mode == 'expand':
        # 重新生成subset，包含缺失字符
        # （需要完整Noto Sans SC字体）
        pass
    
    elif mode == 'warn':
        return {
            'action': 'no_fix',
            'warning': f'Missing {len(missing_chars)} rare characters',
            'recommendation': 'Use fallback characters or expand font subset'
        }
```

---

## 2. 中文排版规范

### 2.1 核心排版原则

**核心原则**（源自《中文排版需求》W3C标准 + Apple中文排版指南）：

#### 2.1.1 行高与字距

```yaml
line_height:
  body_text: 1.6-1.8        # 中文需要比英文更宽松（英文1.4-1.5）
  titles: 1.2-1.4           # 标题可以紧凑
  
letter_spacing:
  normal: 0                 # 中文不需要额外字距
  emphasis: 0.05em          # 强调时轻微增加
  
word_spacing:
  chinese_only: 0           # 纯中文无词间距
  mixed_cn_en: 0.25em       # 中英混排时增加英文词间距
```

#### 2.1.2 字号标准

```yaml
minimum_font_sizes:
  body_text: 20pt           # 演示场景（英文14pt足够，中文需更大）
  subtitle: 28pt
  title: 36pt
  hero_title: 48-60pt
  
rationale: |
  中文笔画复杂，小字号下难以识别
  投影场景需考虑远距离可读性
  
reference_standard:
  - Apple Human Interface Guidelines（中文字号建议）
  - Microsoft PowerPoint 中文模板标准
  - GB/T 18358-2009《中文出版物夹用英文的编辑规范》
```

### 2.2 中英混排Baseline对齐

**问题**：中英文基线不一致导致参差不齐。

**解决方案**：
```python
def apply_baseline_alignment(text_runs):
    """
    应用中英混排baseline对齐
    
    Noto Sans SC特性：
    - 内置中英基线对齐（无需手动调整）
    - Latin字符采用OpenType features自动对齐
    
    对于其他字体（如微软雅黑）：
    - 中文baseline: 0
    - 英文baseline: -0.1em（向下微调）
    """
    for run in text_runs:
        if run.font_family != 'Noto Sans SC':
            # 检测语言
            if is_latin(run.text):
                run.baseline_shift = -0.1  # em单位
            else:
                run.baseline_shift = 0
    
    return text_runs

def is_latin(text):
    """检测是否为拉丁字符"""
    return all(ord(c) < 0x4E00 or ord(c) > 0x9FFF for c in text if c.strip())
```

### 2.3 避免孤字（Widows/Orphans）

**中文特有规则**：
```yaml
avoid_single_char_line:
  - 标点符号不能单独成行（。！？，等）
  - 单字不能独占一行（如"的"、"了"）
  
implementation:
  - 检测行尾标点
  - 自动调整上一行宽度，强制标点与前文同行
  
example:
  ❌ 错误:
    "这是一个完整的句子
     。"
  
  ✅ 正确:
    "这是一个完整的
     句子。"
```

**自动修复算法**：
```python
def fix_orphan_punctuation(text_box):
    """
    修复孤立标点
    """
    lines = text_box.text.split('\n')
    
    for i, line in enumerate(lines):
        # 检测孤立标点
        if line.strip() in '。！？，、；：':
            # 合并到上一行
            if i > 0:
                lines[i-1] += line.strip()
                lines[i] = ''
    
    # 重建文本
    text_box.text = '\n'.join(l for l in lines if l)
    
    # 调整行宽（可能需要减小以容纳额外标点）
    text_box.width *= 0.95
```

---

## 3. 跨平台兼容性

**目标平台**：
- Windows PowerPoint 2019+ / Microsoft 365
- macOS Keynote 10.0+
- WPS Office 2019+
- Google Slides（通过PDF导出）

### 3.1 字体嵌入策略

```yaml
embedding_modes:
  full_embed:
    description: 完整嵌入字体文件到PPTX
    pros: 100%兼容，跨平台一致
    cons: 文件大（+500KB per font）
    use_case: 最终交付物
  
  subset_embed:
    description: 仅嵌入使用的字符
    pros: 文件小（200-500KB），兼容性好
    cons: 需要fonttools处理
    use_case: 默认模式（推荐）
  
  system_font:
    description: 依赖系统安装的字体
    pros: 文件最小
    cons: 跨平台不一致（Windows缺Noto Sans SC）
    use_case: 内部协作（统一环境）
```

**PPTX嵌入配置**：
```python
from pptx import Presentation
from pptx.util import Pt

def embed_font_to_pptx(pptx_path, font_path, font_name):
    """
    嵌入字体到PPTX（python-pptx库）
    
    注意：python-pptx不直接支持字体嵌入
    需要手动操作PPTX的XML结构
    """
    import zipfile
    import os
    from lxml import etree
    
    # 1. 添加字体文件到PPTX（ZIP结构）
    with zipfile.ZipFile(pptx_path, 'a') as pptx_zip:
        pptx_zip.write(font_path, f'ppt/fonts/{os.path.basename(font_path)}')
    
    # 2. 修改presentation.xml添加字体引用
    # （详细XML操作省略，需要修改[Content_Types].xml和ppt/presentation.xml）
    
    # 3. 验证嵌入
    with zipfile.ZipFile(pptx_path, 'r') as pptx_zip:
        font_files = [f for f in pptx_zip.namelist() if f.startswith('ppt/fonts/')]
        assert len(font_files) > 0, "Font embedding failed"
    
    return {
        'embedded': True,
        'font_name': font_name,
        'font_file': os.path.basename(font_path),
        'file_size': os.path.getsize(font_path)
    }
```

### 3.2 字体Fallback链

**策略**：定义字体回退顺序，确保在目标平台未安装Noto Sans SC时有备选。

```yaml
font_stack:
  primary: "Noto Sans SC"
  fallbacks:
    - "PingFang SC"        # macOS默认
    - "Microsoft YaHei"    # Windows默认
    - "SimSun"             # Windows备选
    - "sans-serif"         # 系统默认

pptx_implementation:
  # PowerPoint支持font substitution table
  # 在theme.xml中定义：
  <a:fontScheme name="Custom">
    <a:majorFont>
      <a:latin typeface="Noto Sans SC"/>
      <a:ea typeface="Noto Sans SC"/>
      <a:cs typeface="Noto Sans SC"/>
    </a:majorFont>
    <a:minorFont>
      <a:latin typeface="Noto Sans SC"/>
      <a:ea typeface="Noto Sans SC"/>
      <a:cs typeface="Noto Sans SC"/>
    </a:minorFont>
    <a:font script="Hans" typeface="Noto Sans SC">
      <a:altFont typeface="PingFang SC"/>
      <a:altFont typeface="Microsoft YaHei"/>
    </a:font>
  </a:fontScheme>
```

### 3.3 渲染测试（可选，高级特性）

**自动化测试脚本**：
```bash
#!/bin/bash
# 跨平台渲染一致性测试

# 1. Windows PowerPoint测试（需Windows VM或Wine）
convert_pptx_to_pdf_windows() {
    powershell.exe -Command "
        \$ppt = New-Object -ComObject PowerPoint.Application
        \$pres = \$ppt.Presentations.Open('$1')
        \$pres.SaveAs('output_windows.pdf', 32)  # 32 = PDF格式
        \$pres.Close()
        \$ppt.Quit()
    "
}

# 2. macOS Keynote测试
convert_pptx_to_pdf_macos() {
    osascript -e "
        tell application \"Keynote\"
            open POSIX file \"$1\"
            export front document to POSIX file \"output_macos.pdf\" as PDF
            close front document
        end tell
    "
}

# 3. LibreOffice测试（跨平台）
convert_pptx_to_pdf_libre() {
    libreoffice --headless --convert-to pdf "$1" --outdir .
}

# 4. 对比PDF差异（ImageMagick）
compare -metric AE \
    output_windows.pdf[0] \
    output_macos.pdf[0] \
    diff.png

# 如果差异像素 < 1000，认为渲染一致
```

---

## 4. 集成接口

### 4.1 输入格式

```yaml
slides_md:
  path: "docs/presentations/online-ps/slides.md"
  encoding: "utf-8"
  
base_font:
  path: "fonts/NotoSansSC-Regular.otf"
  version: "2.004"
  source: "Google Fonts"

config:
  mode: "subset"  # subset | full | system
  coverage_threshold: 0.98
  auto_fix_missing: true
  fallback_fonts: ["PingFang SC", "Microsoft YaHei"]
```

### 4.2 输出格式

```yaml
font_subset:
  path: "docs/presentations/online-ps/fonts/NotoSansSC-Subset.woff2"
  size: 487KB
  format: "woff2"

coverage_report:
  path: "docs/presentations/online-ps/coverage_report.json"
  content:
    status: "pass"
    coverage_rate: 0.998
    total_chars: 1280
    covered_chars: 1278
    missing_chars: ["𠮷", "𣎴"]
    warnings:
      - "2 rare CJK Extension A characters not covered"
      - "Consider using fallback: 𠮷 → 吉"

embedding_config:
  path: "docs/presentations/online-ps/embedding_config.json"
  content:
    font_name: "Noto Sans SC"
    font_file: "NotoSansSC-Subset.woff2"
    embedding_mode: "subset"
    fallback_stack: ["PingFang SC", "Microsoft YaHei", "SimSun"]
    platform_compatibility:
      windows: "PowerPoint 2019+"
      macos: "Keynote 10.0+"
      wps: "WPS Office 2019+"
```

---

## 5. 最佳实践

### 5.1 字体处理规范

**DO**：
- ✅ **优先使用subset模式**：平衡文件大小和兼容性（500KB vs 20MB完整字体）
- ✅ **验证字符覆盖**：在PPTX生成前运行 `validate_font_coverage()`
- ✅ **定义fallback字体**：确保跨平台一致性（Noto Sans SC → PingFang SC → Microsoft YaHei → SimSun）
- ✅ **测试罕见字**：使用coverage check捕获CJK扩展A/B字符
- ✅ **使用WOFF2格式**：相比OTF/TTF减少30%文件大小

**DON'T**：
- ❌ **不要硬编码字体路径**：使用相对路径或配置文件（避免跨平台路径问题）
- ❌ **不要忽略缺失字符警告**：可能导致PPTX显示"□"（tofu字符）
- ❌ **不要使用系统字体模式交付**：跨平台不可控（Windows无Noto Sans SC）
- ❌ **不要混用多种中文字体**：保持一致性（除非有特殊设计需求）
- ❌ **不要跳过coverage验证**：直到QA阶段才发现缺字问题代价高昂

### 5.2 中文排版规范

**DO**：
- ✅ **遵循最小字号标准**：body ≥20pt, title ≥36pt（演示场景远距离可读）
- ✅ **设置合适行高**：line-height ≥1.6（中文需比英文1.4更宽松）
- ✅ **修复孤立标点**：使用 `fix_orphan_punctuation()` 避免标点单独成行
- ✅ **应用baseline对齐**：中英混排时确保基线一致（Noto Sans SC自带对齐特性）
- ✅ **测试不同平台渲染**：Windows PowerPoint、macOS Keynote、WPS Office

**DON'T**：
- ❌ **不要使用过小字号**：<20pt在投影场景难以识别（中文笔画复杂）
- ❌ **不要使用过紧行高**：<1.4会导致中文字符上下挤压
- ❌ **不要忽略孤立标点**：影响专业性和阅读体验
- ❌ **不要强制对齐非等宽字体**：可能导致baseline错位
- ❌ **不要假设所有平台渲染一致**：需实际测试验证

---

## 6. 完整实现示例

```python
import os
import re
import json
from typing import Dict, Set, List, Any
from fontTools import subset
from fontTools.ttLib import TTFont
from pptx import Presentation
from pptx.util import Pt
import zipfile
from lxml import etree


class ChineseTypographyProcessor:
    """
    完整的中文字体处理引擎
    
    功能：
      - 字符集提取和统计
      - 字体子集生成（fonttools）
      - 字符覆盖验证
      - 排版规则应用
      - PPTX字体嵌入
      - 跨平台兼容性测试
    """
    
    def __init__(self, base_font_path='fonts/NotoSansSC-Regular.otf'):
        self.base_font_path = base_font_path
        self.fallback_fonts = ['PingFang SC', 'Microsoft YaHei', 'SimSun']
    
    def extract_used_characters(self, slides_md_path: str) -> tuple[Dict[str, Set], Dict[str, Any]]:
        """
        提取slides.md中使用的所有字符
        
        Returns:
            char_sets: 按类型分类的字符集
            stats: 统计信息
        """
        with open(slides_md_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        char_sets = {
            'chinese': set(re.findall(r'[\u4e00-\u9fff]', content)),
            'latin': set(re.findall(r'[A-Za-z]', content)),
            'digits': set(re.findall(r'[0-9]', content)),
            'punctuation': set(re.findall(r'[，。！？；：""''（）、—…《》]', content)),
            'ascii_punct': set(re.findall(r'[.,!?;:\'"()\-]', content))
        }
        
        # 统计信息
        stats = {
            'total_chars': sum(len(s) for s in char_sets.values()),
            'chinese_count': len(char_sets['chinese']),
            'latin_count': len(char_sets['latin']),
            'estimated_subset_size': len(char_sets['chinese']) * 0.15  # KB
        }
        
        return char_sets, stats
    
    def generate_font_subset(
        self,
        char_sets: Dict[str, Set],
        output_path='fonts/NotoSansSC-Subset.woff2',
        format='woff2'
    ) -> str:
        """
        生成字体子集（使用fonttools）
        
        Args:
            char_sets: 字符集字典
            output_path: 输出路径
            format: 输出格式（woff2, ttf, otf）
        
        Returns:
            subset_path: 生成的子集文件路径
        """
        # 合并所有字符集
        all_chars = set()
        for char_set in char_sets.values():
            all_chars.update(char_set)
        
        # 生成unicode列表
        unicodes = [f"U+{ord(c):04X}" for c in all_chars]
        
        # Subset配置
        options = subset.Options()
        options.flavor = format
        options.layout_features = ['*']  # 保留所有OpenType features
        options.name_IDs = ['*']  # 保留字体名称信息
        options.name_legacy = True
        options.name_languages = ['*']
        
        # 加载并subset字体
        font = subset.load_font(self.base_font_path, options)
        subsetter = subset.Subsetter(options=options)
        subsetter.populate(unicodes=unicodes)
        subsetter.subset(font)
        
        # 保存
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        subset.save_font(font, output_path, options)
        
        return output_path
    
    def validate_font_coverage(
        self,
        font_path: str,
        slides_md_path: str
    ) -> Dict[str, Any]:
        """
        验证字体覆盖率
        
        Returns:
            coverage_report: 包含status、coverage_rate、missing_chars等
        """
        # 加载字体
        font = TTFont(font_path)
        cmap = font.getBestCmap()  # Unicode -> Glyph映射
        supported_chars = set(cmap.keys())
        
        # 提取使用的字符
        used_chars, _ = self.extract_used_characters(slides_md_path)
        all_used = set()
        for char_set in used_chars.values():
            all_used.update(ord(c) for c in char_set)
        
        # 检查覆盖
        missing = all_used - supported_chars
        coverage_rate = (len(all_used) - len(missing)) / len(all_used) if all_used else 1.0
        
        missing_details = []
        if missing:
            # 定位缺失字符位置
            with open(slides_md_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for unicode_val in missing:
                char = chr(unicode_val)
                for i, line in enumerate(lines, 1):
                    if char in line:
                        missing_details.append({
                            'char': char,
                            'unicode': f'U+{unicode_val:04X}',
                            'location': f'line {i}: {line.strip()[:50]}'
                        })
                        break
        
        return {
            'status': 'pass' if not missing else 'fail',
            'coverage_rate': coverage_rate,
            'total_chars': len(all_used),
            'covered_chars': len(all_used) - len(missing),
            'missing_chars': [chr(u) for u in missing],
            'missing_details': missing_details
        }
    
    def apply_typography_rules(
        self,
        text_frame,
        font_name='Noto Sans SC',
        min_font_size=20,
        line_height=1.6
    ):
        """
        应用中文排版规则到text_frame
        
        Args:
            text_frame: python-pptx TextFrame对象
            font_name: 字体名称
            min_font_size: 最小字号（pt）
            line_height: 行高倍数
        """
        for paragraph in text_frame.paragraphs:
            # 设置行高
            paragraph.line_spacing = line_height
            
            for run in paragraph.runs:
                # 应用字体
                run.font.name = font_name
                
                # 确保最小字号
                if run.font.size and run.font.size < Pt(min_font_size):
                    run.font.size = Pt(min_font_size)
                
                # Baseline对齐（针对中英混排）
                if self._is_latin(run.text):
                    run.font.baseline = Pt(-0.1)  # 向下微调
        
        # 修复孤立标点
        self._fix_orphan_punctuation(text_frame)
    
    def _is_latin(self, text: str) -> bool:
        """检测是否为拉丁字符"""
        return all(ord(c) < 0x4E00 or ord(c) > 0x9FFF for c in text if c.strip())
    
    def _fix_orphan_punctuation(self, text_frame):
        """修复孤立标点符号"""
        text = text_frame.text
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            # 检测孤立标点
            if line.strip() in '。！？，、；：':
                # 合并到上一行
                if i > 0:
                    lines[i-1] += line.strip()
                    lines[i] = ''
        
        # 重建文本
        text_frame.text = '\n'.join(l for l in lines if l)
    
    def embed_font_to_pptx(
        self,
        pptx_path: str,
        font_path: str,
        font_name: str
    ) -> Dict[str, Any]:
        """
        嵌入字体到PPTX文件
        
        Args:
            pptx_path: PPTX文件路径
            font_path: 字体文件路径
            font_name: 字体名称
        
        Returns:
            embedding_result: 嵌入结果
        """
        # 1. 添加字体文件到PPTX（ZIP结构）
        with zipfile.ZipFile(pptx_path, 'a') as pptx_zip:
            font_filename = os.path.basename(font_path)
            pptx_zip.write(font_path, f'ppt/fonts/{font_filename}')
        
        # 2. 验证嵌入
        with zipfile.ZipFile(pptx_path, 'r') as pptx_zip:
            font_files = [f for f in pptx_zip.namelist() if f.startswith('ppt/fonts/')]
            embedded = len(font_files) > 0
        
        return {
            'embedded': embedded,
            'font_name': font_name,
            'font_file': os.path.basename(font_path),
            'file_size': os.path.getsize(font_path)
        }
    
    def full_workflow(
        self,
        slides_md_path: str,
        pptx_path: str,
        output_dir='fonts'
    ) -> Dict[str, Any]:
        """
        完整的中文字体处理工作流
        
        Steps:
          1. 提取使用的字符
          2. 生成字体子集
          3. 验证覆盖率
          4. 应用排版规则
          5. 嵌入字体到PPTX
          6. 生成报告
        """
        results = {}
        
        # 1. 提取字符
        char_sets, stats = self.extract_used_characters(slides_md_path)
        results['char_extraction'] = stats
        print(f"✅ Extracted {stats['total_chars']} characters ({stats['chinese_count']} Chinese)")
        
        # 2. 生成子集
        subset_path = os.path.join(output_dir, 'NotoSansSC-Subset.woff2')
        subset_path = self.generate_font_subset(char_sets, subset_path)
        results['subset_path'] = subset_path
        results['subset_size'] = os.path.getsize(subset_path) // 1024  # KB
        print(f"✅ Generated font subset: {results['subset_size']}KB")
        
        # 3. 验证覆盖
        coverage = self.validate_font_coverage(subset_path, slides_md_path)
        results['coverage'] = coverage
        if coverage['status'] == 'fail':
            print(f"⚠️  Warning: {len(coverage['missing_chars'])} characters not covered")
            for detail in coverage['missing_details'][:5]:  # 仅显示前5个
                print(f"   - {detail['char']} ({detail['unicode']}) at {detail['location']}")
        else:
            print(f"✅ Coverage: {coverage['coverage_rate']*100:.2f}%")
        
        # 4. 应用排版规则
        prs = Presentation(pptx_path)
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, 'text_frame'):
                    self.apply_typography_rules(shape.text_frame, font_name='Noto Sans SC')
        prs.save(pptx_path)
        print(f"✅ Applied typography rules to PPTX")
        
        # 5. 嵌入字体
        embed_result = self.embed_font_to_pptx(pptx_path, subset_path, 'Noto Sans SC')
        results['embedding'] = embed_result
        print(f"✅ Font embedded: {embed_result['embedded']}")
        
        # 6. 生成报告
        report_path = os.path.join(output_dir, 'typography_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        results['report_path'] = report_path
        print(f"✅ Report saved: {report_path}")
        
        return results


# 使用示例
if __name__ == '__main__':
    processor = ChineseTypographyProcessor(
        base_font_path='fonts/NotoSansSC-Regular.otf'
    )
    
    result = processor.full_workflow(
        slides_md_path='docs/online-ps-slides.md',
        pptx_path='output.pptx',
        output_dir='fonts'
    )
    
    print(f"\n📊 Final Results:")
    print(f"   - Characters: {result['char_extraction']['total_chars']}")
    print(f"   - Subset Size: {result['subset_size']}KB")
    print(f"   - Coverage: {result['coverage']['coverage_rate']*100:.2f}%")
    print(f"   - Embedded: {result['embedding']['embedded']}")
```

---

## 7. 资源和参考

### 7.1 标准文档

- **W3C《中文排版需求》** (Requirements for Chinese Text Layout) - 中文排版权威标准
- **GB/T 18358-2009** - 《中文出版物夹用英文的编辑规范》
- **GB/T 18358-2009** - 《信息技术 中文Linux系统字体配置规范》
- **Apple Human Interface Guidelines** - Typography (Chinese) - macOS中文排版指南
- **Microsoft Typography Guidelines for CJK** - Windows中文字体规范

### 7.2 工具和库

- **fontTools** - [GitHub](https://github.com/fonttools/fonttools) - 字体子集生成和操作
- **Google Fonts - Noto Sans SC** - [官方页面](https://fonts.google.com/noto/specimen/Noto+Sans+SC) - 开源中文字体
- **思源黑体 Source Han Sans** - [GitHub](https://github.com/adobe-fonts/source-han-sans) - Adobe开源字体
- **OpenType Feature File Specification** - [AFDKO](https://adobe-type-tools.github.io/afdko/OpenTypeFeatureFileSpecification.html)
- **Can I Use - WOFF2** - [浏览器兼容性](https://caniuse.com/woff2)
- **PowerPoint Font Embedding Guide** - Microsoft Docs官方文档

### 7.3 相关 Skills

- `ppt-export.skill` - 调用字体子集嵌入到PPTX最终交付物
- `ppt-theme-manager.skill` - 提供typography system规范（font-family、font-size、line-height）
- `ppt-markdown-parser.skill` - 提供slides.md内容用于字符集提取
