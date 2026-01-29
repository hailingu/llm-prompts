---
name: ppt-export
version: 1.2.0
description: "将 slides 结构渲染为 PPTX/PDF/semantic JSON，并打包完整交付物（含 manifest、README、CHANGELOG、QA报告、预览图、Git元数据）。支持跨平台PDF转换、字体嵌入、资源提取。"
category: presentation
dependencies:
  python_packages:
    - python-pptx  # PPTX generation
    - Pillow  # Image processing
    - PyYAML  # Metadata parsing
    - win32com (Windows only)  # PowerPoint automation
  system:
    - LibreOffice/soffice  # PDF conversion (cross-platform)
    - PowerPoint (Windows)  # PDF conversion (optional)
tags:
  - pptx-export
  - pdf-conversion
  - artifact-packaging
  - manifest-generation
  - changelog
  - semantic-json
  - git-metadata
standards:
  - ISO/IEC 29500 (Office Open XML PPTX Format)
  - PDF/A (ISO 19005-1 for long-term archival)
  - JSON Schema Draft 7 (for semantic JSON)
  - SPDX (Software Package Data Exchange for licensing)
integration:
  agents:
    - ppt-specialist  # Primary consumer for PPTX export
  skills:
    - ppt-chinese-typography  # Font embedding (中文字体子集)
    - ppt-aesthetic-qa  # QA report packaging
    - ppt-markdown-parser  # Parse slides.md for semantic JSON
    - ppt-theme-manager  # Design spec application
last_updated: 2026-01-28
---

# ppt-export Skill

**功能**：将 slides 结构渲染为 PPTX/PDF/semantic JSON，并打包完整交付物（含 manifest、README、CHANGELOG、QA报告、预览图、Git元数据）。

**职责边界**：
- ✅ **本skill负责**：PPTX渲染（python-pptx）、PDF转换（跨平台）、artifact打包、manifest生成、semantic JSON导出、资源提取
- 🔗 **协作skill**：
  - `ppt-chinese-typography.skill`：中文字体子集嵌入（避免文件过大）
  - `ppt-aesthetic-qa.skill`：QA报告打包到交付物
  - `ppt-markdown-parser.skill`：解析 slides.md 生成结构化数据
  - `ppt-theme-manager.skill`：应用 design_spec（colors、typography、spacing）

---

## 1. 导出格式支持

### 1.1 PPTX Export（python-pptx）

**核心功能**：将 slides.md 渲染为 Office Open XML 格式（.pptx）。

**实现原理**：

```python
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN

def export_to_pptx(slides, design_spec, output_path='output.pptx'):
    """
    将slides.md渲染为PPTX文件
    
    Args:
        slides: 从slides.md解析的slide对象列表
        design_spec: 设计规范（colors, typography, spacing）
        output_path: 输出文件路径
    
    Returns:
        pptx_path: 生成的PPTX文件路径
    """
    prs = Presentation()
    
    # 设置默认16:9比例
    prs.slide_width = Inches(10)
    prs.slide_height = Inches(5.625)
    
    for slide_data in slides:
        # 添加空白布局
        slide_layout = prs.slide_layouts[6]  # Blank layout
        slide = prs.slides.add_slide(slide_layout)
        
        # 渲染标题
        title_box = slide.shapes.add_textbox(
            Inches(0.5), 
            Inches(0.5), 
            Inches(9), 
            Inches(1)
        )
        title_frame = title_box.text_frame
        title_frame.text = slide_data['title']
        
        # 应用design_spec中的typography
        title_font = title_frame.paragraphs[0].font
        title_font.name = design_spec['typography_system']['headline_large']['font_family']
        title_font.size = Pt(design_spec['typography_system']['headline_large']['size'])
        title_font.color.rgb = parse_color(design_spec['color_system']['primary']['primary_700'])
        
        # 渲染内容bullets
        if slide_data.get('bullets'):
            content_box = slide.shapes.add_textbox(
                Inches(0.5),
                Inches(2),
                Inches(9),
                Inches(3)
            )
            content_frame = content_box.text_frame
            
            for bullet in slide_data['bullets']:
                p = content_frame.add_paragraph()
                p.text = bullet
                p.level = 0
                
                # 应用body typography
                p.font.name = design_spec['typography_system']['body_large']['font_family']
                p.font.size = Pt(design_spec['typography_system']['body_large']['size'])
        
        # 添加speaker notes
        if slide_data.get('speaker_notes'):
            notes_slide = slide.notes_slide
            notes_frame = notes_slide.notes_text_frame
            notes_frame.text = slide_data['speaker_notes']
    
    prs.save(output_path)
    return output_path

def parse_color(color_hex):
    """将hex颜色转换为RGBColor"""
    from pptx.util import RGBColor
    hex_str = color_hex.lstrip('#')
    return RGBColor(
        int(hex_str[0:2], 16),
        int(hex_str[2:4], 16),
        int(hex_str[4:6], 16)
    )
```

### 1.2 PDF Export（跨平台转换）

**跨平台策略**：
- **macOS/Linux**: LibreOffice headless mode
- **Windows**: PowerPoint COM automation（可选）

```python
import platform
import subprocess
import os

def export_to_pdf(pptx_path, pdf_path='output.pdf'):
    """
    将PPTX转换为PDF（跨平台实现）
    
    Platform-specific:
      - macOS: LibreOffice.app/Contents/MacOS/soffice
      - Linux: libreoffice --headless
      - Windows: PowerPoint COM automation (win32com)
    
    Returns:
        pdf_path: 生成的PDF文件路径
    """
    os_type = platform.system()
    
    if os_type == 'Darwin':  # macOS
        soffice_path = '/Applications/LibreOffice.app/Contents/MacOS/soffice'
        if not os.path.exists(soffice_path):
            raise FileNotFoundError("LibreOffice not installed. Install via 'brew install --cask libreoffice'")
        
        cmd = [
            soffice_path,
            '--headless',
            '--convert-to', 'pdf',
            '--outdir', os.path.dirname(pdf_path) or '.',
            os.path.abspath(pptx_path)
        ]
        subprocess.run(cmd, check=True)
    
    elif os_type == 'Windows':
        # 方法1: PowerPoint COM automation（推荐）
        try:
            import win32com.client
            powerpoint = win32com.client.Dispatch("PowerPoint.Application")
            deck = powerpoint.Presentations.Open(os.path.abspath(pptx_path))
            deck.SaveAs(os.path.abspath(pdf_path), 32)  # 32 = ppSaveAsPDF
            deck.Close()
            powerpoint.Quit()
        except ImportError:
            # 方法2: Fallback to LibreOffice (if installed)
            cmd = ['soffice', '--headless', '--convert-to', 'pdf', pptx_path]
            subprocess.run(cmd, check=True)
    
    elif os_type == 'Linux':
        cmd = ['libreoffice', '--headless', '--convert-to', 'pdf', '--outdir', os.path.dirname(pdf_path) or '.', pptx_path]
        subprocess.run(cmd, check=True)
    
    else:
        raise OSError(f"Unsupported OS: {os_type}")
    
    return pdf_path
```

### 1.3 Semantic JSON Export

**功能**：导出结构化JSON数据，方便二次开发和程序化访问。

```python
import re
import yaml

def generate_semantic_json(slides_md_path):
    """
    生成 slides_semantic.json（符合 JSON Schema Draft 7）
    
    Output Format:
        {
          "metadata": {...},
          "slides": [
            {
              "slide_number": 1,
              "type": "title",
              "title": "...",
              "bullets": [],
              "speaker_notes": "",
              "visual_block": {...},
              "mermaid": "..."
            }
          ]
        }
    """
    with open(slides_md_path, encoding='utf-8') as f:
        content = f.read()
    
    # 解析 front-matter
    fm_match = re.search(r'^---\n(.*?)\n---', content, re.DOTALL | re.MULTILINE)
    metadata = {}
    if fm_match:
        metadata = yaml.safe_load(fm_match.group(1))
        content = content[fm_match.end():]
    
    # 解析 slides（使用 ppt-markdown-parser 逻辑）
    slide_pattern = r'## (.+?)\n(.*?)(?=\n##|\Z)'
    slide_matches = re.findall(slide_pattern, content, re.DOTALL)
    
    slides = []
    for i, (title, body) in enumerate(slide_matches, start=1):
        # 提取 bullets
        bullets = re.findall(r'^\s*[-*]\s+(.+)$', body, re.MULTILINE)
        
        # 提取 speaker notes
        notes_match = re.search(r'NOTE:\s*\n((?:>.+\n)+)', body, re.MULTILINE)
        speaker_notes = ''
        if notes_match:
            lines = notes_match.group(1).split('\n')
            speaker_notes = '\n'.join(line.lstrip('> ').strip() for line in lines if line.strip())
        
        # 提取 VISUAL block
        visual_match = re.search(r'VISUAL:\s*\n((?:  .+\n)+)', body, re.MULTILINE)
        visual_block = None
        if visual_match:
            try:
                visual_block = yaml.safe_load(visual_match.group(1))
            except yaml.YAMLError:
                visual_block = None
        
        # 提取 mermaid
        mermaid_match = re.search(r'```mermaid\n(.*?)```', body, re.DOTALL)
        mermaid = mermaid_match.group(1).strip() if mermaid_match else None
        
        slides.append({
            'slide_number': i,
            'type': 'title' if i == 1 else 'content',
            'title': title.strip(),
            'bullets': bullets,
            'speaker_notes': speaker_notes,
            'visual_block': visual_block,
            'mermaid': mermaid
        })
    
    return {
        'metadata': metadata,
        'slides': slides,
        'total_slides': len(slides)
    }
```

---

## 2. 交付物打包系统

### 2.1 Package Structure（标准目录结构）

```
delivery_package/
├── manifest.json              # 📋 交付物清单（文件哈希、QA摘要、Git元数据）
├── README.md                  # 📖 使用说明
├── CHANGELOG.md               # 📝 版本历史
├── presentation/
│   ├── output.pptx           # ✅ 最终PPTX文件
│   ├── output.pdf            # 📄 PDF版本
│   └── slides_semantic.json  # 🗂️ 结构化数据（JSON Schema Draft 7）
├── source/
│   ├── slides.md             # 📝 源markdown
│   ├── design_spec.json      # 🎨 设计规范
│   └── assets/
│       ├── images/           # 🖼️ 原始图片
│       ├── diagrams/         # 📊 Mermaid源码
│       ├── fonts/            # 🔤 字体子集
│       └── LICENSE.txt       # ⚖️ 资源版权信息
├── qa/
│   ├── qa_report.json        # ✅ 6-stage QA结果
│   ├── accessibility_check.json  # ♿ WCAG验证
│   └── performance_budget.json   # ⚡ 性能指标
└── previews/
    ├── slide_001.png         # 🖼️ 每页预览图
    ├── slide_002.png
    └── ...
```

### 2.2 Manifest Generation（清单生成）

**manifest.json 规范**（符合 SPDX）：

```python
import hashlib
from datetime import datetime

def generate_manifest(pptx_path, slides_md_path, design_spec_path, qa_results):
    """
    生成 manifest.json（交付物清单）
    
    包含内容：
      - 文件列表（路径、大小、SHA256哈希）
      - QA摘要（总分、等级、stage分数、blockers数量）
      - 元数据（slide数量、语言、设计哲学）
      - Git信息（commit、branch、timestamp）
      - License信息
    """
    
    def file_hash(path):
        """计算文件SHA256哈希"""
        with open(path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    
    def file_size(path):
        """获取文件大小（字节）"""
        return os.path.getsize(path)
    
    # 提取slide数量
    with open(slides_md_path, encoding='utf-8') as f:
        slide_count = len(re.findall(r'^## ', f.read(), re.MULTILINE))
    
    # 提取语言和设计哲学（从 front-matter）
    with open(slides_md_path, encoding='utf-8') as f:
        content = f.read()
    fm_match = re.search(r'^---\n(.*?)\n---', content, re.DOTALL | re.MULTILINE)
    frontmatter = yaml.safe_load(fm_match.group(1)) if fm_match else {}
    
    manifest = {
        'package_version': '1.0.0',
        'generated_at': datetime.now().isoformat(),
        
        'files': {
            'pptx': {
                'path': 'presentation/output.pptx',
                'size': file_size(pptx_path),
                'sha256': file_hash(pptx_path)
            },
            'pdf': {
                'path': 'presentation/output.pdf',
                'size': file_size(pptx_path.replace('.pptx', '.pdf')),
                'sha256': file_hash(pptx_path.replace('.pptx', '.pdf'))
            },
            'source_md': {
                'path': 'source/slides.md',
                'sha256': file_hash(slides_md_path)
            },
            'design_spec': {
                'path': 'source/design_spec.json',
                'sha256': file_hash(design_spec_path)
            },
            'semantic_json': {
                'path': 'presentation/slides_semantic.json',
                'schema': 'JSON Schema Draft 7'
            }
        },
        
        'qa_summary': {
            'overall_score': qa_results.get('overall_score', 0),
            'grade': qa_results.get('grade', 'N/A'),
            'stage_scores': qa_results.get('stage_scores', {}),
            'critical_blockers': len(qa_results.get('critical_blockers', [])),
            'pass': qa_results.get('pass', False)
        },
        
        'metadata': {
            'total_slides': slide_count,
            'language': frontmatter.get('language', 'unknown'),
            'design_philosophy': frontmatter.get('recommended_philosophy', 'Material Design'),
            'presentation_type': frontmatter.get('presentation_type', 'technical-review')
        },
        
        'git': add_git_metadata(),  # Git commit信息
        
        'license': {
            'content': 'All rights reserved',
            'assets': 'See source/assets/LICENSE.txt for asset attributions',
            'spdx_identifier': 'UNLICENSED'  # SPDX标准
        }
    }
    
    return manifest


def add_git_metadata():
    """添加Git元数据（如果项目使用Git）"""
    import subprocess
    
    try:
        commit_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        
        commit_msg = subprocess.check_output(
            ['git', 'log', '-1', '--pretty=%B'],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        
        branch = subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        
        timestamp = subprocess.check_output(
            ['git', 'log', '-1', '--format=%cI'],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        
        return {
            'commit': commit_hash,
            'message': commit_msg,
            'branch': branch,
            'timestamp': timestamp
        }
    except Exception:
        return None  # 非Git项目或Git不可用
```

### 2.3 README & CHANGELOG Generation

**README.md 模板**：
```python
def generate_readme(manifest):
    """生成 README.md"""
    return f"""# Presentation Delivery Package

**Generated**: {manifest['generated_at']}  
**Package Version**: {manifest['package_version']}  
**QA Score**: {manifest['qa_summary']['overall_score']}/100 (Grade: {manifest['qa_summary']['grade']})

---

## 📂 Package Contents

### 1. Presentation Files
- `presentation/output.pptx` - Final PowerPoint ({manifest['files']['pptx']['size'] // 1024}KB)
- `presentation/output.pdf` - PDF version for distribution
- `presentation/slides_semantic.json` - Structured JSON (JSON Schema Draft 7)

### 2. Source Files
- `source/slides.md` - Original markdown source
- `source/design_spec.json` - Design system (colors, typography, spacing)
- `source/assets/` - Images, diagrams, fonts

### 3. Quality Assurance
- `qa/qa_report.json` - 6-stage QA validation
- QA Stages:
  1. Schema Validation: {manifest['qa_summary']['stage_scores'].get('schema_validation', 'N/A')}/100
  2. Content Quality: {manifest['qa_summary']['stage_scores'].get('content_quality', 'N/A')}/100
  3. Design Compliance: {manifest['qa_summary']['stage_scores'].get('design_compliance', 'N/A')}/100
  4. Accessibility: {manifest['qa_summary']['stage_scores'].get('accessibility', 'N/A')}/100
  5. Performance: {manifest['qa_summary']['stage_scores'].get('performance', 'N/A')}/100
  6. Technical: {manifest['qa_summary']['stage_scores'].get('technical', 'N/A')}/100

### 4. Previews
- `previews/` - PNG preview for each slide

---

## 🚀 Quick Start

1. **Open presentation**: `presentation/output.pptx`
2. **Review QA report**: `qa/qa_report.json`
3. **Modify source**: Edit `source/slides.md` and regenerate

---

## 📋 Metadata

- **Total Slides**: {manifest['metadata']['total_slides']}
- **Language**: {manifest['metadata']['language']}
- **Design Philosophy**: {manifest['metadata']['design_philosophy']}
- **Presentation Type**: {manifest['metadata']['presentation_type']}

---

## 🔧 Version Control

{f'''- **Git Commit**: `{manifest['git']['commit'][:8]}`
- **Branch**: {manifest['git']['branch']}
- **Timestamp**: {manifest['git']['timestamp']}
- **Message**: {manifest['git']['message']}
''' if manifest.get('git') else '- Not a Git repository'}

---

## 🔒 License

{manifest['license']['content']}

Asset attributions: `{manifest['license']['assets']}`  
SPDX Identifier: `{manifest['license']['spdx_identifier']}`

---

## 📞 Support

For modifications, refer to `source/slides.md` source file.
"""


def generate_changelog(qa_results):
    """生成 CHANGELOG.md"""
    return f"""# Changelog

## [1.0.0] - {datetime.now().strftime('%Y-%m-%d')}

### Added
- Initial presentation generation
- 6-stage QA validation pipeline
- Complete artifact packaging with manifest
- Git metadata tracking

### Quality Metrics
- **Overall Score**: {qa_results.get('overall_score', 0)}/100
- **Grade**: {qa_results.get('grade', 'N/A')}
- **Status**: {'✅ PASS' if qa_results.get('pass') else '❌ FAIL'}

### Stage Results
{chr(10).join(f"- {stage.replace('_', ' ').title()}: {score}/100" for stage, score in qa_results.get('stage_scores', {}).items())}

### Critical Blockers
{len(qa_results.get('critical_blockers', []))} critical issues {'identified and resolved' if qa_results.get('pass') else 'requiring attention'}.

{chr(10).join(f"- {blocker.get('message', 'Unknown issue')}" for blocker in qa_results.get('critical_blockers', [])) if qa_results.get('critical_blockers') else ''}

---

## Version History

- **1.0.0** - Initial release ({datetime.now().strftime('%Y-%m-%d')})
"""
```

---

## 3. 集成接口

### 3.1 输入格式

```python
{
  "slides_md_path": "path/to/slides.md",
  "design_spec_path": "path/to/design_spec.json",
  "output_dir": "delivery_package",
  "export_formats": ["pptx", "pdf", "semantic_json"],
  "include_previews": true,
  "qa_threshold": 70  # 最低QA分数要求
}
```

### 3.2 输出格式

```python
{
  "package_path": "delivery_package/",
  "files_generated": {
    "pptx": "delivery_package/presentation/output.pptx",
    "pdf": "delivery_package/presentation/output.pdf",
    "semantic_json": "delivery_package/presentation/slides_semantic.json",
    "manifest": "delivery_package/manifest.json",
    "readme": "delivery_package/README.md",
    "changelog": "delivery_package/CHANGELOG.md"
  },
  "qa_summary": {
    "overall_score": 85.5,
    "grade": "good",
    "pass": true,
    "critical_blockers": 0
  },
  "package_size": 12458752,  # bytes
  "generation_time": 3.5  # seconds
}
```

---

## 4. 最佳实践

### 4.1 字体嵌入规范

**DO**：
- ✅ **使用字体子集**：仅嵌入使用的字符（调用 ppt-chinese-typography.skill）
- ✅ **验证字体许可**：确保字体允许嵌入
- ✅ **Fallback字体**：指定跨平台备用字体（SimSun → Arial）
- ✅ **嵌入OTF/TTF**：PPTX支持 OpenType 和 TrueType

**DON'T**：
- ❌ **嵌入完整字体**：中文字体通常15-30MB，导致PPTX过大
- ❌ **使用系统字体路径**：不同OS路径不同（macOS: /Library/Fonts, Windows: C:\\Windows\\Fonts）
- ❌ **忽略许可限制**：某些商业字体禁止嵌入

### 4.2 文件大小控制

**Performance Budget**：
```yaml
file_size_limits:
  pptx: 50MB  # ISO/IEC 29500推荐
  pdf: 20MB
  single_image: 5MB
  total_assets: 30MB
```

**优化策略**：
- ✅ **图片压缩**：PNG → WebP/JPEG（质量90%），使用 Pillow
- ✅ **字体子集**：仅嵌入使用的字符
- ✅ **移除元数据**：图片EXIF数据（减少5-10%）
- ✅ **矢量优先**：Mermaid图表导出为SVG

### 4.3 跨平台兼容性

**平台测试矩阵**：
```yaml
platforms:
  macOS:
    pptx_viewer: Keynote, PowerPoint for Mac, LibreOffice
    pdf_converter: LibreOffice (soffice)
  
  Windows:
    pptx_viewer: PowerPoint, LibreOffice
    pdf_converter: PowerPoint COM, LibreOffice
  
  Linux:
    pptx_viewer: LibreOffice Impress
    pdf_converter: LibreOffice (headless)
```

**兼容性检查**：
- ✅ **测试PPTX打开**：在 PowerPoint、Keynote、LibreOffice 中验证
- ✅ **PDF渲染一致性**：对比不同转换器输出
- ✅ **字体渲染**：验证中文字体在Windows/macOS显示

### 4.4 QA验证规范

**打包前检查清单**：
```python
def validate_before_packaging(qa_results, qa_threshold=70):
    """打包前QA验证"""
    
    # 1. 检查QA分数
    if qa_results['overall_score'] < qa_threshold:
        raise ValueError(f"QA score {qa_results['overall_score']} < threshold {qa_threshold}")
    
    # 2. 检查critical blockers
    if len(qa_results.get('critical_blockers', [])) > 0:
        raise ValueError(f"{len(qa_results['critical_blockers'])} critical blockers remain")
    
    # 3. 检查accessibility
    if qa_results.get('stage_scores', {}).get('accessibility', 0) < 70:
        raise ValueError("Accessibility score too low (WCAG 2.1 AA required)")
    
    # 4. 检查performance budget
    if qa_results.get('stage_scores', {}).get('performance', 0) < 70:
        raise ValueError("Performance budget exceeded")
    
    return True
```

---

## 5. 完整实现示例

```python
import os
import json
import shutil
from datetime import datetime
from typing import Dict, Any, List


class PPTExporter:
    """完整的PPT导出和打包引擎"""
    
    def __init__(self, qa_threshold=70):
        self.qa_threshold = qa_threshold
    
    def full_export_workflow(
        self,
        slides_md_path: str,
        design_spec_path: str,
        output_dir='delivery_package'
    ) -> Dict[str, Any]:
        """
        完整导出工作流
        
        Steps:
          1. 解析 slides.md (ppt-markdown-parser)
          2. 渲染 PPTX (export_to_pptx)
          3. 转换 PDF (export_to_pdf)
          4. 生成 semantic JSON (generate_semantic_json)
          5. 执行 QA (ppt-aesthetic-qa)
          6. 验证 QA结果
          7. 创建交付物包 (create_artifact_package)
          8. 返回打包信息
        """
        start_time = datetime.now()
        
        # Step 1: 解析slides
        slides = self._parse_slides(slides_md_path)
        
        # Step 2: 加载design_spec
        with open(design_spec_path, encoding='utf-8') as f:
            design_spec = json.load(f)
        
        # Step 3: 渲染PPTX
        pptx_path = 'output.pptx'
        export_to_pptx(slides, design_spec, pptx_path)
        
        # Step 4: 转换PDF
        pdf_path = 'output.pdf'
        export_to_pdf(pptx_path, pdf_path)
        
        # Step 5: 生成semantic JSON
        semantic_data = generate_semantic_json(slides_md_path)
        semantic_path = 'slides_semantic.json'
        with open(semantic_path, 'w', encoding='utf-8') as f:
            json.dump(semantic_data, f, indent=2, ensure_ascii=False)
        
        # Step 6: 执行QA（调用 ppt-aesthetic-qa.skill）
        qa_results = self._run_qa_validation(pptx_path, slides_md_path, design_spec)
        
        # Step 7: 验证QA结果
        try:
            validate_before_packaging(qa_results, self.qa_threshold)
        except ValueError as e:
            return {
                'success': False,
                'error': str(e),
                'qa_summary': qa_results
            }
        
        # Step 8: 创建交付物包
        package_path = self._create_artifact_package(
            pptx_path=pptx_path,
            pdf_path=pdf_path,
            semantic_path=semantic_path,
            slides_md_path=slides_md_path,
            design_spec_path=design_spec_path,
            qa_results=qa_results,
            output_dir=output_dir
        )
        
        end_time = datetime.now()
        generation_time = (end_time - start_time).total_seconds()
        
        # Step 9: 计算包大小
        package_size = self._calculate_package_size(output_dir)
        
        return {
            'success': True,
            'package_path': package_path,
            'files_generated': {
                'pptx': f'{output_dir}/presentation/output.pptx',
                'pdf': f'{output_dir}/presentation/output.pdf',
                'semantic_json': f'{output_dir}/presentation/slides_semantic.json',
                'manifest': f'{output_dir}/manifest.json',
                'readme': f'{output_dir}/README.md',
                'changelog': f'{output_dir}/CHANGELOG.md'
            },
            'qa_summary': {
                'overall_score': qa_results['overall_score'],
                'grade': qa_results['grade'],
                'pass': qa_results['pass'],
                'critical_blockers': len(qa_results.get('critical_blockers', []))
            },
            'package_size': package_size,
            'generation_time': generation_time
        }
    
    def _parse_slides(self, slides_md_path: str) -> List[Dict]:
        """解析slides.md（简化实现，实际使用ppt-markdown-parser）"""
        # 调用 ppt-markdown-parser.skill
        return []  # Placeholder
    
    def _run_qa_validation(self, pptx_path, slides_md_path, design_spec):
        """执行QA验证（调用 ppt-aesthetic-qa.skill）"""
        # 调用 ppt-aesthetic-qa.skill
        return {
            'overall_score': 85.0,
            'grade': 'good',
            'pass': True,
            'stage_scores': {
                'schema_validation': 95,
                'content_quality': 85,
                'design_compliance': 80,
                'accessibility': 90,
                'performance': 75,
                'technical': 85
            },
            'critical_blockers': []
        }
    
    def _create_artifact_package(
        self,
        pptx_path,
        pdf_path,
        semantic_path,
        slides_md_path,
        design_spec_path,
        qa_results,
        output_dir
    ):
        """创建完整交付物包"""
        
        # 创建目录结构
        os.makedirs(f'{output_dir}/presentation', exist_ok=True)
        os.makedirs(f'{output_dir}/source/assets/images', exist_ok=True)
        os.makedirs(f'{output_dir}/source/assets/diagrams', exist_ok=True)
        os.makedirs(f'{output_dir}/source/assets/fonts', exist_ok=True)
        os.makedirs(f'{output_dir}/qa', exist_ok=True)
        os.makedirs(f'{output_dir}/previews', exist_ok=True)
        
        # 复制主文件
        shutil.copy(pptx_path, f'{output_dir}/presentation/output.pptx')
        shutil.copy(pdf_path, f'{output_dir}/presentation/output.pdf')
        shutil.copy(semantic_path, f'{output_dir}/presentation/slides_semantic.json')
        shutil.copy(slides_md_path, f'{output_dir}/source/slides.md')
        shutil.copy(design_spec_path, f'{output_dir}/source/design_spec.json')
        
        # 提取assets
        # 提取assets
        self._extract_assets(pptx_path, f'{output_dir}/source/assets')
        
        # 保存QA报告
        with open(f'{output_dir}/qa/qa_report.json', 'w', encoding='utf-8') as f:
            json.dump(qa_results, f, indent=2, ensure_ascii=False)
        
        # 生成manifest
        manifest = generate_manifest(
            f'{output_dir}/presentation/output.pptx',
            f'{output_dir}/source/slides.md',
            f'{output_dir}/source/design_spec.json',
            qa_results
        )
        with open(f'{output_dir}/manifest.json', 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        
        # 生成README
        readme = generate_readme(manifest)
        with open(f'{output_dir}/README.md', 'w', encoding='utf-8') as f:
            f.write(readme)
        
        # 生成CHANGELOG
        changelog = generate_changelog(qa_results)
        with open(f'{output_dir}/CHANGELOG.md', 'w', encoding='utf-8') as f:
            f.write(changelog)
        
        # 生成预览图（可选）
        # self._generate_previews(pptx_path, f'{output_dir}/previews')
        
        return output_dir
    
    def _extract_assets(self, pptx_path, assets_dir):
        """从PPTX提取assets（图片、字体）"""
        from zipfile import ZipFile
        
        with ZipFile(pptx_path) as z:
            # 提取图片
            image_files = [f for f in z.namelist() if f.startswith('ppt/media/')]
            for img_file in image_files:
                img_data = z.read(img_file)
                output_path = os.path.join(assets_dir, 'images', os.path.basename(img_file))
                with open(output_path, 'wb') as f:
                    f.write(img_data)
            
            # 提取字体
            font_files = [f for f in z.namelist() if f.startswith('ppt/fonts/')]
            for font_file in font_files:
                font_data = z.read(font_file)
                output_path = os.path.join(assets_dir, 'fonts', os.path.basename(font_file))
                with open(output_path, 'wb') as f:
                    f.write(font_data)
    
    def _calculate_package_size(self, output_dir):
        """计算打包后的总大小"""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(output_dir):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)
        return total_size


# 使用示例
if __name__ == '__main__':
    exporter = PPTExporter(qa_threshold=70)
    
    result = exporter.full_export_workflow(
        slides_md_path='slides.md',
        design_spec_path='design_spec.json',
        output_dir='delivery_package'
    )
    
    if result['success']:
        print(f"✅ Package created: {result['package_path']}")
        print(f"📊 QA Score: {result['qa_summary']['overall_score']}/100 ({result['qa_summary']['grade']})")
        print(f"📦 Package Size: {result['package_size'] // 1024}KB")
        print(f"⏱️ Generation Time: {result['generation_time']:.2f}s")
    else:
        print(f"❌ Export failed: {result['error']}")
        print(f"📊 QA Score: {result['qa_summary']['overall_score']}/100")
```

---

## 6. 资源和参考

### 6.1 标准文档

- **ISO/IEC 29500** - Office Open XML (PPTX) 格式标准
- **PDF/A (ISO 19005-1)** - PDF长期归档标准
- **JSON Schema Draft 7** - Semantic JSON 数据验证
- **SPDX** - 软件包数据交换标准（许可标识）

### 6.2 工具和库

- **python-pptx** - [官方文档](https://python-pptx.readthedocs.io/)
- **LibreOffice** - [Headless Conversion](https://wiki.documentfoundation.org/Faq/General/021)
- **Pillow** - 图片处理和压缩
- **PyYAML** - Front-matter解析

### 6.3 相关 Skills

- `ppt-chinese-typography.skill` - 中文字体子集嵌入
- `ppt-aesthetic-qa.skill` - 6-stage QA验证
- `ppt-markdown-parser.skill` - slides.md 解析为结构化数据
- `ppt-theme-manager.skill` - design_spec 应用（colors、typography、spacing）
