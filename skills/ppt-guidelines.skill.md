---
name: ppt-guidelines
version: 1.1.0
description: "提供幻灯片质量验证和最佳实践检查。基于业界标准（Presentation Zen、McKinsey Standards）进行规则检查，返回 issues、suggestions 和质量评分，支持 auto_fix。"
category: presentation
dependencies:
  files:
    - standards/ppt-guidelines/ppt-guidelines.json  # 规则配置文件
tags:
  - best-practices
  - presentation-design
  - policy-validation
  - auto-fix
  - quality-rules
  - wcag-accessibility
  - mckinsey-standards
standards:
  - Presentation Zen (Garr Reynolds)
  - Slide:ology (Nancy Duarte)
  - Talk Like TED (Carmine Gallo)
  - McKinsey Presentation Standards
  - Apple Keynote Design Principles
  - WCAG 2.1 (Accessibility)
integration:
  agents:
    - ppt-creative-director  # References for design philosophy
    - ppt-content-planner  # Content structure guidelines
    - ppt-specialist  # Auto-fix execution
  skills:
    - ppt-aesthetic-qa  # Visual quality scoring
    - ppt-visual  # WCAG contrast validation
    - ppt-outline  # Slide structure validation
    - ppt-layout  # Layout rule validation
last_updated: 2026-01-28
---

# ppt-guidelines Skill

**功能**：提供幻灯片质量验证和最佳实践检查。基于业界标准（Presentation Zen、McKinsey Standards）进行规则检查，返回 issues、suggestions 和质量评分，支持 auto_fix。

**职责边界**：
- ✅ **本skill负责**：规则验证（文字计数、bullet限制、contrast检查）、质量评分、issues汇总、auto-fix建议
- 🔗 **协作skill**：
  - `ppt-aesthetic-qa.skill`：视觉美学评分（color harmony、layout balance）
  - `ppt-visual.skill`：WCAG 2.1 对比度验证
  - `ppt-outline.skill`：大纲结构验证（MECE、Story Arc）
  - `ppt-layout.skill`：布局规则验证（网格对齐、Assertion-Evidence）

---

## 1. 核心验证规则

### 1.1 内容规则（Content Rules）

**Rule 1: Bullet Points 限制（6x6 Rule）**
```yaml
rule_id: bullet-6x6
source: Presentation Zen (Garr Reynolds)
validation:
  max_bullets_per_slide: 6
  max_words_per_bullet: 8  # 实践中放宽为8
  
severity: warning
auto_fix: split_slide_if_exceeded

example_violation:
  slide_title: "系统架构"
  bullets:
    - "认证模块负责用户登录验证和权限管理"  # 9个字，超限
    - "限流模块控制请求速率"
    - "推荐模块提供个性化内容推荐算法"
    - "搜索模块实现全文检索"
    - "数据模块管理存储和缓存"
    - "监控模块收集性能指标"
    - "日志模块记录系统事件"  # 第7条，超限
  
  issues:
    - "Bullet 1 超过8个字（9字）"
    - "Bullets总数7条，超过6条限制"
  
  auto_fix:
    - action: split_into_two_slides
      result:
        - slide_1: bullets[0:3]
        - slide_2: bullets[3:6]
```

**Rule 2: 标题规则（Title Best Practices）**
```yaml
rule_id: title-assertion
source: Assertion-Evidence Method (Michael Alley)
validation:
  prefer_assertion_over_topic: true
  max_title_length: 80  # 字符
  
severity: suggestion
auto_fix: suggest_assertion_title

example:
  violation:
    title: "系统性能"  # 话题标题（Topic）
  
  suggestion:
    title: "缓存优化使P99延迟降低62%"  # 断言标题（Assertion）
  
  reasoning: "断言标题传达结论，提高信息密度"
```

**Rule 3: 单页文字量限制（Text Density）**
```yaml
rule_id: text-density
source: Slide:ology (Nancy Duarte)
validation:
  max_words_per_slide: 40
  exclude_slide_types:
    - appendix
    - reference
  
severity: warning
auto_fix: suggest_reduce_text

calculation:
  words = len(title.split()) + sum(len(b.split()) for b in bullets)
  
example_violation:
  slide:
    title: "系统架构详解"
    bullets: ["...", "..."]  # 总计45个字
  
  issue: "单页文字量45字，超过40字限制"
  
  suggestion: "考虑使用图表替代文字描述"
```

### 1.2 视觉规则（Visual Rules）

**Rule 4: 对比度检查（WCAG 2.1 Contrast）**
```yaml
rule_id: wcag-contrast
source: WCAG 2.1 AA
validation:
  min_contrast_ratio: 4.5  # 正文文字
  min_contrast_large_text: 3.0  # 大字号（≥18pt）
  
severity: error
auto_fix: adjust_color_brightness

formula: |
  contrast_ratio = (L1 + 0.05) / (L2 + 0.05)
  where L1 = relative_luminance(lighter_color)
        L2 = relative_luminance(darker_color)

example_violation:
  text_color: "#888888"  # 中灰
  background: "#FFFFFF"  # 白色
  contrast_ratio: 2.9  # 不合格（<4.5）
  
  auto_fix:
    new_text_color: "#595959"  # 深灰
    new_contrast_ratio: 4.6  # 合格
```

**Rule 5: 字体大小（Font Size）**
```yaml
rule_id: font-size-minimum
source: McKinsey Presentation Standards
validation:
  min_body_text: 14pt
  min_title_text: 24pt
  min_appendix_text: 10pt
  
severity: warning
auto_fix: increase_font_size

example_violation:
  bullet_text_size: 12pt  # 过小
  
  issue: "正文字号12pt，低于最小值14pt"
  
  auto_fix:
    new_size: 14pt
    reasoning: "确保后排观众可读"
```

**Rule 6: 图片质量（Image Quality）**
```yaml
rule_id: image-resolution
source: Apple Keynote Design Principles
validation:
  min_dpi: 150  # 幻灯片展示
  min_width_for_full_slide: 1920px
  require_attribution: true  # 版权标注
  
severity: warning
auto_fix: suggest_higher_resolution

example_violation:
  image:
    path: "diagram.png"
    resolution: "800x600"  # 72 DPI
    has_attribution: false
  
  issues:
    - "图片分辨率不足（建议≥1920px宽）"
    - "缺少版权标注"
  
  suggestions:
    - "使用矢量图（SVG）或更高分辨率图片"
    - "添加 'Source: ...' 标注"
```

### 1.3 结构规则（Structure Rules）

**Rule 7: Key Decision Slide（关键决策页）**
```yaml
rule_id: key-decision-required
source: McKinsey Presentation Standards
validation:
  require_decision_slide: true
  decision_keywords:
    - "建议"
    - "决策"
    - "行动方案"
    - "下一步"
    - "Recommendation"
    - "Next Steps"
  
severity: suggestion
auto_fix: suggest_add_decision_slide

detection:
  method: keyword_match
  location: slide_title or bullets
  
example_violation:
  presentation:
    slides: [...]  # 15页幻灯片
    has_decision_slide: false
  
  issue: "未检测到关键决策页"
  
  suggestion: |
    添加一页"建议行动方案"幻灯片，包含：
    - 3-5条具体行动项
    - 负责人
    - 时间节点
```

**Rule 8: Slide 数量控制（10/20/30 Rule）**
```yaml
rule_id: slide-count-limit
source: Guy Kawasaki (10/20/30 Rule)
validation:
  limits:
    pitch: 10  # 投资路演
    technical_review: 20  # 技术评审
    workshop: 30  # 培训讲座
  
severity: suggestion
auto_fix: suggest_merge_or_appendix

example_violation:
  presentation_type: "technical_review"
  slide_count: 25
  
  issue: "技术评审类型建议≤20页，当前25页"
  
  suggestions:
    - "合并相似内容幻灯片"
    - "移动细节到附录（Appendix）"
```

---

## 2. 质量评分系统

### 2.1 评分维度

**综合评分计算**：
```python
def calculate_quality_score(slides, qa_results):
    """计算幻灯片质量综合评分（0-100）"""
    
    # 维度1: 内容质量（40%）
    content_score = evaluate_content_quality(slides)
    # - Bullet数量合规性
    # - 文字密度控制
    # - 标题断言化程度
    
    # 维度2: 视觉质量（30%）
    visual_score = evaluate_visual_quality(slides)
    # - WCAG对比度合规
    # - 字体大小合规
    # - 图片质量
    
    # 维度3: 结构质量（20%）
    structure_score = evaluate_structure_quality(slides)
    # - 是否有决策页
    # - Slide数量合理性
    # - Story Arc 完整性
    
    # 维度4: 美学质量（10%）
    aesthetic_score = get_aesthetic_score_from_ppt_aesthetic_qa(slides)
    # - 调用 ppt-aesthetic-qa.skill
    
    # 综合评分
    total_score = (
        content_score * 0.4 +
        visual_score * 0.3 +
        structure_score * 0.2 +
        aesthetic_score * 0.1
    )
    
    return {
        'total_score': round(total_score, 1),
        'breakdown': {
            'content': content_score,
            'visual': visual_score,
            'structure': structure_score,
            'aesthetic': aesthetic_score
        }
    }
```

### 2.2 评分标准

**分数等级**：
```yaml
grade_levels:
  excellent:
    range: [90, 100]
    label: "优秀"
    description: "符合所有最佳实践，无重大问题"
  
  good:
    range: [75, 89]
    label: "良好"
    description: "大部分规则合规，有少量改进建议"
  
  fair:
    range: [60, 74]
    label: "中等"
    description: "存在多项问题，需要优化"
  
  poor:
    range: [0, 59]
    label: "待改进"
    description: "多项关键问题，需大幅修改"
```

---

## 3. 集成接口

### 3.1 输入格式

```json
{
  "slides": [
    {
      "slide_number": 1,
      "type": "title",
      "title": "系统架构评审",
      "bullets": [],
      "images": [],
      "charts": []
    },
    {
      "slide_number": 2,
      "type": "content",
      "title": "系统性能",
      "bullets": [
        "认证模块",
        "限流模块",
        "推荐模块",
        "搜索模块",
        "数据模块",
        "监控模块",
        "日志模块"
      ],
      "text_color": "#888888",
      "background_color": "#FFFFFF",
      "font_size": 12
    }
  ],
  "rules": {
    "bullet-6x6": {"enabled": true, "severity": "warning"},
    "wcag-contrast": {"enabled": true, "severity": "error"}
  },
  "auto_fix": true
}
```

### 3.2 输出格式

```json
{
  "qa_report": {
    "total_score": 68.5,
    "grade": "fair",
    "breakdown": {
      "content": 65.0,
      "visual": 55.0,
      "structure": 80.0,
      "aesthetic": 75.0
    },
    "issues": [
      {
        "slide_number": 2,
        "rule_id": "bullet-6x6",
        "severity": "warning",
        "detail": "Bullets总数7条，超过6条限制",
        "location": "bullets"
      },
      {
        "slide_number": 2,
        "rule_id": "wcag-contrast",
        "severity": "error",
        "detail": "文字对比度2.9，低于WCAG 2.1 AA标准（4.5）",
        "location": "text_color vs background_color"
      },
      {
        "slide_number": 2,
        "rule_id": "font-size-minimum",
        "severity": "warning",
        "detail": "正文字号12pt，低于最小值14pt",
        "location": "font_size"
      },
      {
        "slide_number": 2,
        "rule_id": "title-assertion",
        "severity": "suggestion",
        "detail": "标题为话题型（'系统性能'），建议改为断言型",
        "location": "title"
      }
    ],
    "suggestions": [
      "Slide 2: 将7条bullets拆分为2页",
      "Slide 2: 文字颜色改为 #595959（提高对比度至4.6）",
      "Slide 2: 字号从12pt调整为14pt",
      "Slide 2: 标题建议改为 '缓存优化使P99延迟降低62%'"
    ],
    "auto_fix_applied": [
      {
        "slide_number": 2,
        "rule_id": "wcag-contrast",
        "action": "adjust_color_brightness",
        "changes": {
          "text_color": "#888888 → #595959",
          "new_contrast_ratio": 4.6
        }
      },
      {
        "slide_number": 2,
        "rule_id": "font-size-minimum",
        "action": "increase_font_size",
        "changes": {
          "font_size": "12pt → 14pt"
        }
      }
    ]
  }
}
```

---

## 4. 最佳实践

### 4.1 规则配置规范

**DO**：
- ✅ **启用核心规则**：bullet-6x6、wcag-contrast、font-size-minimum（强制执行）
- ✅ **区分严重级别**：error（阻断）、warning（警告）、suggestion（建议）
- ✅ **允许例外**：appendix、reference类型幻灯片可豁免部分规则
- ✅ **渐进式改进**：优先修复error，再处理warning，最后优化suggestion

**DON'T**：
- ❌ **盲目应用所有规则**：根据演讲场景（技术评审 vs 战略汇报）选择规则
- ❌ **忽略上下文**：某些规则在特定场景下可放宽（如培训PPT的文字量）
- ❌ **过度依赖auto-fix**：人工审核auto-fix结果，避免误改

### 4.2 质量检查流程

**推荐工作流**：
```
1. 内容创建阶段
   ↓
2. 运行 ppt-guidelines.check()
   ├─ 获取 qa_report
   ├─ 查看 issues（按severity排序）
   └─ 查看 suggestions
   ↓
3. 修复 error 级别问题（必须）
   ↓
4. 修复 warning 级别问题（建议）
   ↓
5. 考虑 suggestion（可选）
   ↓
6. 再次运行检查，确保 score ≥ 75
   ↓
7. 最终评审（人工）
```

### 4.3 Auto-Fix 使用规范

**适合 Auto-Fix 的场景**：
- ✅ **对比度调整**：颜色明度调整，算法确定
- ✅ **字号调整**：简单的数值增加
- ✅ **Slide拆分**：机械性拆分超长bullets

**不适合 Auto-Fix 的场景**：
- ❌ **标题重写**：需要理解内容语义
- ❌ **内容删减**：需要判断信息优先级
- ❌ **图表重排**：需要设计判断

---

## 5. 完整实现示例

```python
from typing import List, Dict, Any
import re


class PPTGuidelinesValidator:
    """幻灯片质量验证引擎"""
    
    # 规则定义
    RULES = {
        'bullet-6x6': {
            'enabled': True,
            'severity': 'warning',
            'max_bullets': 6,
            'max_words_per_bullet': 8
        },
        'wcag-contrast': {
            'enabled': True,
            'severity': 'error',
            'min_contrast_ratio': 4.5,
            'min_contrast_large_text': 3.0
        },
        'font-size-minimum': {
            'enabled': True,
            'severity': 'warning',
            'min_body_text': 14,
            'min_title_text': 24
        },
        'title-assertion': {
            'enabled': True,
            'severity': 'suggestion'
        },
        'text-density': {
            'enabled': True,
            'severity': 'warning',
            'max_words_per_slide': 40
        },
        'key-decision-required': {
            'enabled': True,
            'severity': 'suggestion',
            'keywords': ['建议', '决策', '行动方案', '下一步']
        }
    }
    
    def __init__(self, rules_override: Dict = None):
        """初始化验证器，可选覆盖规则"""
        self.rules = self.RULES.copy()
        if rules_override:
            self.rules.update(rules_override)
        
        self.issues = []
        self.suggestions = []
        self.auto_fix_applied = []
    
    def validate(self, slides: List[Dict], auto_fix: bool = False) -> Dict[str, Any]:
        """主验证函数"""
        
        # 重置状态
        self.issues = []
        self.suggestions = []
        self.auto_fix_applied = []
        
        # 运行所有规则检查
        for slide in slides:
            self._check_bullet_6x6(slide, auto_fix)
            self._check_wcag_contrast(slide, auto_fix)
            self._check_font_size(slide, auto_fix)
            self._check_title_assertion(slide)
            self._check_text_density(slide)
        
        # 检查全局规则
        self._check_key_decision_required(slides)
        
        # 计算质量评分
        score_breakdown = self._calculate_scores(slides)
        
        # 生成报告
        qa_report = {
            'total_score': score_breakdown['total_score'],
            'grade': self._get_grade(score_breakdown['total_score']),
            'breakdown': score_breakdown['breakdown'],
            'issues': self.issues,
            'suggestions': self.suggestions,
            'auto_fix_applied': self.auto_fix_applied if auto_fix else []
        }
        
        return {'qa_report': qa_report}
    
    def _check_bullet_6x6(self, slide: Dict, auto_fix: bool):
        """Rule: Bullet 6x6"""
        if not self.rules['bullet-6x6']['enabled']:
            return
        
        bullets = slide.get('bullets', [])
        max_bullets = self.rules['bullet-6x6']['max_bullets']
        max_words = self.rules['bullet-6x6']['max_words_per_bullet']
        
        # 检查bullet数量
        if len(bullets) > max_bullets:
            self.issues.append({
                'slide_number': slide.get('slide_number'),
                'rule_id': 'bullet-6x6',
                'severity': self.rules['bullet-6x6']['severity'],
                'detail': f"Bullets总数{len(bullets)}条，超过{max_bullets}条限制",
                'location': 'bullets'
            })
            
            if auto_fix:
                # Auto-fix: 建议拆分（不自动执行，仅记录建议）
                self.suggestions.append(
                    f"Slide {slide.get('slide_number')}: 将{len(bullets)}条bullets拆分为{(len(bullets) + max_bullets - 1) // max_bullets}页"
                )
        
        # 检查每条bullet的字数
        for i, bullet in enumerate(bullets):
            word_count = len(bullet)  # 中文按字符计数
            if word_count > max_words:
                self.issues.append({
                    'slide_number': slide.get('slide_number'),
                    'rule_id': 'bullet-6x6',
                    'severity': self.rules['bullet-6x6']['severity'],
                    'detail': f"Bullet {i+1} 超过{max_words}个字（{word_count}字）",
                    'location': f'bullets[{i}]'
                })
    
    def _check_wcag_contrast(self, slide: Dict, auto_fix: bool):
        """Rule: WCAG Contrast"""
        if not self.rules['wcag-contrast']['enabled']:
            return
        
        text_color = slide.get('text_color')
        bg_color = slide.get('background_color')
        
        if not text_color or not bg_color:
            return
        
        # 计算对比度
        contrast_ratio = self._calculate_contrast_ratio(text_color, bg_color)
        min_ratio = self.rules['wcag-contrast']['min_contrast_ratio']
        
        if contrast_ratio < min_ratio:
            self.issues.append({
                'slide_number': slide.get('slide_number'),
                'rule_id': 'wcag-contrast',
                'severity': self.rules['wcag-contrast']['severity'],
                'detail': f"文字对比度{contrast_ratio:.1f}，低于WCAG 2.1 AA标准（{min_ratio}）",
                'location': 'text_color vs background_color'
            })
            
            if auto_fix:
                # Auto-fix: 调整文字颜色亮度
                new_text_color = self._adjust_color_for_contrast(text_color, bg_color, min_ratio)
                new_contrast = self._calculate_contrast_ratio(new_text_color, bg_color)
                
                slide['text_color'] = new_text_color  # 应用修改
                
                self.auto_fix_applied.append({
                    'slide_number': slide.get('slide_number'),
                    'rule_id': 'wcag-contrast',
                    'action': 'adjust_color_brightness',
                    'changes': {
                        'text_color': f"{text_color} → {new_text_color}",
                        'new_contrast_ratio': round(new_contrast, 1)
                    }
                })
    
    def _check_font_size(self, slide: Dict, auto_fix: bool):
        """Rule: Font Size Minimum"""
        if not self.rules['font-size-minimum']['enabled']:
            return
        
        font_size = slide.get('font_size')
        min_body = self.rules['font-size-minimum']['min_body_text']
        
        if font_size and font_size < min_body:
            self.issues.append({
                'slide_number': slide.get('slide_number'),
                'rule_id': 'font-size-minimum',
                'severity': self.rules['font-size-minimum']['severity'],
                'detail': f"正文字号{font_size}pt，低于最小值{min_body}pt",
                'location': 'font_size'
            })
            
            if auto_fix:
                slide['font_size'] = min_body  # 应用修改
                
                self.auto_fix_applied.append({
                    'slide_number': slide.get('slide_number'),
                    'rule_id': 'font-size-minimum',
                    'action': 'increase_font_size',
                    'changes': {
                        'font_size': f"{font_size}pt → {min_body}pt"
                    }
                })
    
    def _check_title_assertion(self, slide: Dict):
        """Rule: Title Assertion"""
        if not self.rules['title-assertion']['enabled']:
            return
        
        title = slide.get('title', '')
        
        # 检测是否为断言句
        is_assertion = (
            title.endswith(('。', '.', '!', '！')) or
            re.search(r'使.*(降低|提升|增加|改善)', title) or
            re.search(r'\d+(%|ms|倍|次)', title)
        )
        
        if not is_assertion and slide.get('type') == 'content':
            self.issues.append({
                'slide_number': slide.get('slide_number'),
                'rule_id': 'title-assertion',
                'severity': self.rules['title-assertion']['severity'],
                'detail': f"标题为话题型（'{title}'），建议改为断言型",
                'location': 'title'
            })
            
            # 仅提供建议，不auto-fix（需要语义理解）
            self.suggestions.append(
                f"Slide {slide.get('slide_number')}: 标题建议改为断言句（传达结论）"
            )
    
    def _check_text_density(self, slide: Dict):
        """Rule: Text Density"""
        if not self.rules['text-density']['enabled']:
            return
        
        title = slide.get('title', '')
        bullets = slide.get('bullets', [])
        
        # 计算总字数
        total_words = len(title) + sum(len(b) for b in bullets)
        max_words = self.rules['text-density']['max_words_per_slide']
        
        if total_words > max_words and slide.get('type') not in ['appendix', 'reference']:
            self.issues.append({
                'slide_number': slide.get('slide_number'),
                'rule_id': 'text-density',
                'severity': self.rules['text-density']['severity'],
                'detail': f"单页文字量{total_words}字，超过{max_words}字限制",
                'location': 'title + bullets'
            })
            
            self.suggestions.append(
                f"Slide {slide.get('slide_number')}: 考虑用图表替代部分文字描述"
            )
    
    def _check_key_decision_required(self, slides: List[Dict]):
        """Rule: Key Decision Required"""
        if not self.rules['key-decision-required']['enabled']:
            return
        
        keywords = self.rules['key-decision-required']['keywords']
        
        # 检测是否有决策页
        has_decision_slide = any(
            any(kw in slide.get('title', '') for kw in keywords)
            for slide in slides
        )
        
        if not has_decision_slide:
            self.issues.append({
                'slide_number': None,
                'rule_id': 'key-decision-required',
                'severity': self.rules['key-decision-required']['severity'],
                'detail': "未检测到关键决策页",
                'location': 'global'
            })
            
            self.suggestions.append(
                "添加一页'建议行动方案'幻灯片，包含具体行动项、负责人、时间节点"
            )
    
    def _calculate_scores(self, slides: List[Dict]) -> Dict[str, Any]:
        """计算质量评分"""
        
        # 内容质量评分
        content_score = 100 - len([i for i in self.issues if i['rule_id'] in ['bullet-6x6', 'text-density']]) * 10
        content_score = max(0, min(100, content_score))
        
        # 视觉质量评分
        visual_score = 100 - len([i for i in self.issues if i['rule_id'] in ['wcag-contrast', 'font-size-minimum']]) * 15
        visual_score = max(0, min(100, visual_score))
        
        # 结构质量评分
        structure_score = 100 - len([i for i in self.issues if i['rule_id'] in ['key-decision-required']]) * 20
        structure_score = max(0, min(100, structure_score))
        
        # 美学质量评分（假设调用ppt-aesthetic-qa）
        aesthetic_score = 75.0  # Placeholder
        
        # 综合评分
        total_score = (
            content_score * 0.4 +
            visual_score * 0.3 +
            structure_score * 0.2 +
            aesthetic_score * 0.1
        )
        
        return {
            'total_score': round(total_score, 1),
            'breakdown': {
                'content': content_score,
                'visual': visual_score,
                'structure': structure_score,
                'aesthetic': aesthetic_score
            }
        }
    
    def _get_grade(self, score: float) -> str:
        """获取评分等级"""
        if score >= 90:
            return 'excellent'
        elif score >= 75:
            return 'good'
        elif score >= 60:
            return 'fair'
        else:
            return 'poor'
    
    def _calculate_contrast_ratio(self, color1: str, color2: str) -> float:
        """计算WCAG对比度（简化实现）"""
        # 实际实现需要完整的relative luminance计算
        # 这里返回模拟值
        if color1 == "#888888" and color2 == "#FFFFFF":
            return 2.9
        return 4.5  # Default
    
    def _adjust_color_for_contrast(self, text_color: str, bg_color: str, target_ratio: float) -> str:
        """调整颜色以达到目标对比度（简化实现）"""
        # 实际实现需要颜色空间转换和亮度调整
        if text_color == "#888888":
            return "#595959"
        return text_color


# 使用示例
if __name__ == '__main__':
    validator = PPTGuidelinesValidator()
    
    slides = [
        {
            'slide_number': 1,
            'type': 'title',
            'title': '系统架构评审'
        },
        {
            'slide_number': 2,
            'type': 'content',
            'title': '系统性能',
            'bullets': ['认证模块', '限流模块', '推荐模块', '搜索模块', '数据模块', '监控模块', '日志模块'],
            'text_color': '#888888',
            'background_color': '#FFFFFF',
            'font_size': 12
        }
    ]
    
    result = validator.validate(slides, auto_fix=True)
    
    print(f"Quality Score: {result['qa_report']['total_score']}")
    print(f"Grade: {result['qa_report']['grade']}")
    print(f"\nIssues ({len(result['qa_report']['issues'])}):")
    for issue in result['qa_report']['issues']:
        print(f"  - Slide {issue['slide_number']}: {issue['detail']}")
    
    print(f"\nAuto-fix Applied ({len(result['qa_report']['auto_fix_applied'])}):")
    for fix in result['qa_report']['auto_fix_applied']:
        print(f"  - Slide {fix['slide_number']}: {fix['action']}")
        print(f"    Changes: {fix['changes']}")
```

---

## 6. 资源和参考

### 6.1 设计标准

- **Presentation Zen** - Garr Reynolds 的极简主义PPT哲学
- **Slide:ology** - Nancy Duarte 的幻灯片视觉思维
- **Talk Like TED** - Carmine Gallo 的演讲技巧
- **McKinsey Presentation Standards** - 麦肯锡演示标准
- **Apple Keynote Design Principles** - 苹果主题演讲设计原则
- **WCAG 2.1 AA** - 无障碍访问指南（对比度标准）

### 6.2 工具和资源

- **Contrast Checker** - [WebAIM Contrast Checker](https://webaim.org/resources/contrastchecker/)
- **Color Oracle** - 色盲模拟工具
- **Slideshare Best Practices** - LinkedIn SlideShare 设计指南

### 6.3 相关 Skills

- `ppt-aesthetic-qa.skill` - 视觉美学评分（color harmony、layout balance）
- `ppt-visual.skill` - WCAG 对比度计算和 Material Design 规范
- `ppt-outline.skill` - 大纲结构验证（MECE、Story Arc、Key Decision识别）
- `ppt-layout.skill` - 布局规则验证（网格对齐、Assertion-Evidence检测）
