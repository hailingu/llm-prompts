# PPT Design Guidelines — Industry Best Practices

**目的**：为PPT生成系统提供世界级的设计规范，整合McKinsey、IDEO、Apple Keynote等顶尖机构的设计哲学，转化为可执行、可量化的质量标准。

**适用范围**：技术评审、产品路演、高管汇报、学术报告等所有专业演示场景。

---

## 核心设计原则

### 1. Clarity（清晰性）— Steve Jobs标准
> "Simplicity is the ultimate sophistication." — Leonardo da Vinci (via Steve Jobs)

**执行规则**：
- ✅ 每页聚焦1个核心信息点
- ✅ 标题传达结论，而非话题（Assertion-Evidence原则）
- ✅ 删除一切非必要元素（Signal vs Noise）

### 2. Speakability（可讲性）— TED Talk标准
> "If you can't explain it simply, you don't understand it well enough." — Richard Feynman

**执行规则**：
- ✅ 每页配备Speaker Notes（简短讲稿）
- ✅ 文字作为视觉提示，而非完整讲稿
- ✅ 6x6法则：≤6个bullets，每个≤6个词

### 3. Visual First（视觉优先）— Edward Tufte原则
> "Graphical excellence is that which gives to the viewer the greatest number of ideas in the shortest time with the least ink in the smallest space." — Edward Tufte

**执行规则**：
- ✅ 复杂信息→图表（Cleveland Perception Hierarchy）
- ✅ Data-Ink Ratio最大化（移除图表垃圾）
- ✅ 留白≥30%（Presentation Zen美学）

### 4. Auditability（可审计性）— McKinsey标准
> "Pyramid Principle: Answer first, then group and summarize supporting arguments." — Barbara Minto

**执行规则**：
- ✅ Executive Summary在前3页
- ✅ 关键决策独立呈现（决策+理由+备选方案）
- ✅ 数据来源完整标注（学术诚信）

---

## 7大设计哲学（可组合应用）

### 1️⃣ Presentation Zen — Garr Reynolds（简约主义）

**核心理念**：Less is more. 用留白和视觉引导注意力。

**可执行规则**：
```yaml
max_bullets_per_slide: 3
min_whitespace_ratio: 0.6    # 60%留白
single_topic_per_slide: true
visuals_over_text: true
```

**适用场景**：产品发布、创意提案、品牌故事

**参考**：*Presentation Zen* (2008), Apple Keynote风格

---

### 2️⃣ Assertion-Evidence — Michael Alley（学术标准）

**核心理念**：标题=论断句，正文=证据（图表/数据）。

**可执行规则**：
```yaml
assertion_title: true         # 标题为完整句
evidence_required: true       # 必须有图表/数据
min_visual_elements: 1
require_citation: true        # 学术场景强制来源
```

**检测方法**：
```python
# 标题检测：包含动词或判断词
def is_assertion(title):
    return any(verb in title for verb in ['提升', '降低', '证明', '显示'])
```

**适用场景**：技术评审、学术报告、数据汇报

**参考**：*The Craft of Scientific Presentations* (Michael Alley)

---

### 3️⃣ Guy Kawasaki 10/20/30 Rule（创投标准）

**核心理念**：10张幻灯片、20分钟演讲、最小30pt字号。

**可执行规则**：
```yaml
max_slides_for_pitch: 10      # 核心内容
max_duration_minutes: 20
min_body_font_pt: 30          # 确保后排可见
key_info_within_first_n: 3   # 前3页包含Problem/Solution
```

**必备页面**：
1. Title（标题）
2. Problem（问题）
3. Solution（解决方案）
4. Business Model（商业模式）
5. Market Size（市场规模）
6. Competition（竞争分析）
7. Team（团队）
8. Financials（财务预测）
9. Current Status（当前进展）
10. Timeline（时间表）

**适用场景**：融资路演、高管汇报、项目提案

**参考**：*The Art of the Start 2.0* (Guy Kawasaki, 2015)

---

### 4️⃣ McKinsey Pyramid Principle — Barbara Minto（咨询标准）

**核心理念**：结论先行（SCQA框架）+ MECE逻辑。

**SCQA框架**：
```
Situation（背景）→ Complication（冲突）→ Question（疑问）→ Answer（答案）
```

**可执行规则**：
```yaml
executive_summary_required: true
conclusion_first: true
so_what_test: true           # 标题传达insight
complete_annotation: true    # 图表自解释
```

**图表标注checklist**：
- ✅ 标题（传达结论）
- ✅ 坐标轴单位
- ✅ 数据来源
- ✅ 时间窗口
- ✅ 关键数据标注

**适用场景**：战略咨询、业务分析、决策汇报

**参考**：*The Pyramid Principle* (Barbara Minto, McKinsey方法论)

---

### 5️⃣ Edward Tufte — Data Integrity（数据诚信）

**核心理念**：最大化Data-Ink Ratio，移除Chart Junk，避免误导。

**可执行规则**：
```yaml
y_axis_starts_at_zero: true   # 柱状图Y轴从0开始
max_chart_categories: 5       # 饼图≤5分类
no_3d_effects: true           # 禁用3D/阴影/渐变
no_misleading_scales: true    # 禁止截断Y轴
```

**Chart Junk清单**（需移除）：
- ❌ 3D效果（误导视角）
- ❌ 过多网格线
- ❌ 装饰性渐变
- ❌ 不必要的图例重复
- ❌ 低对比度颜色

**Cleveland Perception Hierarchy**（准确度排序）：
1. Position（位置）→ 误差率 ~5%
2. Length（长度）→ 误差率 ~10%
3. Angle（角度）→ 误差率 ~20%
4. Area（面积）→ 误差率 ~25%
5. Volume（体积）→ 误差率 ~40%

**推荐**：柱状图/折线图 > 饼图 > 3D图

**适用场景**：数据分析、科学报告、金融汇报

**参考**：*The Visual Display of Quantitative Information* (Edward Tufte, 1983)

---

### 6️⃣ Takahashi Method — 高桥征义（极简主义）

**核心理念**：一页一词，超大字号，快速切换。

**可执行规则**：
```yaml
max_words_per_slide: 3
max_bullets_per_slide: 0      # 禁用bullets
min_body_font_pt: 80          # 超大字号
allow_high_slide_count: true  # 允许100+页
```

**适用场景**：快节奏演讲、TED-style Talk、关键词强调

**示例**：
```
[Slide 1]: 简约
[Slide 2]: 专注
[Slide 3]: 震撼
```

**参考**：高桥征义（Ruby社区，2005）

---

### 7️⃣ Signal vs Noise — 37signals原则

**核心理念**：每个元素要么是信号（核心信息），要么是噪音（干扰）。移除噪音。

**可执行规则**：
```yaml
max_unique_colors: 5          # 限制配色
max_font_families: 2          # 限制字体
allow_animations: false       # 禁用动画
allow_decorative_elements: false
logo_only_on_title_end: true  # Logo仅首尾页
```

**Noise清单**（需移除）：
- ❌ 无意义过渡动画
- ❌ 每页重复的Logo/页眉
- ❌ 装饰性图形
- ❌ 过多配色（>5种）
- ❌ 混用多种字体

**适用场景**：极简设计、现代科技风格

**参考**：*Signal vs. Noise* (37signals/Basecamp)

---

## 哲学组合策略（按场景选择）

| 演示类型         | 推荐哲学组合                       | 关键规则                              |
| ---------------- | ---------------------------------- | ------------------------------------- |
| ---------        | ------------                       | ---------                             |
| 技术架构评审     | Assertion-Evidence + Tufte         | 论断标题，图表强制，Y轴从0，数据来源  |
| 融资路演 (Pitch) | 10/20/30 Rule + Pyramid Principle  | ≤10页，30pt字号，结论前置，SCQA       |
| 产品发布会       | Presentation Zen + Signal vs Noise | ≤3 bullets，60%留白，无装饰，视觉优先 |
| 学术报告         | Assertion-Evidence + Tufte         | 论断句，完整标注，来源强制            |
| 快节奏演讲       | Takahashi + Signal vs Noise        | ≤3词/页，80pt字号，无噪音             |
| 战略咨询         | Pyramid Principle + Tufte          | SCQA框架，Executive Summary，数据诚信 |

---

## 强制规则（MUST - 不可违背）

### Typography（字体排版）
```yaml
min_title_font_pt: 36         # 标题≥36pt
min_body_font_pt: 18          # 正文≥18pt（pitch模式30pt）
max_bullets_per_slide: 5      # ≤5条bullets（推荐3条）
max_words_per_bullet: 8       # 每条≤8个词
line_height: 1.5              # 行高1.5x（可读性）
```

### Layout（布局）
```yaml
max_text_density_percent: 40  # 文字占比≤40%
min_whitespace_ratio: 0.3     # 留白≥30%
grid_alignment: true          # 对齐12列网格
margin_px: 48                 # 边距48px（8点网格）
```

### Accessibility（可访问性）
```yaml
wcag_contrast_level: "AA"     # WCAG 2.1 AA标准
min_contrast_ratio: 4.5       # 普通文字≥4.5:1
large_text_ratio: 3.0         # 大文字(≥18pt)≥3:1
alt_text_required: true       # 图片必须有alt text
```

### Content Quality（内容质量）
```yaml
require_speaker_notes: true   # 必须有讲稿
require_decision_slide_within_first_n: 5  # 前5页包含决策
require_image_attribution: true            # 图片必须标注来源
require_data_source: true                  # 图表必须标注数据来源
```

---

## 量化检查标准（QA阈值）

### 评分算法
```python
severity_weights = {
    'critical': -15,   # 严重问题（对比度不足、缺少决策页）
    'major': -5,       # 主要问题（字号过小、bullets过多）
    'minor': -2        # 次要问题（留白不足、动画过长）
}

score = max(0, 100 + sum(deductions))

grade_mapping = {
    90-100: 'A',  # 优秀
    80-89: 'B',   # 良好
    70-79: 'C',   # 合格
    <70: 'D'      # 不合格（阻断发布）
}
```

### Pass门槛
```yaml
qa_pass_threshold: 70         # 最低70分通过
critical_issues_allowed: 0    # Critical问题必须为0
```

---

## 自动修复策略（Auto-Fix）

### 可自动修复
✅ **6x6 Rule违规** → 拆分为2页  
✅ **对比度不足** → 使用深色变体  
✅ **字号过小** → 统一调整为30pt  
✅ **缺少数据来源** → 自动添加"来源: [占位]"  

### 需人工审查
⚠️ **缺少Key Decision页** → 插入占位，标注AUTO_INSERT  
⚠️ **版权问题** → 阻断合并，需法务确认  
⚠️ **逻辑跳跃** → 需作者补充过渡页  

---

## 快速使用（开发者指南）

### Python加载配置
```python
import json
from pathlib import Path

# 加载完整规则配置
rules = json.loads(
    Path('standards/ppt-guidelines/ppt-guidelines.json').read_text(encoding='utf-8')
)

# 选择预设（例如融资路演）
preset = rules['_presets']['executive-pitch']
# → {'philosophy': ['10-20-30-rule', 'pyramid-principle'], 
#    'max_slides_for_pitch': 15, 'min_body_font_pt': 30, ...}

# 运行质量检查
# response = requests.post('http://localhost:8000/skill/ppt-guidelines/check',
#                          json={'slides': slides, 'rules': preset})
```

### CI/CD集成（GitHub Actions）
```yaml
name: PPT Quality Gate
on: [pull_request]
jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Validate Schema
        run: |
          python -c "
          import json, jsonschema
          data = json.load(open('standards/ppt-guidelines/ppt-guidelines.json'))
          schema = json.load(open('standards/ppt-guidelines/schema.json'))
          jsonschema.validate(data, schema)
          print('✅ Schema valid')
          "
      
      - name: Run QA Check
        run: |
          # 调用ppt-guidelines.skill检查所有slides
          # 如果score < 70或有critical问题，fail构建
          python scripts/check_ppt_quality.py --threshold 70
```

---

## Key Decisions模板

每个关键决策页必须包含：

```yaml
决策内容: "采用WebAssembly + WebGL实现前端实时渲染"

备选方案:
  - 方案A: 完全后端渲染（CPU/GPU）
  - 方案B: 纯Canvas2D
  - 方案C: 混合渲染（首屏后端，交互前端）

评估标准:
  - 延迟: P95 < 100ms
  - 成本: < $0.01 / 1000次渲染
  - 离线能力: 支持离线运行

风险分析:
  - 浏览器兼容性（IE不支持WASM）
  - 二进制体积（初始加载3MB）
  - 调试复杂度（WASM调试工具不成熟）

推荐理由: 
  满足P95延迟要求，降低服务器成本67%，支持离线场景
```

---

## 参考资料（业界经典）

### 必读书籍
1. **Presentation Zen** — Garr Reynolds (2008)  
   简约主义与东方美学的演示设计圣经

2. **Slide:ology** — Nancy Duarte (2008)  
   Apple/TED御用设计师的视觉思维方法

3. **The Visual Display of Quantitative Information** — Edward Tufte (1983)  
   数据可视化诚实性的奠基之作

4. **The Pyramid Principle** — Barbara Minto (1987)  
   McKinsey逻辑结构方法论

5. **The Assertion-Evidence Approach** — Michael Alley (2003)  
   学术报告的黄金标准

6. **Resonate** — Nancy Duarte (2010)  
   故事化演讲的结构设计

7. **Clear and to the Point** — Stephen Kosslyn (2007)  
   认知科学视角的幻灯片设计

### 在线资源
- [WCAG 2.1 Contrast Guidelines](https://www.w3.org/WAI/WCAG21/Understanding/contrast-minimum.html)
- [Guy Kawasaki's Blog](https://guykawasaki.com/the_102030_rule/)
- [Edward Tufte's Website](https://www.edwardtufte.com/)
- [Presentation Zen Blog](https://www.presentationzen.com/)

### 工具推荐
- **Figma/Sketch** — 设计Token系统
- **python-pptx** — 自动化生成
- **matplotlib/seaborn** — 科学图表
- **ColorBrewer** — 配色方案（色盲友好）

---

## 版本管理

**当前版本**：v2.0 (2026-01-28)  
**维护者**：PPT Creative Director Team  
**更新频率**：季度review，重大变更需团队投票

**变更日志**：
- v2.0 (2026-01): 整合7大设计哲学，添加McKinsey/Tufte标准
- v1.0 (2025-12): 初始版本，基础规则定义

---

## 行业模板库（Industry Templates）

为不同行业场景提供开箱即用的专业模板，每个模板包含完整的设计规则、数据结构和最佳实践。

### 战略咨询（Strategy Consulting）

#### 1. BCG Growth-Share Matrix（波士顿矩阵）
📂 [templates/ppt/bcg-matrix/](../../templates/ppt/bcg-matrix/)

**用途**：业务组合分析、资源分配决策  
**适用场景**：企业战略规划、投资组合管理、产品线评估

**核心元素**：
- 2×2矩阵（明星/金牛/问题/瘦狗）
- 气泡图（气泡大小=收入/利润）
- 相对市场份额 vs 市场增长率
- 每个象限的战略建议

**快速使用**：
```python
import json
template = json.load(open('templates/ppt/bcg-matrix/template.json'))
# 准备数据：business_units = [...]
slides = generate_bcg_matrix(template, business_units)
```

---

#### 2. SWOT Analysis（态势分析）
📂 [templates/ppt/swot-analysis/](../../templates/ppt/swot-analysis/)

**用途**：优势/劣势/机会/威胁分析  
**适用场景**：产品策略、市场进入决策、竞争分析

**四象限逻辑**：
- **内部因素**：Strengths（优势）+ Weaknesses（劣势）
- **外部因素**：Opportunities（机会）+ Threats（威胁）

**战略组合**：
- SO战略（优势+机会）→ 进攻策略
- WT战略（劣势+威胁）→ 防御策略

---

#### 3. Porter's Five Forces（波特五力模型）
📂 [templates/ppt/porter-five-forces/](../../templates/ppt/porter-five-forces/)

**用途**：行业竞争结构分析  
**适用场景**：市场进入评估、行业吸引力判断

**五种力量**（1-5分评分）：
1. 同业竞争强度（Competitive Rivalry）
2. 供应商议价能力（Supplier Power）
3. 买方议价能力（Buyer Power）
4. 新进入者威胁（New Entrants）
5. 替代品威胁（Substitutes）

**可视化**：五角星雷达图 + 强度评分

---

### 项目管理（Project Management）

#### 4. Gantt Chart（甘特图）
📂 [templates/ppt/gantt-chart/](../../templates/ppt/gantt-chart/)

**用途**：项目进度管理、时间线规划  
**适用场景**：软件开发、工程建设、产品发布

**核心元素**：
- 任务条（长度=持续时间）
- 里程碑标记（菱形）
- 依赖关系（箭头连线）
- 今日线（垂直虚线）
- 进度指示（条形图填充）

**最佳实践**：
- 高亮关键路径（Critical Path）
- 最多显示15个任务（避免拥挤）
- 显示任务负责人

---

### 财务分析（Financial Analysis）

#### 5. Waterfall Chart（瀑布图）
📂 [templates/ppt/waterfall-chart/](../../templates/ppt/waterfall-chart/)

**用途**：利润桥分析、现金流分析  
**适用场景**：财务汇报、成本结构变化、预算vs实际

**视觉要素**：
- 起始柱（蓝色）
- 增项柱（绿色，向上）
- 减项柱（红色，向下）
- 连接线（虚线）
- 结束柱（蓝色）

**公式**：`结束值 = 起始值 + Σ(增项) - Σ(减项)`

---

### 模板选择指南

| 分析目标       | 推荐模板             | 关键输出          |
| -------------- | -------------------- | ----------------- |
| ---------      | ---------            | ---------         |
| 评估业务组合   | BCG Matrix           | 资源分配优先级    |
| 识别优劣势     | SWOT Analysis        | 战略方向（SO/WT） |
| 评估行业吸引力 | Porter's Five Forces | 进入/退出决策     |
| 制定项目计划   | Gantt Chart          | 时间线+关键路径   |
| 分析利润变化   | Waterfall Chart      | 贡献因素分解      |

---

### 模板使用流程

1. **选择模板**：根据分析目标选择合适模板
2. **准备数据**：按照`template.json`的`data_structure`准备数据
3. **验证规则**：运行`rules.yaml`检查数据完整性
4. **生成幻灯片**：调用生成函数创建PPT
5. **质量检查**：使用`ppt-guidelines.skill`验证设计规范

---

## 相关文档

- 📄 [ppt-guidelines.json](ppt-guidelines.json) — 机器可读配置
- 📄 [ppt-guidelines.md](ppt-guidelines.md) — 完整中文规范（详细版）
- 🛠️ [skills/ppt-guidelines.skill.md](../../skills/ppt-guidelines.skill.md) — 检查引擎实现
- 🎨 [standards/ppt-agent-collaboration-protocol.md](../ppt-agent-collaboration-protocol.md) — Agent协作流程
- 📦 [templates/ppt/](../../templates/ppt/) — 行业模板库

---

**同步策略**：更新规则时，同步修改以下3个文件：
1. `ppt-guidelines.json`（运行时配置）
2. `GUIDELINES.md`（本文件，设计哲学）
3. `skills/ppt-guidelines.skill.md`（检查实现）
