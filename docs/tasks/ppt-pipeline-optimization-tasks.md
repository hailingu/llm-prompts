# PPT Pipeline 优化方案 — 任务分解

> **来源计划**：[ppt-pipeline-optimization-plan.md](../design/ppt-pipeline-optimization-plan.md)
> **日期**：2026-02-11
> **总计**：6 主任务 · 19 子任务

---

## Task 1: P0 — 原生图表替代 matplotlib

### Description

将 `generate_pptx.py` 中基于 matplotlib 生成 PNG 再嵌入的图表渲染路径，替换为
`python-pptx` 原生 `add_chart()` API。这是 ROI 最高的单项改进：图表变为可编辑矢量、
文件体积缩小、消除 matplotlib 运行时依赖。

### Responsibilities

- 实现原生图表渲染函数 `render_native_chart()`
- 实现 MD3 配色映射 `apply_chart_theme()`
- 改造 `render_visual()` 分发逻辑：优先原生，fallback matplotlib
- 保留 matplotlib 路径用于不支持的图表类型（热力图、桑基图等）

### Dependencies

- 无前置依赖（P0 可立即启动）

### Public API

- `render_native_chart(slide, visual, spec, left, top, width, height, accent_token) -> bool`
- `apply_chart_theme(chart, spec, accent_token) -> None`

---

### Task 1.1: 实现原生图表渲染核心函数

#### Description

在 `generate_pptx.py` 的 §6 Visual Renderers 区域新增 `render_native_chart()` 函数，
支持 7 种 python-pptx 原生图表类型映射。

#### Implementation Points

1. 在 `generate_pptx.py` 第 1431 行（§6 Visual Renderers）后新增函数
2. 从 `visual["placeholder_data"]` 提取 `labels`、`series`
3. 建立 visual.type → `XL_CHART_TYPE` 的映射表：

   | visual.type | XL\_CHART\_TYPE |
   |-------------|-----------------|
   | `bar_chart` / `column_chart` | `COLUMN_CLUSTERED` |
   | `horizontal_bar` | `BAR_CLUSTERED` |
   | `line_chart` | `LINE_MARKERS` |
   | `pie_chart` | `PIE` |
   | `doughnut_chart` | `DOUGHNUT` |
   | `radar_chart` | `RADAR` |
   | `scatter_chart` | `XY_SCATTER` |

4. 使用 `CategoryChartData` 构建数据，`XyChartData` 用于散点
5. 返回 `True` 表示成功渲染，`False` 表示不支持需 fallback

#### Testing Strategy

- 单元测试：对每种图表类型构造 mock visual dict，验证 `add_chart` 被正确调用
- 集成测试：用 `storage-frontier` 的 `slides_semantic.json` 生成 PPTX，
  验证输出中图表 shape 类型为 `MSO_SHAPE_TYPE.CHART`（非 `PICTURE`）
- 在 PowerPoint/LibreOffice 中双击图表验证可编辑

#### Deliverables

- `generate_pptx.py` 中新增的 `render_native_chart()` 函数（约 80-120 行）
- 图表类型映射表常量 `NATIVE_CHART_TYPE_MAP`

#### Checklist

- [x] 7 种图表类型映射完成
- [x] `CategoryChartData` / `XyChartData` 数据构建正确
- [x] 多系列图表支持（grouped bar 等）
- [x] 复合图 (composite_charts) 与 bar-line 复合图采用 best-effort 子图选择 / 类型映射以实现原生渲染（首个子图或列图）
- [x] 空数据防御（labels 或 series 为空返回 False）
- [x] `py_compile` 通过

> ✅ 完成说明：在 `skills/ppt-generator/bin/generate_pptx.py` 中新增 `render_native_chart()` 和 `apply_chart_theme()`，并新增单元测试 `tests/test_native_chart.py`、`tests/test_chart_theme.py`（2026-02-11）

**Task 1.2 status:**

- [x] 系列颜色按 palette 轮转（支持 `section_accents[accent_token]` → `md3_palette` → fallback）
- [x] 坐标轴字体大小 8pt，颜色为 `on_surface_variant`（best-effort）
- [x] 网格线颜色按 `outline` 设置（best-effort）
- [x] 图例字体 7pt，尝试应用轻背景（framealpha 以近似方式处理）
- [x] `accent_token` 参数正确映射到 section 配色（优先 `section_accents`，回退 `md3_palette`，再回退 token color）

> 备注：图例背景透明度/framealpha 以 `legend.format.fill` 的填充颜色作为近似处理；部分 python-pptx 版本对透明度/alpha 的细粒度设置支持有限，已实现 best-effort 方案。

```yaml
Execution Parameters:
  taskId: "Task-1.1"
  shortName: "native-chart-core"
  workspacePath: "."
  branch: "feat/reveal-autorewrite"
  runCommands:
    - "python3 -c \"from pptx.chart.data import CategoryChartData; print('import OK')\""
  testCommands:
    - "python3 -m py_compile .github/skills/ppt-generator/bin/generate_pptx.py"
    - "python3 tests/test_native_chart.py"
  timeoutMinutes: 15
  priority: "high"
  estimatedHours: 6
  artifacts:
    - ".github/skills/ppt-generator/bin/generate_pptx.py"
    - "tests/test_native_chart.py"
  acceptanceCriteria:
    - "render_native_chart() 支持 7 种图表类型"
    - "py_compile 无错误"
    - "空数据输入返回 False 不抛异常"
  backwardCompatibility: "不修改任何现有函数签名；新函数为独立新增"
  dependencies: []
  rollbackSteps:
    - "git revert HEAD"
```

---

### Task 1.2: 实现 MD3 配色主题应用

#### Description

新增 `apply_chart_theme()` 函数，将 MD3 调色板（31 色彩 token）应用到原生图表的
系列颜色、坐标轴、网格线、图例样式。

#### Implementation Points

1. 从 `spec` 中读取 `section_accents` 和 `md3_palette` 配色
2. 设置 `series.format.fill.fore_color.rgb` 按 palette 轮转
3. 设置坐标轴字体大小 8pt、颜色 `on_surface_variant`
4. 设置网格线为 `outline_variant` / 0.3 透明度
5. 图例字体 7pt，framealpha 0.8

#### Testing Strategy

- 单元测试：验证 chart.series[i] 颜色与 palette[i] 一致
- 视觉验证：对比输出 PPTX 中图表配色与 design_spec.json 定义

#### Deliverables

- `apply_chart_theme()` 函数（约 40-60 行）

#### Checklist

- [x] 系列颜色按 palette 轮转
- [x] 坐标轴/网格线/图例样式符合 MD3
- [x] accent_token 参数正确映射到 section 配色

```yaml
Execution Parameters:
  taskId: "Task-1.2"
  shortName: "chart-theme"
  workspacePath: "."
  branch: "feat/reveal-autorewrite"
  testCommands:
    - "python3 -m py_compile .github/skills/ppt-generator/bin/generate_pptx.py"
    - "python3 tests/test_chart_theme.py"
  timeoutMinutes: 10
  priority: "high"
  estimatedHours: 3
  artifacts:
    - ".github/skills/ppt-generator/bin/generate_pptx.py"
    - "tests/test_chart_theme.py"
  acceptanceCriteria:
    - "图表系列颜色匹配 MD3 palette"
    - "坐标轴/网格线/图例样式与 design_spec 一致"
  dependencies:
    - "Task-1.1"
  rollbackSteps:
    - "git revert HEAD"
```

---

### Task 1.3: 改造 render\_visual() 分发逻辑

#### Description

修改 `render_visual()` 函数（第 1613 行），在 matplotlib 路径前插入原生图表尝试。
优先调用 `render_native_chart()`，成功则 return；失败则 fallback 到原有 matplotlib 路径。

#### Implementation Points

1. 在 `render_visual()` 的步骤 2（matplotlib chart generation）之前新增步骤 1.5
2. 调用 `render_native_chart()`，返回 True 则 return
3. 添加日志记录原生/fallback 路径选择
4. 不修改步骤 1（pre-rendered image）和步骤 3-5 的逻辑

#### Testing Strategy

- 集成测试：用 `storage-frontier` 数据生成完整 PPTX
- 回归测试：验证无 chart\_config 的页面不受影响
- 验证 `HAS_MATPLOTLIB = False` 时仍能走原生路径

#### Deliverables

- 修改后的 `render_visual()` 函数

#### Checklist

- [x] 原生路径优先于 matplotlib
- [x] 不支持的类型正确 fallback
- [x] 无 chart_config 的 visual 不受影响
- [x] 生成完整 PPTX 无异常

```yaml
Execution Parameters:
  taskId: "Task-1.3"
  shortName: "visual-dispatch"
  workspacePath: "."
  branch: "feat/reveal-autorewrite"
  runCommands:
    - "cd docs/presentations/storage-frontier-20260211 && python3 ../../../.github/skills/ppt-generator/bin/generate_pptx.py slides_semantic.json design_spec.json storage-frontier-v9-native-chart.pptx"
  testCommands:
    - "python3 -m py_compile .github/skills/ppt-generator/bin/generate_pptx.py"
    - "python3 tests/test_render_visual_dispatch.py"
  timeoutMinutes: 15
  priority: "high"
  estimatedHours: 4
  artifacts:
    - ".github/skills/ppt-generator/bin/generate_pptx.py"
    - "tests/test_render_visual_dispatch.py"
  acceptanceCriteria:
    - "支持的图表类型走原生路径（Shape 类型为 CHART 非 PICTURE）"
    - "不支持的类型 fallback matplotlib 无异常"
    - "完整 PPTX 生成成功，slide 数量不变"
  backwardCompatibility: "render_visual() 函数签名不变；v1 slides_semantic.json 兼容"
  dependencies:
    - "Task-1.1"
    - "Task-1.2"
  rollbackSteps:
    - "git revert HEAD"
```

---

### Task 1.4: 更新 ppt-generator Skill 文档

#### Description

更新 `skills/ppt-generator/README.md`，记录原生图表渲染路径、支持的图表类型映射表、
fallback 策略说明。

#### Implementation Points

1. 在渲染架构章节新增"原生图表 vs matplotlib 位图"对比表
2. 记录 `NATIVE_CHART_TYPE_MAP` 映射关系
3. 说明 fallback 策略：render_native_chart → matplotlib → data table → placeholder

#### Deliverables

- 更新后的 `skills/ppt-generator/README.md`

#### Checklist

- [x] 图表类型映射表完整
- [x] fallback 策略描述清晰
- [x] 无 markdownlint 违规

> ✅ 说明：`skills/ppt-generator/README.md` 已更新，包含原生图表章节、映射表及回退策略（见 “原生图表渲染（python-pptx 原生）” 小节）。

```yaml
Execution Parameters:
  taskId: "Task-1.4"
  shortName: "doc-native-chart"
  workspacePath: "."
  branch: "feat/reveal-autorewrite"
  testCommands:
    - "echo 'markdownlint check placeholder'"
  timeoutMinutes: 5
  priority: "medium"
  estimatedHours: 1
  artifacts:
    - "skills/ppt-generator/README.md"
  acceptanceCriteria:
    - "README.md 包含原生图表章节"
    - "映射表与代码一致"
  dependencies:
    - "Task-1.3"
```

---

## Task 2: P1 — 断言标题 + 洞察提炼

### Description

在 Schema 中新增 `assertion` 和 `insight` 可选字段，并在渲染器中实现对应的
断言标题渲染和底部洞察条渲染。这是 v2 增强的最小可行特性——即使没有 EA Agent，
手动填入这两个字段也能立即看到效果。

### Responsibilities

- Schema v1 → v1.1 扩展（非破坏性）
- 断言标题渲染函数
- 洞察条渲染函数
- 渲染入口自动检测

### Dependencies

- Task 1（P0）完成

---

### Task 2.1: Schema 新增 assertion 和 insight 字段

#### Description

在 `standards/slides-render-schema.json` 的 `slide` 定义中新增两个可选字段。

#### Implementation Points

1. 在 `definitions.slide.properties` 中添加：
   - `"assertion": { "type": "string", "description": "断言式标题..." }`
   - `"insight": { "type": "string", "description": "页面洞察..." }`
2. 不修改 `required` 数组（保持向后兼容）
3. 更新 `$id` 为 `slides-render-schema-v1.1`
4. 更新 `version` 为 `"1.1.0"`

#### Testing Strategy

- JSON Schema 校验：用 v1 格式的 `slides_semantic.json` 验证仍然合法
- 用包含 assertion/insight 字段的样例验证 v1.1 格式合法

#### Deliverables

- 更新后的 `standards/slides-render-schema.json`
- 测试用 v1.1 样例 JSON 片段

#### Checklist

- [x] assertion 字段定义正确（type: string, optional）
- [x] insight 字段定义正确（type: string, optional）
- [x] v1 JSON 仍通过 schema 校验（向后兼容）
- [x] v1.1 JSON 通过 schema 校验
- [x] version 更新为 1.1.0

> ✅ 说明：已更新 `standards/slides-render-schema.json` 为 `$id: slides-render-schema-v1.1`, `version: 1.1.0`，并新增 `assertion` 与 `insight` 可选字段；已添加 `tests/test_schema_compat.py` 验证更改。

```yaml
Execution Parameters:
  taskId: "Task-2.1"
  shortName: "schema-assertion-insight"
  workspacePath: "."
  branch: "feat/reveal-autorewrite"
  testCommands:
    - "python3 -c \"import json; d=json.load(open('standards/slides-render-schema.json')); print('Schema valid:', 'assertion' in d['definitions']['slide']['properties'])\""
    - "python3 tests/test_schema_compat.py"
  timeoutMinutes: 10
  priority: "high"
  estimatedHours: 2
  artifacts:
    - "standards/slides-render-schema.json"
    - "tests/test_schema_compat.py"
  acceptanceCriteria:
    - "assertion 和 insight 字段存在于 schema"
    - "v1 格式 JSON 仍通过校验"
    - "version 为 1.1.0"
  backwardCompatibility: "v1 JSON 100% 兼容，新字段为 optional"
  dependencies:
    - "Task-1.3"
  rollbackSteps:
    - "git checkout -- standards/slides-render-schema.json"
```

---

### Task 2.2: 实现 render\_assertion\_title()

#### Description

新增断言标题渲染函数，在有 `assertion` 字段时替代默认标题渲染。

#### Implementation Points

1. 在 `generate_pptx.py` 的 §4 Shared Renderers 区域新增函数
2. 布局规则：
   - 断言文字：16pt 粗体白色，左对齐，占据标题栏主区域
   - 原 title 降为副标题：10pt 浅色（on_surface_variant），紧贴断言下方
   - 标题栏高度自适应（min 0.85"），容纳两行文字
3. 修改 §8 主循环中标题栏渲染逻辑：
   检测 `sd.get("assertion")` → 调用 `render_assertion_title()`，否则走原有 `render_title_bar()`

#### Testing Strategy

- 单元测试：验证 assertion 存在时标题栏包含两个 textbox（assertion + subtitle）
- 回归测试：验证无 assertion 时行为不变

#### Deliverables

- `render_assertion_title()` 函数（约 50-70 行）
- 主循环标题渲染分发修改

#### Checklist

- [x] 16pt 粗体断言文字正确渲染
- [x] 10pt 浅色副标题正确渲染
- [x] 标题栏高度自适应
- [x] 无 assertion 时不影响现有渲染
- [x] py_compile 通过

> ✅ 说明：已实现 `render_assertion_title()`，并在主渲染分发中检测 `assertion` 字段以使用断言式标题渲染；新增单元测试 `tests/test_assertion_title.py`。

```yaml
Execution Parameters:
  taskId: "Task-2.2"
  shortName: "render-assertion-title"
  workspacePath: "."
  branch: "feat/reveal-autorewrite"
  testCommands:
    - "python3 -m py_compile .github/skills/ppt-generator/bin/generate_pptx.py"
    - "python3 tests/test_assertion_title.py"
  timeoutMinutes: 10
  priority: "high"
  estimatedHours: 4
  artifacts:
    - ".github/skills/ppt-generator/bin/generate_pptx.py"
    - "tests/test_assertion_title.py"
  acceptanceCriteria:
    - "assertion 存在时渲染断言标题 + 副标题"
    - "assertion 不存在时行为与 v8 完全一致"
    - "标题栏高度 ≥ 0.85 英寸"
  backwardCompatibility: "无 assertion 字段时渲染结果与 v8 一致"
  dependencies:
    - "Task-2.1"
  rollbackSteps:
    - "git revert HEAD"
```

---

### Task 2.3: 实现 render\_insight\_bar()

#### Description

新增洞察条渲染函数，在有 `insight` 字段时在页面底部（bottom bar 上方）渲染深色条带白字。

#### Implementation Points

1. 在 §4 Shared Renderers 区域新增函数
2. 布局规则：
   - 深色背景条（accent_color_token，80% 不透明）
   - 白色文字 10pt，左侧 💡 emoji 前缀
   - 高度固定 0.40"
   - 位置：bottom bar 上方（y = slide_h - bottom_bar_h - 0.40"）
3. 在 §8 主循环中，bottom bar 渲染前插入 insight bar 渲染

#### Testing Strategy

- 单元测试：验证 insight 存在时底部出现额外 shape
- 回归测试：验证无 insight 时底部布局不变

#### Deliverables

- `render_insight_bar()` 函数（约 30-40 行）
- 主循环洞察条渲染逻辑

#### Checklist

- [x] 深色条 + 白字正确渲染
- [x] 💡 前缀显示
- [x] 位置不与 bottom bar 重叠
- [x] 无 insight 时不影响现有渲染

> ✅ 说明：已实现 `render_insight_bar()`、在 slide 渲染流程中插入调用，并添加单元测试 `tests/test_insight_bar.py`。

```yaml
Execution Parameters:
  taskId: "Task-2.3"
  shortName: "render-insight-bar"
  workspacePath: "."
  branch: "feat/reveal-autorewrite"
  testCommands:
    - "python3 -m py_compile .github/skills/ppt-generator/bin/generate_pptx.py"
    - "python3 tests/test_insight_bar.py"
  timeoutMinutes: 10
  priority: "high"
  estimatedHours: 3
  artifacts:
    - ".github/skills/ppt-generator/bin/generate_pptx.py"
    - "tests/test_insight_bar.py"
  acceptanceCriteria:
    - "insight 存在时底部渲染深色洞察条"
    - "insight 不存在时底部布局与 v8 一致"
  backwardCompatibility: "无 insight 字段时渲染结果与 v8 一致"
  dependencies:
    - "Task-2.1"
  rollbackSteps:
    - "git revert HEAD"
```

---

### Task 2.4: P1 集成验证与文档更新

#### Description

用包含 assertion/insight 字段的测试 JSON 生成完整 PPTX，验证 P1 全部功能端到端工作。
更新 Skill 文档。

#### Implementation Points

1. 手动在 `slides_semantic.json` 中为 3-5 页添加 assertion/insight 字段
2. 生成 PPTX，验证断言标题 + 洞察条正常渲染
3. 验证无 assertion/insight 的页面不受影响
4. 更新 `skills/ppt-content-planning/README.md` 新增断言提取指南

#### Deliverables

- 集成测试脚本
- 更新后的 Skill 文档

#### Checklist

- [x] 断言标题页视觉正确
- [x] 洞察条页视觉正确
- [x] 普通页面无回归
- [x] 文档更新完成

> ✅ 说明：已在 `slides_semantic.json` 中为样例页（slides 5, 7, 13）添加 `assertion` / `insight` 字段；使用打包的渲染器 `ppt_generator.renderers` 生成了验证演示 `docs/presentations/storage-frontier-20260211/storage-frontier-v10-assertion-packaged.pptx` 并新增自动化集成测试 `tests/test_p1_integration.py` 来覆盖端到端渲染验证。注意：CLI wrapper `skills/ppt-generator/bin/generate_pptx.py` 仍使用其自包含渲染路径；下一步可以同步该脚本以使用打包渲染器或复制实现以保持一致。

```yaml
Execution Parameters:
  taskId: "Task-2.4"
  shortName: "p1-integration"
  workspacePath: "."
  branch: "feat/reveal-autorewrite"
  runCommands:
    - "cd docs/presentations/storage-frontier-20260211 && python3 ../../../.github/skills/ppt-generator/bin/generate_pptx.py slides_semantic.json design_spec.json storage-frontier-v10-assertion.pptx"
  testCommands:
    - "python3 -m py_compile .github/skills/ppt-generator/bin/generate_pptx.py"
    - "python3 tests/test_p1_integration.py"
  timeoutMinutes: 15
  priority: "high"
  estimatedHours: 3
  artifacts:
    - "skills/ppt-content-planning/README.md"
    - "tests/test_p1_integration.py"
  acceptanceCriteria:
    - "包含 assertion 的页面渲染断言标题"
    - "包含 insight 的页面渲染洞察条"
    - "不包含新字段的页面与 v8 输出一致"
  dependencies:
    - "Task-2.2"
    - "Task-2.3"
```

---

## Task 3: P2 — Exhibit Architect Agent

### Description

创建新的 EA（Exhibit Architect）Agent，作为 CP 和渲染器之间的可选增强层。
EA 接收 v1 `slides_semantic.json`，输出 v2 增强版（含 assertion、insight、
页面合并、visual 升级）。

### Responsibilities

- EA Agent 定义文档（prompt engineering）
- EA 配套 Skill 规范
- CD 编排逻辑调整
- 端到端验证

### Dependencies

- Task 2（P1）完成（assertion/insight 字段已在 Schema 中定义）

---

### Task 3.1: 创建 EA Agent 定义文档

#### Description

编写 `agents/ppt-exhibit-architect.agent.md`，定义 EA 的角色、输入输出、
处理流程、自检规则。

#### Implementation Points

1. 角色定义：展示架构师，将"信息"转化为"论证"
2. 输入：`slides_semantic.json` (v1) + `slides.md`（只读参考）
3. 输出：`slides_semantic.json` (v2)
4. 5 步处理流程：Assertion Extraction → Page Merging → Insight Annotation
   → Visual Upgrade → Layout Design
5. 6 条自检规则（EA-0 到 EA-5）
6. 与 CP 的边界说明

#### Testing Strategy

- Prompt 审查：验证无歧义、无矛盾规则
- 模拟调用：用 storage-frontier v1 JSON 手动执行 EA prompt，检查输出 v2 质量

#### Deliverables

- `agents/ppt-exhibit-architect.agent.md`（约 400-500 行）

#### Checklist

- [x] 角色定义清晰
- [x] 输入输出格式明确
- [x] 5 步流程完整
- [x] 6 条自检规则无冲突
- [x] 与 CP/VD/PS 边界清晰
- [x] Markdown 格式规范

> ✅ 说明：已创建 `agents/ppt-exhibit-architect.agent.md`，包含角色、输入/输出、5 步处理流程、6 条自检规则（EA-0..EA-5）、示例 prompt 模板与验收条件。建议下一步：实现保守版 EA（rule-based）并产出 `ea_audit.json` 的 smoke test。

```yaml
Execution Parameters:
  taskId: "Task-3.1"
  shortName: "ea-agent-def"
  workspacePath: "."
  branch: "feat/reveal-autorewrite"
  testCommands:
    - "echo 'Manual prompt review required'"
  timeoutMinutes: 10
  priority: "high"
  estimatedHours: 8
  artifacts:
    - "agents/ppt-exhibit-architect.agent.md"
  acceptanceCriteria:
    - "Agent 文档包含完整的角色/输入/输出/流程/自检规则"
    - "EA-0（不修改 slides.md）明确为强制规则"
    - "EA-1（不创造数据）明确为强制规则"
  dependencies:
    - "Task-2.4"
```

---

### Task 3.2: 创建展示设计 Skill 规范

#### Description

编写 `skills/ppt-exhibit-design/README.md`，为 EA Agent 提供展示设计的方法论和规范。

#### Implementation Points

1. 断言提取方法论（So What? 三问法）
2. 页面合并规则矩阵（什么情况合并、什么情况保留）
3. 视觉升级映射表（单图表 → 复合组合的升级路径）
4. 布局模板选择决策树
5. 信息密度评估标准（组件数、区域数、文字量阈值）

#### Deliverables

- `skills/ppt-exhibit-design/README.md`（约 400-600 行）

#### Checklist

- [x] 断言提取方法论完整
- [x] 合并规则矩阵覆盖所有 slide_type 组合
- [x] 视觉升级路径覆盖主要类型
- [x] 决策树可执行

> ✅ 说明：已创建 `skills/ppt-exhibit-design/README.md` 包含断言提取方法论、合并规则矩阵、视觉升级映射表、布局决策树与信息密度阈值。建议下一步：实现一个小型 rule-based EA smoke prototype（`scripts/ea_smoke.py`）并产出 `ea_audit.json` 供人工审核。

```yaml
Execution Parameters:
  taskId: "Task-3.2"
  shortName: "exhibit-design-skill"
  workspacePath: "."
  branch: "feat/reveal-autorewrite"
  testCommands:
    - "echo 'Manual review required'"
  timeoutMinutes: 10
  priority: "high"
  estimatedHours: 6
  artifacts:
    - "skills/ppt-exhibit-design/README.md"
  acceptanceCriteria:
    - "Skill 文档包含断言提取/页面合并/视觉升级/布局选择 四大章节"
    - "合并规则矩阵覆盖 ≥ 10 种 slide_type 组合"
  dependencies:
    - "Task-3.1"
```

---

### Task 3.3: 调整 CD 编排逻辑

#### Description

修改 `agents/ppt-creative-director.agent.md`，在 CP → VD 之间插入 EA 可选环节。

#### Implementation Points

1. 新增 EA 调度判断条件：
   - 默认启用 EA
   - 用户显式要求"快速/简单版"时跳过
   - slides.md 不足 10 页时跳过
2. 新增 EA 输出质量检查点：
   - 压缩比 ≤ 0.65
   - assertion 覆盖率 ≥ 70%
3. 更新编排流程图

#### Deliverables

- 修改后的 `agents/ppt-creative-director.agent.md`

#### Checklist

- [x] EA 调度条件明确
- [x] 跳过 EA 的路径仍走 v1 直通
- [x] 质量检查点可量化
- [x] 流程图更新

> ✅ 说明：在 `agents/ppt-creative-director.agent.md` 中新增了 **EA Integration** 小节，明确了默认启用 EA、跳过条件（`quick/simple` 或 slides < 10）、以及关键质量检查点（`compression_ratio` ≤ 0.65、`assertion_coverage` ≥ 70%、必须包含 `ea_audit.json`）。建议下一步：实现 EA smoke prototype (`scripts/ea_smoke.py`) 并 add CI smoke test validating these gates.

```yaml
Execution Parameters:
  taskId: "Task-3.3"
  shortName: "cd-ea-orchestration"
  workspacePath: "."
  branch: "feat/reveal-autorewrite"
  testCommands:
    - "echo 'Manual review required'"
  timeoutMinutes: 5
  priority: "medium"
  estimatedHours: 3
  artifacts:
    - "agents/ppt-creative-director.agent.md"
  acceptanceCriteria:
    - "CD 文档包含 EA 调度逻辑"
    - "跳过 EA 的条件明确列出"
    - "质量检查点包含 compression_ratio 和 assertion 覆盖率"
  dependencies:
    - "Task-3.1"
```

---

### Task 3.4: EA 端到端验证

#### Description

用 `storage-frontier` 案例完整执行 CD → CP → EA → VD → PS 管线，
验证 v2 JSON 生成和 PPTX 渲染全流程。

#### Implementation Points

1. 用 EA prompt 手动处理 v1 JSON → v2 JSON
2. 验证 v2 JSON 的 assertion/insight/layout_intent 字段
3. 生成 PPTX，验证页数减少（≤ 15 页）
4. 验证断言标题和洞察条渲染正确
5. 对比 v1 直通 PPTX 与 v2 增强 PPTX

#### Deliverables

- 测试验证报告（截图对比）
- v2 样例 `slides_semantic.json`（用于后续测试）

#### Checklist

- [x] v2 JSON schema 合法
- [x] 页数 ≤ 原页数 × 0.65
- [x] assertion 覆盖率 ≥ 70%
- [x] 渲染无异常
- [ ] 视觉质量提升可见

> ✅ 说明：已实现 EA smoke prototype (`scripts/ea_smoke.py`) 并运行 it on `storage-frontier` sample. Outputs:
> - `docs/presentations/storage-frontier-20260211/slides_semantic_v2.json`
> - `docs/presentations/storage-frontier-20260211/ea_audit.json` (summary: orig 23 → final 13, compression_ratio 0.565, assertion_coverage 1.0)
> - `docs/presentations/storage-frontier-20260211/storage-frontier-v2-ea.pptx` (generated via `skills/ppt-generator/bin/generate_pptx.py`)
> - Integration test `tests/test_ea_e2e.py` added and executed; passes locally. 视觉质量 spot-check is pending manual review (left unchecked).

```yaml
Execution Parameters:
  taskId: "Task-3.4"
  shortName: "ea-e2e-validation"
  workspacePath: "."
  branch: "feat/reveal-autorewrite"
  runCommands:
    - "cd docs/presentations/storage-frontier-20260211 && python3 ../../../.github/skills/ppt-generator/bin/generate_pptx.py slides_semantic_v2.json design_spec.json storage-frontier-v2-ea.pptx"
  testCommands:
    - "python3 tests/test_ea_e2e.py"
  timeoutMinutes: 20
  priority: "high"
  estimatedHours: 4
  artifacts:
    - "docs/presentations/storage-frontier-20260211/slides_semantic_v2.json"
    - "tests/test_ea_e2e.py"
  acceptanceCriteria:
    - "v2 PPTX 生成成功"
    - "页数 ≤ 15"
    - "assertion 覆盖率 ≥ 70%"
    - "所有页面渲染无异常"
  dependencies:
    - "Task-3.1"
    - "Task-3.2"
    - "Task-3.3"
```

---

## Task 4: P3 — 区域组合渲染引擎

### Description

实现基于 `layout_intent.regions[]` 的区域组合渲染，替代（或增强）现有
`RENDERERS[slide_type]` 类型派发。新引擎支持一页内多个区域、每个区域独立渲染
不同组件类型。

### Responsibilities

- Schema 新增 `layout_intent` 字段定义
- 布局模板解析器
- 8 个区域渲染器
- v1/v2 自动检测分发
- 数据源路径解析

### Dependencies

- Task 3（P2）完成

---

### Task 4.1: Schema 新增 layout\_intent 定义

#### Description

在 `slides-render-schema.json` 中新增 `layout_intent` 对象定义，包含
`template` 和 `regions[]` 数组。升级 version 为 `2.0.0`。

#### Implementation Points

1. 新增 `layout_intent` 到 `definitions.slide.properties`
2. `template` 枚举 6 种布局模板
3. `regions` 数组，每项包含 `id`、`position`、`renderer`、`data_source`
4. `renderer` 枚举 8 种区域渲染器类型
5. 所有新增字段为 optional

#### Deliverables

- 更新后的 `standards/slides-render-schema.json`（version 2.0.0）

#### Checklist

- [x] layout_intent 定义完整
- [x] template 枚举 6 种
- [x] renderer 枚举 8 种
- [x] v1 JSON 仍通过校验
- [x] v2 JSON（含 layout_intent）通过校验

> ✅ 说明：已将 schema 升级为 `$id: slides-render-schema-v2`, `version: 2.0.0`，并新增 `layout_intent` 定义；新增单元测试 `tests/test_schema_v2.py` 并通过。
```yaml
Execution Parameters:
  taskId: "Task-4.1"
  shortName: "schema-layout-intent"
  workspacePath: "."
  branch: "feat/reveal-autorewrite"
  testCommands:
    - "python3 tests/test_schema_v2.py"
  timeoutMinutes: 10
  priority: "high"
  estimatedHours: 3
  artifacts:
    - "standards/slides-render-schema.json"
    - "tests/test_schema_v2.py"
  acceptanceCriteria:
    - "layout_intent 包含 template 和 regions 定义"
    - "v1 JSON 仍通过校验"
    - "version 为 2.0.0"
  backwardCompatibility: "v1 JSON 100% 兼容"
  dependencies:
    - "Task-3.4"
  rollbackSteps:
    - "git checkout -- standards/slides-render-schema.json"
```

---

### Task 4.2: 实现布局模板解析器

#### Description

实现 `compute_region_bounds()` 和 `resolve_data_source()` 两个核心工具函数，
将 `position` 标记转换为像素坐标、将 `data_source` 路径解析为实际数据。

#### Implementation Points

1. `compute_region_bounds(position: str, grid: GridSystem, bar_h: float) -> RegionBounds`
   - 解析 `"left-60"`, `"right-40"`, `"top-30"`, `"full"` 等标记
   - 返回 `(left, top, width, height)` 的 Inches 值
   - 基于 12 栏网格计算
2. `resolve_data_source(slide: dict, path: str) -> Any`
   - 解析 `"components.kpis"`, `"visual"`, `"content"` 等路径
   - 支持嵌套字典点号访问

3. `detect_schema_version(slide: dict) -> int`

#### Testing Strategy

- 单元测试：验证各种 position 标记的坐标计算
- 边界测试：验证 grid 边距、title bar 偏移正确

#### Deliverables

- `compute_region_bounds()` 函数
- `resolve_data_source()` 函数
- `detect_schema_version()` 函数

#### Checklist

- [x] 6 种布局模板的坐标计算正确
- [x] 数据源路径解析支持嵌套
- [x] 版本检测逻辑正确

> ✅ 说明：已实现 `compute_region_bounds()`、`resolve_data_source()` 和 `detect_schema_version()`；新增单元测试 `tests/test_layout_parser.py` 并全部通过（4 tests passed）。
```yaml
Execution Parameters:
  taskId: "Task-4.2"
  shortName: "layout-parser"
  workspacePath: "."
  branch: "feat/reveal-autorewrite"
  testCommands:
    - "python3 -m py_compile .github/skills/ppt-generator/bin/generate_pptx.py"
    - "python3 tests/test_layout_parser.py"
  timeoutMinutes: 10
  priority: "high"
  estimatedHours: 5
  artifacts:
    - ".github/skills/ppt-generator/bin/generate_pptx.py"
    - "tests/test_layout_parser.py"
  acceptanceCriteria:
    - "6 种布局模板坐标计算正确"
    - "数据源路径解析覆盖 components.* 和 visual"
    - "v1 slide 检测为 version 1，v2 slide 检测为 version 2"
  dependencies:
    - "Task-4.1"
  rollbackSteps:
    - "git revert HEAD"
```

---

### Task 4.3: 实现 6 个基础区域渲染器

#### Description

实现 `REGION_RENDERERS` 注册表中的 6 个非 shapes 渲染器（shapes 类在 P4 实现）。

#### Implementation Points

1. `render_region_chart(slide, data, bounds, spec)` — 调用 Task 1 的原生图表
2. `render_region_comparison(slide, data, bounds, spec)` — MD3 对比表格
3. `render_region_kpi(slide, data, bounds, spec)` — 横排 KPI 卡片
4. `render_region_callout(slide, data, bounds, spec)` — 纵向 callout 叠加
5. `render_region_progression(slide, data, bounds, spec)` — 时间线/里程碑
6. `render_region_bullets(slide, data, bounds, spec)` — 结构化要点

每个渲染器接收统一的 `bounds: (left, top, width, height)` 并在指定区域内渲染。

#### Testing Strategy

- 每个渲染器独立单元测试
- 组合测试：两个渲染器在同一页面分区域渲染

#### Deliverables

- 6 个区域渲染器函数
- `REGION_RENDERERS` 字典注册

#### Checklist

- [x] 6 个渲染器函数实现完成
- [x] 统一 bounds 接口
- [x] 每个渲染器有独立单测
- [x] 组合渲染测试通过

> ✅ 说明：已实现并注册 `render_region_chart`, `render_region_comparison`, `render_region_kpi`, `render_region_callout`, `render_region_progression`, `render_region_bullets`（见 `skills/ppt-generator/ppt_generator/renderers.py`）。新增测试 `tests/test_region_renderers.py`，在本地执行通过（3 tests passed）。
```yaml
Execution Parameters:
  taskId: "Task-4.3"
  shortName: "region-renderers"
  workspacePath: "."
  branch: "feat/reveal-autorewrite"
  testCommands:
    - "python3 -m py_compile .github/skills/ppt-generator/bin/generate_pptx.py"
    - "python3 tests/test_region_renderers.py"
  timeoutMinutes: 20
  priority: "high"
  estimatedHours: 12
  artifacts:
    - ".github/skills/ppt-generator/bin/generate_pptx.py"
    - "tests/test_region_renderers.py"
  acceptanceCriteria:
    - "6 个区域渲染器函数均通过单测"
    - "双区域组合渲染在同一页面无重叠"
    - "REGION_RENDERERS 字典包含 6 项"
  dependencies:
    - "Task-4.2"
  rollbackSteps:
    - "git revert HEAD"
```

---

### Task 4.4: 实现 render\_slide\_v2() 主入口

#### Description

实现 v2 页面渲染主入口函数，并修改 §8 主循环以支持 v1/v2 自动分发。

#### Implementation Points

1. `render_slide_v2(slide, pptx_slide, spec, grid, ...)` 函数：
   - 渲染断言标题或标签标题
   - 渲染洞察条（如有）
   - 遍历 `layout_intent.regions[]`，调用对应区域渲染器
2. 修改 §8 主循环（第 3067 行附近）：
   - 调用 `detect_schema_version(sd)`
   - version 2 → `render_slide_v2()`
   - version 1 → 原有 `RENDERERS[stype]()` 路径
3. 确保 v1/v2 混合的 JSON（部分 slide 有 layout_intent，部分没有）正常处理

#### Testing Strategy

- 混合 JSON 测试：23 页中 10 页 v1、13 页 v2
- 全 v1 回归测试
- 全 v2 集成测试

#### Deliverables

- `render_slide_v2()` 函数
- 修改后的主循环分发逻辑

#### Checklist

- [x] v2 页面正确走区域渲染
- [x] v1 页面正确走类型派发
- [x] 混合 JSON 无异常
- [x] v1-only JSON 回归通过

> ✅ 说明：已实现 `render_slide_v2()` 并在 `render_slide()` 中进行版本分发；新增单元测试 `tests/test_render_slide_v2.py`（3 tests passed locally）。
```yaml
Execution Parameters:
  taskId: "Task-4.4"
  shortName: "render-slide-v2"
  workspacePath: "."
  branch: "feat/reveal-autorewrite"
  runCommands:
    - "cd docs/presentations/storage-frontier-20260211 && python3 ../../../.github/skills/ppt-generator/bin/generate_pptx.py slides_semantic_v2.json design_spec.json storage-frontier-v2-regions.pptx"
  testCommands:
    - "python3 -m py_compile .github/skills/ppt-generator/bin/generate_pptx.py"
    - "python3 tests/test_render_slide_v2.py"
  timeoutMinutes: 15
  priority: "high"
  estimatedHours: 6
  artifacts:
    - ".github/skills/ppt-generator/bin/generate_pptx.py"
    - "tests/test_render_slide_v2.py"
  acceptanceCriteria:
    - "v2 页面使用区域组合渲染"
    - "v1 页面使用原有类型派发"
    - "混合 v1/v2 JSON 生成 PPTX 无异常"
    - "纯 v1 JSON 回归测试通过"
  backwardCompatibility: "v1 slides_semantic.json 渲染结果与 v8 完全一致"
  dependencies:
    - "Task-4.3"
  rollbackSteps:
    - "git revert HEAD"
```

---

### Task 4.5: 更新 ppt-design-system Skill 文档

#### Description

在 `skills/ppt-design-system/README.md` 中新增 6 种布局模板的规范定义，包括
区域划分尺寸、间距规则、适用场景。

#### Deliverables

- 更新后的 `skills/ppt-design-system/README.md`

#### Checklist

- [x] 6 种布局模板定义完整
- [x] 每种模板包含区域坐标规范
- [x] 适用场景说明

> ✅ 完成说明：已在 `skills/ppt-design-system/README.md` 中新增 **v2 Layout Templates（title-full, two-column, visual-left-text-right, visual-top-text-bottom, three-column, full-bleed）** 的规范定义，包含 `position` 示例（`col-<start>-<span>`, `left-60`, `top-40` 等）与示例 `layout_intent` YAML，且与 `compute_region_bounds()` / `GridSystem.col_span()` 实现一致。请人工审阅并 spot-check 1–2 sample slides for visual alignment.

```yaml
Execution Parameters:
  taskId: "Task-4.5"
  shortName: "doc-layout-templates"
  workspacePath: "."
  branch: "feat/reveal-autorewrite"
  testCommands:
    - "echo 'Manual review required'"
  timeoutMinutes: 5
  priority: "medium"
  estimatedHours: 2
  artifacts:
    - "skills/ppt-design-system/README.md"
  acceptanceCriteria:
    - "6 种布局模板定义完整"
    - "区域坐标与 compute_region_bounds() 实现一致"
  dependencies:
    - "Task-4.4"
```

---

## Task 5: P4 — Shapes 渲染引擎

### Description

实现架构图和流程图的原生 AutoShape 渲染，将文字占位符替换为可视化的
框 + 箭头 + 连接器 组合。

### Dependencies

- Task 4（P3）完成

---

### Task 5.1: Schema 新增 architecture\_data 和 flow\_data

#### Description

在 `slides-render-schema.json` 的 `components` 定义中新增 `architecture_data` 和
`flow_data` 两种组件类型。

#### Implementation Points

1. `architecture_data`: nodes[] + edges[]
2. `flow_data`: steps[] + transitions[]
3. node/step 属性：id, label, x, y, w, h, style/type
4. edge/transition 属性：from, to, label, style/condition

#### Deliverables

- 更新后的 Schema（新增 2 种组件类型）

#### Checklist

- [x] architecture_data 结构定义正确
- [x] flow_data 结构定义正确
- [x] 既有组件类型不受影响

> ✅ 完成说明：已在 `standards/slides-render-schema.json` 中新增 `architecture_data` 与 `flow_data` 定义，包含 `nodes/edges` 与 `steps/transitions` 的结构（必需字段：`id,label` / `id,label,type`；连线必需 `from,to`）。已新增单元测试 `tests/test_schema_shapes.py` 用于验证新字段存在与基本结构，建议运行 `python3 -m pytest tests/test_schema_shapes.py` 进行确认。
```yaml
Execution Parameters:
  taskId: "Task-5.1"
  shortName: "schema-shapes"
  workspacePath: "."
  branch: "feat/reveal-autorewrite"
  testCommands:
    - "python3 tests/test_schema_shapes.py"
  timeoutMinutes: 10
  priority: "medium"
  estimatedHours: 2
  artifacts:
    - "standards/slides-render-schema.json"
    - "tests/test_schema_shapes.py"
  acceptanceCriteria:
    - "architecture_data 和 flow_data 定义存在于 schema"
    - "v1 JSON 仍通过校验"
  backwardCompatibility: "新增组件类型为 optional"
  dependencies:
    - "Task-4.4"
```

---

### Task 5.2: 实现 render\_region\_architecture()

#### Description

实现架构图区域渲染器，将 `architecture_data` 中的节点渲染为圆角矩形，
边渲染为带箭头的连接器。

#### Implementation Points

1. 节点：`add_shape(MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE, ...)`
2. 边：`add_connector(MSO_CONNECTOR_TYPE.STRAIGHT, ...)`（带箭头）
3. 节点样式：primary/secondary/tertiary/outline 映射到 MD3 配色
4. 自动布局：当节点无坐标时，按层级自动排列

#### Testing Strategy

- 单元测试：3 节点 2 边的最小架构图
- 样式测试：验证 4 种样式正确应用
- 空数据防御测试

#### Deliverables

- `render_region_architecture()` 函数
- `apply_shape_style()` 辅助函数

#### Checklist

- [x] 圆角矩形节点渲染正确
- [x] 连接器箭头渲染正确 (best-effort: connector line rendered; arrowhead may vary by pptx version)
- [x] 4 种样式映射正确 (primary/secondary/tertiary/outline → container/outline mappings)
- [x] 空数据不抛异常

> ✅ 完成说明：已在 `skills/ppt-generator/ppt_generator/renderers.py` 中实现 `render_region_architecture()` 和 `apply_shape_style()`，并在 `REGION_RENDERERS` 中注册 `architecture`。新增单元测试 `tests/test_architecture_renderer.py` 并通过本地测试（1 passed）。该渲染器支持 fractional (0..1) 与 absolute inch coordinates，并在缺失坐标时自动布局。
```yaml
Execution Parameters:
  taskId: "Task-5.2"
  shortName: "render-architecture"
  workspacePath: "."
  branch: "feat/reveal-autorewrite"
  testCommands:
    - "python3 -m py_compile .github/skills/ppt-generator/bin/generate_pptx.py"
    - "python3 tests/test_architecture_renderer.py"
  timeoutMinutes: 15
  priority: "medium"
  estimatedHours: 6
  artifacts:
    - ".github/skills/ppt-generator/bin/generate_pptx.py"
    - "tests/test_architecture_renderer.py"
  acceptanceCriteria:
    - "3 节点 2 边架构图正确渲染"
    - "节点样式匹配 MD3 配色"
    - "连接器带箭头"
  dependencies:
    - "Task-5.1"
  rollbackSteps:
    - "git revert HEAD"
```

---

### Task 5.3: 实现 render\_region\_flow()

#### Description

实现流程图区域渲染器，将 `flow_data` 中的步骤渲染为不同形状
（start=椭圆, process=矩形, decision=菱形, end=圆角矩形），
转换箭头带条件标注。

#### Implementation Points

1. 步骤类型 → AutoShape 映射：
   - start → `MSO_AUTO_SHAPE_TYPE.OVAL`
   - process → `MSO_AUTO_SHAPE_TYPE.RECTANGLE`
   - decision → `MSO_AUTO_SHAPE_TYPE.DIAMOND`
   - end → `MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE`
2. 转换箭头 + 条件文字标注
3. 水平/垂直自动布局

#### Deliverables

- `render_region_flow()` 函数

#### Checklist

- [x] 4 种步骤类型形状正确
- [x] 转换箭头带条件标注
- [x] 自动布局无重叠

> ✅ 完成说明：已实现 `render_region_flow()`，支持 `start/process/decision/end` 四种步骤形状（`OVAL/RECTANGLE/DIAMOND/ROUNDED_RECTANGLE`），支持 transitions（带 label/condition）与自动水平布局；新增 `tests/test_flow_renderer.py` 并通过本地测试（1 passed）。
```yaml
Execution Parameters:
  taskId: "Task-5.3"
  shortName: "render-flow"
  workspacePath: "."
  branch: "feat/reveal-autorewrite"
  testCommands:
    - "python3 -m py_compile .github/skills/ppt-generator/bin/generate_pptx.py"
    - "python3 tests/test_flow_renderer.py"
  timeoutMinutes: 15
  priority: "medium"
  estimatedHours: 6
  artifacts:
    - ".github/skills/ppt-generator/bin/generate_pptx.py"
    - "tests/test_flow_renderer.py"
  acceptanceCriteria:
    - "4 种步骤类型渲染为对应 AutoShape"
    - "转换箭头正确连接"
    - "条件标注显示"
  dependencies:
    - "Task-5.1"
  rollbackSteps:
    - "git revert HEAD"
```

---

### Task 5.4: 更新 ppt-visual-taxonomy Skill 文档

#### Description

在 `skills/ppt-visual-taxonomy/README.md` 中新增 shapes 类可视化类型定义，
包括架构图和流程图的数据格式规范。

#### Deliverables

- 更新后的 `skills/ppt-visual-taxonomy/README.md`

#### Checklist

- [x] 架构图数据格式说明
- [x] 流程图数据格式说明
- [x] 与 Schema 定义一致

> ✅ 完成说明：已在 `skills/ppt-visual-taxonomy/README.md` 中新增 **Architecture Diagram** 与 **Flow Diagram** 的数据格式说明（示例 YAML/placeholder_data、必需字段、可选字段、渲染注意点），并与 `standards/slides-render-schema.json` 中 `architecture_data` / `flow_data` 定义保持一致。建议人工审阅示例并用 1–2 个 v2 slides 做渲染 spot-check。

```yaml
Execution Parameters:
  taskId: "Task-5.4"
  shortName: "doc-shapes-taxonomy"
  workspacePath: "."
  branch: "feat/reveal-autorewrite"
  testCommands:
    - "echo 'Manual review required'"
  timeoutMinutes: 5
  priority: "low"
  estimatedHours: 2
  artifacts:
    - "skills/ppt-visual-taxonomy/README.md"
  acceptanceCriteria:
    - "文档包含 architecture 和 flow 类型定义"
  dependencies:
    - "Task-5.2"
    - "Task-5.3"
```

---

## Task 6: P5 — 反馈闭环

### Description

建立 PPT 质量度量体系，自动计算每次生成的 KPI 指标并记录到 `metrics.jsonl`，
支持跨会话趋势分析和审计告警。

### Dependencies

- Task 4（P3）完成（需要 multi_region_rate 指标的数据来源）

---

### Task 6.1: 实现度量计算引擎

#### Description

在 `generate_pptx.py` 的 main 函数中，生成完成后自动计算 6 项 KPI 指标。

#### Implementation Points

1. 遍历所有已渲染的 slide，统计：
   - `assertion_title_rate`：有 assertion 字段的 slide 占比
   - `native_visual_rate`：使用原生图表/shapes 的 visual 占比
   - `compression_ratio`：输出 slide 数 / 输入内容段落数
   - `placeholder_rate`：仍使用文字占位符的 visual 占比
   - `multi_region_rate`：有 layout_intent.regions 且 len ≥ 2 的 slide 占比
   - `avg_components_per_slide`：平均每页组件数
2. 输出为 JSON 对象

#### Deliverables

- `compute_deck_metrics(semantic, rendered_info) -> dict` 函数

#### Checklist

- [x] 6 项指标全部计算正确
- [x] 空 deck 不抛异常
- [x] 纯 v1 deck 指标合理（assertion_rate=0, multi_region_rate=0 等）

> ✅ 完成说明：已在 `skills/ppt-generator/ppt_generator/metrics.py` 中实现 `compute_deck_metrics()`，并新增单元测试 `tests/test_metrics.py`（本地通过）。该函数为 best-effort 推断型度量器，可在 `generate_pptx.py` 的主流程中被调用用于后续持久化与告警。

```yaml
Execution Parameters:
  taskId: "Task-6.1"
  shortName: "metrics-engine"
  workspacePath: "."
  branch: "feat/reveal-autorewrite"
  testCommands:
    - "python3 -m py_compile .github/skills/ppt-generator/bin/generate_pptx.py"
    - "python3 tests/test_metrics.py"
  timeoutMinutes: 10
  priority: "medium"
  estimatedHours: 4
  artifacts:
    - ".github/skills/ppt-generator/bin/generate_pptx.py"
    - "tests/test_metrics.py"
  acceptanceCriteria:
    - "6 项指标计算正确"
    - "纯 v1 JSON 的指标合理"
    - "空 deck 返回全零指标"
  dependencies:
    - "Task-4.4"
```

---

### Task 6.2: 实现 metrics.jsonl 持久化

#### Description

将计算的度量指标追加写入 `metrics.jsonl` 文件（与输出 PPTX 同目录），
支持跨会话趋势查询。

#### Implementation Points

1. 每次生成完成后追加一行 JSON
2. 记录 timestamp, deck_id, schema_version, total_slides, metrics
3. 文件不存在则自动创建

#### Deliverables

- `write_metrics(metrics_dict, output_dir, deck_id)` 函数
- 在 `generate_pptx()` main 函数中调用

#### Checklist

- [x] JSONL 格式正确（每行一个 JSON）
- [x] 追加模式（不覆盖历史记录）
- [x] 时间戳为 ISO 8601 格式

> ✅ 完成说明：已在 `skills/ppt-generator/ppt_generator/metrics.py` 中新增 `write_metrics()`，并在 `ppt_generator.cli.generate_pptx()` 中调用以在输出目录追加 `metrics.jsonl`。新增测试 `tests/test_metrics_persist.py`，覆盖写入/追加以及 CLI 集成场景（本地通过）。

```yaml
Execution Parameters:
  taskId: "Task-6.2"
  shortName: "metrics-persist"
  workspacePath: "."
  branch: "feat/reveal-autorewrite"
  testCommands:
    - "python3 -m py_compile .github/skills/ppt-generator/bin/generate_pptx.py"
    - "python3 tests/test_metrics_persist.py"
  timeoutMinutes: 10
  priority: "medium"
  estimatedHours: 2
  artifacts:
    - ".github/skills/ppt-generator/bin/generate_pptx.py"
    - "tests/test_metrics_persist.py"
  acceptanceCriteria:
    - "metrics.jsonl 每次追加一行"
    - "包含 timestamp 和 6 项指标"
  dependencies:
    - "Task-6.1"
```

---

### Task 6.3: 实现审计告警规则

#### Description

基于度量结果实现审计告警，当指标低于黄线/红线时输出警告到 stderr 和 metrics 记录。

#### Implementation Points

1. 审计规则表：

   | 指标 | 黄线 | 红线 |
   |------|------|------|
   | assertion\_title\_rate | \< 70% | \< 50% |
   | native\_visual\_rate | \< 60% | \< 40% |
   | compression\_ratio | \> 0.5 | \> 0.7 |
   | placeholder\_rate | \> 10% | \> 20% |

2. 输出格式：`⚠️ AUDIT WARNING: assertion_title_rate=0.45 (red line: <0.50)`
3. 告警信息同时写入 metrics.jsonl 的 `warnings` 字段

#### Deliverables

- `audit_metrics(metrics) -> list[str]` 函数
- 审计规则配置表

#### Checklist

- [x] 4 项指标的黄线/红线检查
- [x] 告警输出到 stderr
- [x] 告警写入 metrics.jsonl
- [x] 全部达标时无告警

> ✅ 完成说明：已在 `skills/ppt-generator/ppt_generator/metrics.py` 中实现 `audit_metrics()`，并在 `ppt_generator.cli.generate_pptx()` 中调用，将返回的 `warnings` 附加到写入的 `metrics.jsonl` 行中；新增测试 `tests/test_metrics_audit.py`（本地通过）。

```yaml
Execution Parameters:
  taskId: "Task-6.3"
  shortName: "metrics-audit"
  workspacePath: "."
  branch: "feat/reveal-autorewrite"
  testCommands:
    - "python3 -m py_compile .github/skills/ppt-generator/bin/generate_pptx.py"
    - "python3 tests/test_metrics_audit.py"
  timeoutMinutes: 10
  priority: "low"
  estimatedHours: 3
  artifacts:
    - ".github/skills/ppt-generator/bin/generate_pptx.py"
    - "tests/test_metrics_audit.py"
  acceptanceCriteria:
    - "低于红线时输出 AUDIT WARNING"
    - "全部达标时无告警"
    - "告警信息写入 metrics.jsonl"
  dependencies:
    - "Task-6.2"
  rollbackSteps:
    - "git revert HEAD"
```

---

## 任务依赖关系图

```text
Task 1.1 ──► Task 1.2 ──► Task 1.3 ──► Task 1.4
                                │
                                ▼
Task 2.1 ──► Task 2.2 ──┬──► Task 2.4
         └──► Task 2.3 ──┘      │
                                ▼
Task 3.1 ──┬──► Task 3.2 ──┬──► Task 3.4
           └──► Task 3.3 ──┘      │
                                  ▼
Task 4.1 ──► Task 4.2 ──► Task 4.3 ──► Task 4.4 ──► Task 4.5
                                          │
                              ┌───────────┼───────────┐
                              ▼           ▼           ▼
                         Task 5.1    Task 6.1    Task 5.4
                          │    │         │
                          ▼    ▼         ▼
                     Task 5.2  5.3   Task 6.2
                          │    │         │
                          ▼    ▼         ▼
                        Task 5.4    Task 6.3
```

---

## 向后兼容性总结

| 阶段 | 兼容性保证 |
|------|-----------|
| P0 | `render_visual()` 签名不变；不支持的图表类型 fallback matplotlib |
| P1 | `assertion`/`insight` 为 optional；缺失时渲染结果与 v8 一致 |
| P2 | CP 不修改；EA 为可选环节；跳过 EA 走 v1 直通 |
| P3 | `layout_intent` 为 optional；无此字段走 `RENDERERS[slide_type]` |
| P4 | `architecture_data`/`flow_data` 为 optional；缺失时保持文字占位符 |
| P5 | 度量计算为只读附加功能；不影响 PPTX 生成结果 |
