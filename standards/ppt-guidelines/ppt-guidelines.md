# PPT Guidelines — 设计规范（v1.0，正式化）

## 目的与适用范围
- 目的：为自动化与人工流程（PPT Specialist Agent、生成脚本、人工设计审查）提供一套**可验证、可量化**的幻灯片设计规范，提升可讲性、可审计性与视觉一致性。
- 适用范围：用于内部设计/技术评审、产品 Roadmap、架构/项目汇报等技术/产品类幻灯片；覆盖自动生成（agent）与人工编辑场景。

## 核心设计原则（必须遵循）
- 清晰（Clarity）: 每张幻灯片聚焦 1 个主要信息点。避免在一页堆砌多个复杂主题。
- 可讲（Speakability）: 每张页应包含或映射到 speaker notes（简短讲稿），便于复述与审查。
- 视觉优先（Visuals first）: 复杂信息应优先使用图表/图示表达，文本作为补充。
- 审计与决策（Auditability）: 关键决策需独立展现并包含决策理由与替代方案。

## 设计哲学参考（业界公认方法论）
本规范整合以下业界验证的设计哲学，并将其转化为可执行规则：

### 1. Presentation Zen（Garr Reynolds）— 简约主义与东方美学
**核心理念**：Less is more。用视觉与留白引导注意力，而非文字堆砌。  
**可执行规则**：
- 每页 bullets ≤ 3（推荐值，对应 `max_bullets_per_slide=3` strict mode）
- 页面留白 ≥ 60%（text density ≤ 40%）
- 单张幻灯片单一主题（可通过 semantic check 检测重复主题）
- 图像优先于文字（架构/流程/数据页必须含图示）

### 2. Assertion-Evidence Approach（Michael Alley，学术界标准）
**核心理念**：标题应为**论断句**（assertion），正文提供**证据**（evidence：图表、数据、引用）。  
**可执行规则**：
- 标题为完整句（检测：title 含动词或判断词，而非名词短语）
- 正文优先展示数据/图表，文字作为图注（检测：图表占比 > 50%）
- 每页至少 1 个可视化元素（图、表、公式）或明确数据引用
- 来源标注强制（学术/技术汇报场景启用 `require_citation=true`）

### 3. Guy Kawasaki 10/20/30 Rule（创投演讲标准）
**核心理念**：10 张幻灯片、20 分钟演讲、最小字号 30pt。  
**可执行规则**（适用于高管汇报/Pitch）：
- 总页数建议 ≤ 15（对于 20 分钟演讲，`max_slides_for_pitch=15`）
- 最小字号 ≥ 30pt（严格模式，覆盖默认 18pt，`min_body_font_pt=30`）
- 核心信息前置（Key Decisions / Problem / Solution 必须在前 3 页）
- 避免过渡页与装饰页（可检测：页面无实质内容且 text < 20 字符）

### 4. McKinsey/BCG Pyramid Principle（咨询行业标准）
**核心理念**：结论先行（SCQA：Situation-Complication-Question-Answer），逻辑严谨，数据支撑。  
**可执行规则**：
- 第 1-2 页必须包含 Executive Summary / Key Takeaways
- 每页标题符合"So What?"测试（可引入 NLP 检测标题是否传达结论）
- 图表必须自解释（标注完整：标题、坐标轴、单位、来源、时间窗口）
- 逻辑链完整（可选：检测页面间的逻辑跳跃，例如缺少铺垫）

### 5. Edward Tufte — 高信息密度与诚实可视化
**核心理念**：最大化数据-墨水比（data-ink ratio），移除图表垃圾（chartjunk），避免误导性可视化。  
**可执行规则**：
- 图表去除冗余装饰（3D 效果、阴影、渐变、不必要网格线）
- Y 轴必须从 0 开始（对于柱状图，检测 `y_axis_starts_at_zero=true`）
- 避免饼图超过 5 个分类（推荐 bar chart）
- 数据标注完整且精确（小数位数一致，避免四舍五入误差累积）

### 6. Takahashi Method（高桥流，日本极简风格）
**核心理念**：一张幻灯片一个关键词/短语，超大字号，快速切换。  
**适用场景**：快节奏演讲、TED-style talk、强调节奏感。  
**可执行规则**（可选，需显式启用 `style=takahashi`）：
- 每页仅 1 个词/短语（< 10 字符，`max_words_per_slide=3`）
- 字号 ≥ 80pt（`min_body_font_pt=80`）
- 禁止 bullets（`max_bullets_per_slide=0`）
- 幻灯片数量可大幅增加（用于快速节奏，豁免总页数限制）

### 7. Signal vs Noise（对比驱动设计）
**核心理念**：每个元素要么增强信号（关键信息），要么制造噪音（干扰）。移除一切噪音。  
**可执行规则**：
- 禁止无意义动画与过渡效果（检测：PowerPoint XML 内的 `<p:transition>` / `<p:animEffect>`）
- Logo/页眉/页脚仅在封面与结尾出现（中间页禁止，避免重复噪音）
- 配色限制为 3-5 种主色（检测：unique colors ≤ 5，排除黑白灰）
- 字体限制为 1-2 种（检测：unique font families ≤ 2）

### 哲学选择指南（按场景）
| 场景             | 推荐哲学                           | 关键规则启用                                |
| ---------------- | ---------------------------------- | ------------------------------------------- |
| ------           | ---------                          | -------------                               |
| 技术架构评审     | Assertion-Evidence + Tufte         | `require_citation=true`, 图表强制, Y 轴从 0 |
| 高管汇报 (Pitch) | 10/20/30 Rule + Pyramid            | `max_slides=15`, `min_font=30pt`, 结论前置  |
| 产品 Roadmap     | Presentation Zen + Signal vs Noise | `max_bullets=3`, 去除装饰, 视觉优先         |
| 学术报告         | Assertion-Evidence + Tufte         | 标题为论断, 数据完整标注, 来源强制          |
| 快节奏演讲       | Takahashi (可选)                   | `max_words=3`, `min_font=80pt`, 大量幻灯片  |

### 实施建议
- Agent 在生成前询问 `presentation_type`（默认 `technical-review`），自动应用对应哲学的规则组合。
- 人工设计时，在 `GUIDELINES.md` 顶部声明采用的哲学（例如：`philosophy: presentation-zen + pyramid`）。
- CI 检查时，根据声明的哲学加载对应的严格规则（例如：Takahashi 模式下 bullets > 0 视为错误）。

## 规则分级（MUST / SHOULD / MAY）
### MUST（强制）
- `Key Decisions`：如果文档总页数 > 5，则必须在**前 5 页**内包含一页或多页 `Key Decisions`（或 `关键决策`），每条决策含：决策内容、备选方案、评估标准、风险。可由 Agent 插入占位但需标注 `METADATA.auto_inserted=true`。
- 字体最小值：标题 >= **24pt**，正文字号 >= **18pt**（映射到 `min_title_font_pt` / `min_body_font_pt`）。
- 每页 bullets 不超过 **5 条**（`max_bullets_per_slide`）；推荐 3 条以内。
- 页面文字占比（Text density）不得超过 **40%**（`max_text_density_percent`）。
- Speaker notes 必须存在（`require_speaker_notes=true`），或在注释中提供讲稿草案。
- 所有外部素材（图片、图表、第三方图示）必须包含来源/许可与 attribution（`require_image_attribution=true`）。
- 文本与背景颜色对比应满足 WCAG AA（或项目指定 `wcag_contrast_level`），并提供 `min_contrast_ratio` 校验。

### SHOULD（建议，自动化可修复）
- 使用统一且有限的调色板（品牌颜色优先）；避免低对比或饱和度过强的组合。
- 对于架构/流程/时间线/数据类页面，优先生成/嵌入图示（flowchart, timeline, bar/line/pie）并在 notes 中补充解说。
- 图表清晰标注坐标轴、单位、数据来源与时间窗口。
- 表格类数据应优先转换为图表并保留关键数值 highlight。

### MAY（可选）
- 为可访问性提供备用文本（alt text）与高对比色主题版本。
- 对于外部公开材料，包含版权/许可页（Attribution slide）。

## 可量化检查（Skill/CI 可执行）
- max_bullets_per_slide (int)
- min_title_font_pt, min_body_font_pt (number)
- max_text_density_percent (number)
- require_speaker_notes (bool)
- require_decision_slide_within_first_n (int)
- require_image_attribution (bool)
- wcag_contrast_level / min_contrast_ratio
- qa_pass_threshold (number)

这些字段对应文件 `ppt-guidelines.json` 与 `schema.json`；更新规则时请同时更新 JSON 与本文件说明。

## Auto-fix 策略（Agent 可自动应用）
- 自动拆分：当 bullets > `max_bullets_per_slide` 时，拆分为多页并在 notes 标注 "auto-split"。
- 占位插入：若缺失 `Key Decisions`，可在封面后插入占位 `Key Decisions` 页并在 notes 写入 `AUTO_INSERT` 说明。
- Notes 生成：在缺少 speaker notes 时，生成简短 speaker notes 草案并标注来源（自动或基于模板）。

注意：对比度不合格、版权缺失、敏感合规问题不得通过自动修复，需进入 `human-in-loop`（阻断合并）。

## QA 流程与 CI 集成
1. 生成后运行 `ppt-guidelines` skill，输出 `qa_report.json`：{
   - score: 0-100,
   - issues: [{severity: 'critical'|'major'|'minor', path: slide_index, rule: '...'}],
   - suggestions: [ ... ]
}
2. 如果 `score < qa_pass_threshold` 或存在 `critical` 问题：
   - 若可自动修复且 `auto_fix=true`，执行 auto-fix 并重新运行 QA；
   - 否则阻断合并并生成 Review 注释（包含问题与建议）。

CI 示例（GitHub Actions）:
```yaml
- name: PPT Guidelines Check
  run: |
    python -c "import json,jsonschema; data=json.load(open('standards/ppt-guidelines/ppt-guidelines.json')); schema=json.load(open('standards/ppt-guidelines/schema.json')); jsonschema.validate(data,schema); print('schema OK')"
    # run skill (local or service) to check slides and fail on critical issues
```

## Key Decisions 模板（示例）
- 决策：选用 WebAssembly + WebGL 做前端核心运算
- 备选：完全依赖后端（CPU/GPU），或使用纯 Canvas2D
- 评估标准：延迟（p95 < 100ms）, 成本（$ / 1000 用户）, 离线能力
- 风险：浏览器兼容性、WebAssembly 二进制体积、调试成本

## 版本、治理与贡献流程
- 此文档为规范性说明，与 `ppt-guidelines.json` 与 `schema.json` 保持一致。
- 更新流程：修改 JSON schema → 更新 `ppt-guidelines.json` → 更新本文件与 `GUIDELINES.md` → 提交 PR；CI 将校验 schema 与样例。

## 参考资料
### 设计哲学经典著作
- **Presentation Zen** (Garr Reynolds) — 简约主义与视觉叙事
- **Slide:ology** (Nancy Duarte) — 图表设计与视觉思维
- **The Visual Display of Quantitative Information** (Edward Tufte) — 数据可视化诚实性
- **Clear and to the Point** (Stephen Kosslyn) — 认知科学视角的幻灯片设计
- **The Assertion-Evidence Approach** (Michael Alley) — 学术报告标准
- **The Pyramid Principle** (Barbara Minto) — McKinsey 逻辑结构方法

### 在线资源
- [WCAG Contrast Guidelines](https://www.w3.org/WAI/WCAG21/Understanding/contrast-minimum.html)
- [Guy Kawasaki's 10/20/30 Rule](https://guykawasaki.com/the_102030_rule/)
- [Takahashi Method 介绍](https://en.wikipedia.org/wiki/Takahashi_method)
- [McKinsey Pyramid Principle](https://www.mckinsey.com/business-functions/strategy-and-corporate-finance/our-insights/the-pyramid-principle)

### 内部文档
- `standards/design-review-checklist.md`
- `agents/ppt-specialist.agent.md`
- `standards/ppt-guidelines/GUIDELINES.md` — 机器可读快速参考

---

请在团队 review 后，将同意的规则落地成 `ppt-guidelines.json` 的默认值（例如 `max_bullets_per_slide:5`、`qa_pass_threshold:70`）。
