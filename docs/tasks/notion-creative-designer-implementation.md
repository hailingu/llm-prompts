# Notion 风格创造性幻灯片系统 — 实施文档（可逐步执行）

> 日期：2026-02-12  
> 目标：从 `outline-*.txt` + `docs/reports/datasets/*.csv` 生成高质量 KPMG 风格 `slide-*.html`  
> 范围：先交付 MVP（覆盖 slide-1/5/6/9/12 的能力）

---

## 0. 执行前准备

### 0.1 前置检查

- [ ] Python 环境可用（建议 3.10+）
- [ ] 已激活虚拟环境：`.venv`
- [ ] 可访问项目目录：`/Users/guhailin/Git/llm-prompts`

### 0.2 基线命令

```bash
cd /Users/guhailin/Git/llm-prompts
source .venv/bin/activate
python3 --version
```

### 0.3 成功标准（全局）

- [ ] 可输入 `outline-*.txt`，输出对应 `slide-*.html`
- [ ] 页面尺寸固定 1280x720，KPMG 配色一致
- [ ] 图表能力至少覆盖：雷达图、折线图、气泡图、热力图
- [ ] 具备“重写图表建议”的策略（如 outline-9）

---

## Phase 1：定义 MVP 架构（半天）

### Step 1.1 建立目录结构

- [ ] 新建工作目录：`skills/ppt-creative-designer/`
- [ ] 新建模板目录：`skills/ppt-creative-designer/templates/`
- [ ] 新建样例输出目录：`docs/presentations/notion-clone-mvp/`

执行命令：

```bash
mkdir -p skills/ppt-creative-designer/templates
mkdir -p docs/presentations/notion-clone-mvp
```

验收：

- [ ] 目录创建完成且无报错

回滚：

```bash
rm -rf skills/ppt-creative-designer docs/presentations/notion-clone-mvp
```

### Step 1.2 创建核心模块骨架

创建文件：

- `skills/ppt-creative-designer/__init__.py`
- `skills/ppt-creative-designer/engine.py`
- `skills/ppt-creative-designer/parser.py`
- `skills/ppt-creative-designer/decision.py`
- `skills/ppt-creative-designer/renderer.py`

验收：

- [ ] 所有文件可 import，不报语法错误

---

## Phase 2：实现输入解析（1 天）

### Step 2.1 解析 outline 文本

目标：支持解析以下字段

- 标题（第一行）
- 图表要求（如存在 `图表：...`）
- 洞察列表（如存在 `洞察：` + 列表）
- 阶段化内容（如 H1/H2/H3）

实现要点：

- 输出统一结构 `OutlineSpec`
- 允许缺省字段（如 outline-12 没有显式图表）

验收：

- [ ] `outline-1/9/12` 均能解析成功
- [ ] 解析失败时返回可读错误信息

### Step 2.2 解析 CSV 数据

目标：读取 `docs/reports/datasets/*.csv`，输出统一 DataFrame/字典结构

实现要点：

- 优先使用 `pandas`
- 保留列名原始大小写
- 自动识别数值字段（index/growth/confidence 等）

验收：

- [ ] 可读取 `rd_talent_index.csv`、`tech_maturity.csv`、`segment_index.csv`
- [ ] 空文件/坏格式可报错并退出

---

## Phase 3：实现创造性决策引擎（1.5 天）

### Step 3.1 图表类型决策规则（MVP）

建立规则优先级：

1. 数据结构优先于 outline 推荐
2. 多维成熟度对比 → 雷达图
3. 时间序列多系列 → 折线图
4. 三变量（x/y/size）→ 气泡图
5. 二维矩阵强弱 → 热力图

关键场景：

- [ ] outline-9 推荐 grouped_bar 时，若识别 `x=index, y=growth, size=confidence`，自动改为 bubble
- [ ] outline-12 无图表要求时，基于阶段+KPI映射出 heatmap/matrix 方案

验收：

- [ ] 决策结果与 slide-5/6/9/12 对应风格一致
- [ ] 决策日志落盘（JSON）

### Step 3.2 布局决策规则（MVP）

支持 4 类布局：

- `cover_left_brand`（slide-1）
- `chart_left_insights_right`（slide-5/9）
- `chart_top_insights_grid_bottom`（slide-6）
- `heatmap_plus_matrix_plus_insights`（slide-12）

验收：

- [ ] 根据内容类型自动选择布局
- [ ] 页面元素不越界、不重叠

---

## Phase 4：实现 HTML 渲染器（1.5 天）

### Step 4.1 创建基础模板

模板文件建议：

- `base.html.j2`
- `layout_cover.html.j2`
- `layout_chart_insight.html.j2`
- `layout_heatmap_matrix.html.j2`

通用要求：

- 固定尺寸 `1280x720`
- 内置 KPMG 色板：`#00338D`, `#0091DA`, `#483698`
- 字体：`Noto Sans SC`

验收：

- [ ] 模板渲染不报错
- [ ] 生成 HTML 可直接浏览器打开

### Step 4.2 图表渲染接入

- Chart.js：雷达图、折线图、气泡图
- ECharts：热力图（必要时）

验收：

- [ ] slide-5 类页面可显示雷达图
- [ ] slide-6 类页面可显示多线图
- [ ] slide-9 类页面可显示气泡图
- [ ] slide-12 类页面可显示热力图

---

## Phase 5：端到端命令与批量生成（0.5 天）

### Step 5.1 提供 CLI

新增命令（示例）：

```bash
python3 -m skills.ppt-creative-designer.engine \
  --outline outline-9.txt \
  --datasets docs/reports/datasets \
  --output docs/presentations/notion-clone-mvp/slide-9.html
```

批量模式（示例）：

```bash
python3 -m skills.ppt-creative-designer.engine \
  --outline-dir . \
  --pattern "outline-*.txt" \
  --datasets docs/reports/datasets \
  --output-dir docs/presentations/notion-clone-mvp
```

验收：

- [ ] 单页生成成功
- [ ] 批量生成成功
- [ ] 异常输入返回非 0 退出码

---

## Phase 6：验证与对标（1 天）

### Step 6.1 功能验收

对标样例：`slide-1/5/6/9/12`

检查清单：

- [ ] 版式结构相似（非像素级复刻）
- [ ] 图表类型与表达逻辑一致
- [ ] 关键洞察区域完整
- [ ] 页脚和数据源信息存在

### Step 6.2 质量验收

- [ ] HTML 无明显结构错误
- [ ] JS 控制台无报错
- [ ] 首屏加载 < 2 秒（本地）
- [ ] 视觉一致性通过人工评审

---

## 里程碑与交付物

### M1（第 1 天末）

- 交付：解析器 + 模块骨架
- 产物：`parser.py`, `engine.py` 初版

### M2（第 3 天末）

- 交付：决策引擎 + 基础模板
- 产物：`decision.py`, `templates/*.j2`

### M3（第 5 天末）

- 交付：MVP 端到端可运行
- 产物：`slide-1/5/6/9/12.html`（自动生成）

### M4（第 6 天末）

- 交付：验收报告 + 使用文档
- 产物：`docs/reports/notion-clone-mvp-validation.md`

---

## 风险与处理

1. 决策规则不稳定  
   - 处理：为每种图表提供显式 fallback（line/bar）
2. 热力图库兼容性问题  
   - 处理：ECharts 不可用时降级为 CSS 矩阵
3. 大纲信息不足  
   - 处理：进入默认布局 + 默认洞察占位逻辑

---

## 每日执行模板（可复制）

```markdown
## Day X
- 目标：
- 完成：
- 遇到问题：
- 解决方案：
- 明日计划：
- 风险状态（高/中/低）：
```

---

## 启动命令（建议今天就执行）

```bash
cd /Users/guhailin/Git/llm-prompts
source .venv/bin/activate
mkdir -p skills/ppt-creative-designer/templates
mkdir -p docs/presentations/notion-clone-mvp
```

如果需要，我下一步可以直接继续：按这份文档把 `skills/ppt-creative-designer/` 的最小可运行骨架一次性搭好。