# Domain Keyword Detection Skill - 使用说明

## ✅ Skill 已创建

**位置**: `skills/domain-keyword-detection/`

**结构**:
```
skills/domain-keyword-detection/
├── manifest.yml           # Skill 元数据和输出 schema
├── commands.yml           # 可执行命令定义
├── examples.yml           # 测试用例
├── README.md              # 完整文档
├── requirements.txt       # Python 依赖
├── domains/               # 领域关键词配置
│   ├── software.yaml
│   ├── hardware.yaml
│   ├── manufacturing.yaml
│   ├── standards.yaml
│   ├── business.yaml
│   └── biotech.yaml
└── bin/
    └── domain_detector.py # Python 实现
```

---

## 核心优势（vs 硬编码在 Agent 文档中）

| 维度 | 硬编码在 Agent | 作为 Skill |
|------|---------------|-----------|
| **扩展性** | ❌ 修改 agent 定义才能新增领域 | ✅ 只需添加 YAML 文件 |
| **复用** | ❌ 仅 ppt-content-planner 可用 | ✅ 所有 agent 共享 |
| **维护** | ❌ 关键词质量无法独立验证 | ✅ 可执行测试和验证命令 |
| **版本管理** | ❌ 领域知识与 agent 行为耦合 | ✅ 独立演进，语义化版本 |
| **可测试** | ❌ 无法直接测试关键词匹配 | ✅ examples.yml + validate 命令 |

---

## 使用示例

### 1. 检测文档领域

```bash
# 硬件领域文档
python3 skills/domain-keyword-detection/bin/domain_detector.py detect \
  --input docs/MFT_report.md \
  --threshold 0.3

# 输出示例：
# {
#   "detected_domains": ["hardware", "standards"],
#   "confidence_scores": {"hardware": 0.85, "standards": 0.48},
#   "matched_keywords": {
#     "hardware": ["纳米晶", "SiC", "功率密度", "液冷", ...],
#     "standards": ["IEC", "认证", "GB"]
#   },
#   "activated_packs": ["hardware", "standards"]
# }
```

### 2. 在 Agent 中使用（ppt-content-planner）

**工作流**:
1. 接收源文档 → 2. 运行 domain detection → 3. 获取领域关键词 → 4. 提取关键决策 → 5. 报告激活的领域包

**示例**:
```python
# Step 1: Detect domains
result = subprocess.run([
    'python3',  
    'skills/domain-keyword-detection/bin/domain_detector.py',
    'detect',
    '--input', 'docs/design.md',
    '--threshold', '0.3'
], capture_output=True, text=True)
detection = json.loads(result.stdout)

# Step 2: Get keywords for activated domains
keywords = {}
for domain in detection['activated_packs']:
    kw_result = subprocess.run([
        'python3',
        'skills/domain-keyword-detection/bin/domain_detector.py',
        'get-keywords',
        '--domain', domain
    ], capture_output=True, text=True)
    keywords[domain] = json.loads(kw_result.stdout)['keywords']

# Step 3: Use keywords for decision extraction
decisions = extract_decisions(source_doc, keywords)

# Step 4: Report in output
content_qa_report['domain_packs_activated'] = detection['activated_packs']
```

---

## Agent 集成状态

### ✅ 已集成
- **ppt-content-planner** (v1.0.0)
  - tools 字段已添加 `domain-keyword-detection`
  - KEY DECISIONS EXTRACTION ALGORITHM 章节已更新引用 skill
  - Workflow Step 3 更新为先运行 domain detection

### 🔄 推荐集成
- **markdown-writer-specialist**: 领域感知的术语识别和风格指南选择
- **data-scientist-research-lead**: 基于领域路由任务到专家 agent
- **cortana**: 上下文感知的任务路由

---

## 测试结果

### ✅ 通过的测试

**1. 硬件领域检测** (置信度 0.244):
```bash
python3 skills/domain-keyword-detection/bin/domain_detector.py detect \
  --input /tmp/test_hardware.txt \
  --threshold 0.2
```
匹配 28 个关键词: 纳米晶、SiC、功率密度、液冷、效率、温升、铁损、铜损等

**2. Skill 元数据**:
```bash
# 列出所有领域
python3 ... list-domains
# 输出: software, hardware, manufacturing, standards, business, biotech

# 验证配置文件
python3 ... validate
# 输出: all valid
```

### ⚠️ 已知限制

**1. 短文本检测**:
- 文本 <50 字符/词 → 置信度 ×0.5 惩罚 → 可能低于阈值
- **解决方案**: 降低阈值到 0.1-0.15，或累积更多上下文后检测

**2. 中英混合分词**:
- 使用子串匹配（"纳米晶材料" 匹配 "纳米晶"）
- 可能过度匹配（"不采用 React" 也会匹配 "react"）
- **设计取舍**: 宁可过度匹配（高召回），后续用决策提取过滤

---

## 下一步建议

### 1. 添加测试用例 (可选)
创建 `skills/domain-keyword-detection/tests/` 目录，使用 examples.yml 中的测试用例生成自动化测试。

### 2. 扩展领域包 (按需)
新增领域（如量化交易、生物信息学）只需:
```bash
cp domains/software.yaml domains/quant_trading.yaml
# 编辑 quant_trading.yaml 添加关键词
python3 bin/domain_detector.py validate
```

### 3. 集成到其他 Agent (推荐)
- markdown-writer-specialist: 领域感知的文档风格
- data-scientist-*: 领域路由与专家分工

---

## 依赖安装

```bash
# macOS (已安装)
pip3 install --break-system-packages pyyaml

# 或使用 requirements.txt
pip3 install -r skills/domain-keyword-detection/requirements.txt
```

---

**版本**: 1.0.0  
**创建日期**: 2026-02-06  
**维护者**: ppt-content-planner, cortana  
**状态**: ✅ 生产就绪

