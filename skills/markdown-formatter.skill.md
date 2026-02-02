# Markdown Formatter Skill

**Purpose:**
Provide automated format validation and fixing for Markdown documents,
ensuring compliance with markdownlint rules.

This skill is designed to be called by `markdown-writer-specialist` and
other agents to enforce consistent formatting standards.

---

## 核心功能

### 1. Format Validation

检测 Markdown 文件中的格式问题（不修改文件）

### 2. Auto-Fix

自动修复可修复的格式问题（创建备份）

### 3. Report Generation

生成详细的格式问题报告

---

## 使用方式

### 命令行接口

```bash
# 1. 检测格式问题（只报告，不修改）
npx markdownlint-cli <file_or_directory>

# 2. 自动修复（创建 .bak 备份）
npx markdownlint-cli --fix <file_or_directory>

# 3. 使用 prettier 格式化（备选方案）
npx prettier --write <file>

# 4. 检测表格对齐问题（需要 md-table-fixer.skill）
python3 tools/md_table_tool.py detect <file_or_directory>

# 5. 修复表格对齐（需要 md-table-fixer.skill）
python3 tools/md_table_tool.py fix <file_or_directory>
```

### Agent 调用模式

**在 markdown-writer-specialist 中使用**：

```yaml
workflow:
  step1_generate_content:
    action: 生成文档内容
    output: draft.md
  
  step2_format_validation:
    action: 检测格式问题
    command: npx markdownlint-cli {file_path}
    decision:
      - if errors found → step3_auto_fix
      - if no errors → step4_deliver
  
  step3_auto_fix:
    action: 自动修复格式
    commands:
      - npx markdownlint-cli --fix {file_path}
      - python3 tools/md_table_tool.py fix {file_path}  # 表格对齐
    validation: 重新运行 markdownlint 检查剩余错误
  
  step4_deliver:
    action: 交付文档
    guarantee: 0 markdownlint errors
```

---

## 支持的修复规则

### 自动可修复（markdownlint --fix）

| 规则   | 说明                       | 示例                        |
| ------ | -------------------------- | --------------------------- |
| ------ | -------------------------- | --------------------------- |
| MD009  | 删除行尾空格               | `text` → `text`             |
| MD010  | 替换 tab 为空格            | `\\t` → `<4 spaces>`        |
| MD012  | 删除多余空行               | 3 个空行 → 1 个             |
| MD022  | 标题前后添加空行           | 见下方示例                  |
| MD032  | 列表前后添加空行           | 见下方示例                  |
| MD047  | 文件末尾添加换行符         | (EOF) → `\n`                |

### 需手动修复或工具辅助

| 规则  | 说明           | 工具                         |
| ----- | -------------- | ---------------------------- |
| ----- | -------------- | ---------------------------- |
| ----- | -------------- | ---------------------------- |
| MD001 | 标题层级跳跃   | 人工调整                     |
| MD013 | 行过长         | prettier（重排段落）         |
| MD060 | 表格对齐       | md-table-fixer.skill (必需)  |

---

## 常见格式问题与修复

### MD022: Headings should be surrounded by blank lines

**错误示例**：

```markdown
## 标题1
内容开始...
## 标题2
```

**修复后**：

```markdown
## 标题1

内容开始...

## 标题2
```

**自动修复命令**：

```bash
npx markdownlint-cli --fix <file>
```

---

### MD032: Lists should be surrounded by blank lines

**错误示例**：

```markdown
前文内容
- 列表项1
- 列表项2
后文内容
```

**修复后**：

```markdown
前文内容

- 列表项1
- 列表项2

后文内容
```

**自动修复命令**：

```bash
npx markdownlint-cli --fix <file>
```

---

### MD013: Line length

**错误示例**：

```markdown
这是一段非常非常非常非常非常非常非常非常非常非常非常非常非常非常非常长的文本，超过了120字符的限制。
```

**修复方案**：

```bash
# 使用 prettier 自动重排段落（推荐）
npx prettier --write --prose-wrap always <file>

# 或手动断行
这是一段非常非常非常非常非常非常非常非常非常非常非常非常非常非常
非常长的文本，现在符合120字符限制。
```

---

### MD060: Table alignment

**错误示例**：

```markdown
| Name         | Description |
| ------------ | ----------- |
| ----         | ----------- |
| ConfigLoader | 加载配置    |
| HttpClient   | HTTP客户端  |
```

**修复后**（管道符垂直对齐）：

```markdown
| Name         | Description |
| ------------ | ----------- |
| ------------ | ----------- |
| ConfigLoader | 加载配置    |
| HttpClient   | HTTP客户端  |
```

**修复命令**：

```bash
python3 tools/md_table_tool.py fix <file>
```

---

## 工具链要求

### 必需工具

```bash
# 1. markdownlint-cli (核心格式化工具)
npm install -g markdownlint-cli

# 2. prettier (可选，用于段落重排)
npm install -g prettier

# 3. md_table_tool.py (表格对齐工具)
# 已包含在 tools/ 目录中
```

### 配置文件

**`.markdownlint.json`**（项目根目录）：

```json
{
  "default": true,
  "MD013": { "line_length": 120 },
  "MD033": false,
  "MD041": false
}
```

---

## Agent 集成示例

### 完整工作流（markdown-writer-specialist）

```python
def write_and_format(content: str, output_path: str):
    """生成文档并确保格式合规"""
    
    # Step 1: 写入内容
    write_file(output_path, content)
    
    # Step 2: 检测格式问题
    result = run_command(f"npx markdownlint-cli {output_path}")
    
    if result.exit_code != 0:
        # Step 3: 自动修复
        run_command(f"npx markdownlint-cli --fix {output_path}")
        
        # Step 4: 修复表格对齐
        run_command(f"python3 tools/md_table_tool.py fix {output_path}")
        
        # Step 5: 重新验证
        result = run_command(f"npx markdownlint-cli {output_path}")
        
        if result.exit_code != 0:
            # 仍有错误，报告给用户
            return {
                "status": "partial",
                "path": output_path,
                "remaining_errors": result.stdout
            }
    
    return {
        "status": "success",
        "path": output_path,
        "errors": 0
    }
```

---

## 返回值规范

### 成功（0 errors）

```yaml
status: success
file_path: /path/to/document.md
errors_before: 71
errors_after: 0
actions_taken:
  - markdownlint --fix
  - md_table_tool.py fix
```

### 部分成功（仍有错误）

```yaml
status: partial
file_path: /path/to/document.md
errors_before: 71
errors_after: 5
remaining_errors:
  - line: 42
    rule: MD001
    message: "Heading levels should increment by one level at a time"
  - line: 108
    rule: MD013
    message: "Line length: 156 (max 120)"
actions_taken:
  - markdownlint --fix (42 errors fixed)
  - md_table_tool.py fix (24 tables fixed)
manual_fix_required: true
```

### 失败

```yaml
status: error
file_path: /path/to/document.md
error_message: "markdownlint-cli not found. Run: npm install -g markdownlint-cli"
```

---

## 最佳实践

### 1. 生成内容时预防格式问题

```markdown
# 标题前后主动留空行

内容段落

- 列表前后留空行
- 保持一致

下一段内容
```

### 2. 表格使用预对齐模板

```markdown
| Column1      | Column2         | Column3    |
| ------------ | --------------- | ---------- |
| ------------ | --------------- | ---------- |
| Value1       | Value2          | Value3     |
```

### 3. 链接和代码块不换行

```markdown
<!-- 好 ✅ -->
详细文档请参见 [Architecture Design](https://example.com/very-long-url)

<!-- 差 ❌ - 会触发 MD013 -->
详细文档请参见 [Architecture
Design](https://example.com/very-long-url)
```

### 4. 中文段落控制长度

```markdown
<!-- 控制每行在 60 个汉字以内（约 120 字符）-->
这是一段中文内容，通过合理断句和分段来避免单行过长的问题。
可以使用句号、逗号等标点符号自然断开。
```

---

## 故障排查

### markdownlint 不工作

```bash
# 检查安装
which markdownlint

# 重新安装
npm install -g markdownlint-cli@latest

# 验证版本
markdownlint --version  # 应 >= 0.35.0
```

### 表格修复失败

```bash
# 检查 Python 环境
python3 --version  # 应 >= 3.8

# 检查工具存在
ls tools/md_table_tool.py

# 手动测试
python3 tools/md_table_tool.py detect <file>
```

### 修复后仍有错误

```bash
# 查看详细错误信息
npx markdownlint-cli <file>

# 检查 .markdownlint.json 配置
cat .markdownlint.json
```

---

## 限制与注意事项

1. **MD001（标题层级）**：必须人工修复，工具无法自动调整文档结构
2. **MD013（行过长）**：
   - 代码块、链接、表格中的长行会被忽略
   - 普通段落需人工或 prettier 处理
3. **MD060（表格对齐）**：
   - markdownlint 检测不严格，必须使用 md-table-fixer.skill
   - 包含 CJK 字符的表格必须使用显示宽度计算
4. **备份策略**：
   - markdownlint --fix 不创建备份
   - md_table_tool.py 会创建 .bak 文件
   - 建议先提交 git 或手动备份

---

## 相关技能

- [md-table-fixer.skill.md](md-table-fixer.skill.md) - 专门的表格对齐工具
- [ppt-markdown-parser.skill.md](ppt-markdown-parser.skill.md) - PPT Markdown 解析

---

## 更新日志

- 2026-02-02: 创建初始版本
  - 定义格式化工作流
  - 集成 markdownlint + md-table-fixer
  - 提供 Agent 调用接口
