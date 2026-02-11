# 存储行业前沿（2023–2026）报告

说明：本目录存放《存储行业前沿（国内/国外/学术界）发展状况（2023–2026）》相关交付物。

交付物（首版草稿，48 小时里程碑）：
- `storage-frontier-2026.md`：主报告（Markdown，含图表占位与表格）
- `data/*.csv`：用于生成图表的原始数据（CSV）
- `data/case_aws_estimates.csv`、`data/case_aliyun_estimates.csv`、`data/case_huawei_estimates.csv`、`data/case_dell_estimates.csv`：厂商案例示例数据（估算/示例）
- `cases/*.md`：厂商详细案例（示例：`reports/cases/aws.md`、`reports/cases/aliyun.md`、`reports/cases/huawei.md`）
- `figures/*.png`：图表 PNG（已生成，见 `reports/figures/*.png`）；如需重新生成，请使用绘图脚本 `reports/figures/scripts/plot_all.py`
- `figures/scripts/*.py`：绘图脚本（使用 pandas + matplotlib/plotly）

方法与注意事项：
- 报告语言：中文；引用格式：数字编号（例如 [1]）；时间范围：2023–2026
- 数据优先使用公开/开源来源（学术会议、厂商白皮书、公开财报、SNIA、MLPerf 等）；如引用付费报告，将在文末与 `README.md` 说明不可公开的数据与替代策略
- 所有图表均附带 CSV 与图注（数据来源名称与采集时间），并在 `reports/README.md` 中记录未授权或需要付费的数据项

里程碑时间表（已同意）：
- 6–12 小时：详细提纲与数据来源清单（已提交）
- 24–36 小时：主要 CSV 数据表与初步图表草稿（CSV + 绘图脚本/占位）
- 48 小时：完整第一版草稿（`storage-frontier-2026.md` + CSV + 图表或绘图脚本）

负责人：`markdown-writer-specialist`（已委派）

如需调整交付格式或补充指定厂商/案例，请在本 issue/对话中回复。

---

运行与生成图表（建议）：

1. 在仓库根目录创建 Python 虚拟环境并安装依赖：

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt pandas
```

2. 生成/更新 PNG 图表：

```bash
cd reports/figures/scripts
../../.venv/bin/python3 plot_all.py
```

生成的图表存放：`reports/figures/*.png`。