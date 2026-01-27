# Short-Video Evaluation Plan

版本：v0.1 · 2026-01-27
Owner: @data-scientist-evaluator

## 目标
描述可复现的自动与主观评估流程，用于评估短视频替换系统的质量（音频、视觉、同步、合规）。包含：自动化指标计算脚本、样本量计算、主观 MOS 流程、鲁棒性测试矩阵与最终报告模板。

## 数据集
- Test sets:
  - Chronological held-out (时间序列后续数据)
  - Speaker-holdout (某些说话人完全 holdout)
  - Noisy / robustness set（压缩、SNR、转码）
  - High-risk slices（名人、未成年人、敏感话题）

- 标注/格式要求：
  - 每条样本需包含 model_version, data_snapshot_id, consent_id, reference_audio_path, generated_audio_path, reference_video_path, generated_video_path
  - CSV 头：sample_id, slice, ref_audio, gen_audio, ref_video, gen_video, metadata_json

## 自动化指标（脚本化）
- 音频质量：PESQ, SI-SDR, MCD（当可用）
- 说话人相似性：speaker embedding cosine（ECAPA-TDNN）
- 口型同步：LSE-C / LSE-D（Wav2Lip / SyncNet）
- 视觉质量：FID / LPIPS / SSIM（基于关键帧或 frame-level）
- 合规检测：watermark detection rate (detected, confidence)

输出：按样本 CSV 与切片聚合 summary JSON/CSV（含均值、95% CI）

## 主观评估（MOS）
- 抽样策略：总体 N=200（最低），每重要切片 N≥50。
- 评分说明：1-5 Likert（1 非常差 - 5 非常好），MOS 平均及标准误
- 口型一致性：人工打分 + A/B 盲测（是否更贴合于参考）
- 抽样与盲测流程见 `evaluation/scripts/mos_survey_template.md`

## 样本量计算
- 使用 `evaluation/scripts/sample_size_calc.py` 自动计算样本量，默认参数：alpha=0.05, power=0.8, sigma=1 (for MOS)、MDE 设定在 0.3

## 鲁棒性测试矩阵
- 压缩：H.264 CRF {18, 23, 28, 33}
- SNR：{30, 20, 10, 0} dB
- 转码：mp4 -> webm (vp9) -> mp4
- 低带宽/丢帧模拟：10% frame drop
- 每项至少 100 样本

## 对抗测试
- 小幅度扰动（PGD-like）与合成干扰（高斯噪声、速度伸缩）用于检测器鲁棒性

## 报告模板
- 生成 `evaluation/report_template.md`，包含：Executive Summary, Auto Metrics Table, MOS Results, Slice Analysis, Robustness Results, Risk & Recommendations

## 自动化脚本
路径：`evaluation/scripts/compute_metrics.py`, `evaluation/scripts/sample_size_calc.py`
- 使用方法：
  - 安装依赖：`pip install -r evaluation/scripts/requirements.txt`（示例依赖：numpy, scipy, librosa, pesq, torch, pyannote.audio, lpips）
  - 运行：`python evaluation/scripts/compute_metrics.py --input samples.csv --output metrics.csv`

## 验收标准
- 自动指标通过：speaker cosine ≥ 0.8（总体）或按切片达成预设阈值
- MOS ≥ 3.8（总体）且每切片 MOS ≥ 3.5
- Watermark detection rate ≥ 99%（FN ≤ 1%）

## Timeline
- 文档与脚本初稿：1 天（完成）
- 自动化运行与 baseline evaluation：3 天
- 主观 MOS 收集与分析：7-10 天

---

**Last Updated**: 2026-01-27
