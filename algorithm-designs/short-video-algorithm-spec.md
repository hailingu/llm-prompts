# Algorithm Design Spec: Short-Video Replacement System

- **项目名**：短视频批量替换系统 - Algorithm Design Spec
- **Owner**：@data-scientist-algorithm-designer
- **版本 / 日期**：v0.1 · 2026-01-27
- **Reviewers**：@data-scientist-tech-lead, @data-engineer, @legal, @data-scientist-evaluator

---

## 概述
本规范描述实现短视频批量替换系统各模块的算法、基线实现、训练与评估策略、鲁棒性与监控计划。重点覆盖语音转换（VC）、TTS、口型同步、视觉替换/卡通化、watermark 嵌入与检测、以及 fail-safe 人工复核设计。

---

## 1. 模块化架构与 Baseline（含 repo + tag）

| 模块 | Baseline 实现 | Repo / Tag | 备注 |
|------|---------------|------------|------|
| 说话人嵌入 | ECAPA-TDNN | https://github.com/pyannote/pyannote-audio @v2.0 | 说话人验证/聚类用 |
| 语音转换 / TTS | YourTTS (few-shot) | https://github.com/TensorSpeech/yourTTS @commit abcdef1 | few-shot VC baseline |
| TTS（备用） | VITS + HiFi-GAN | VITS repo @v1.0; HiFi-GAN v2 | 高质量合成路径 |
| 口型同步 | Wav2Lip | https://github.com/Rudrabha/Wav2Lip @v0.1.1 | 同步 + 插帧支持 |
| 口型检测 | SyncNet | https://github.com/joonson/syncnet_python (commit xyz) | 用作检测器/报警 |
| 人脸驱动 | First-Order-Motion-Model | https://github.com/AliaksandrSiarohin/first-order-model @master | 视觉驱动 baseline |
| 高保真替换 | FaceShifter | https://github.com/foodoo/face-shifter (fork) | temporal-consistency 强化 |
| 卡通化 | U-GAT-IT / CartoonGAN | Repo + tag as chosen per style | 风格迁移 |
| deepfake 检测 | XceptionNet-based detector | https://github.com/deepfakes/fake-detection-challenge @baseline | 训练与基准 |

> 注：提交实现时必须包含 exact commit hash 或 tag，并在 PR 中注明复现命令与依赖版本（requirements.txt / conda env）。

---

## 2. 数据划分与防泄漏策略

- **切分策略**：主训练/验证/测试按时间序列或说话人隔离两套规则任选其一（以避免同一说话人同时出现在 train 和 test 中）：
  - 时间序列优先用于时间敏感任务（chronological split）
  - 说话人隔离用于 speaker-invariant 测试（speaker holdout）——建议对生成任务两者都保留一个独立的冲突检查集
- **重叠检测**：实现 speaker overlap 检查（Jaccard 融合 speaker_id 集），若 overlap > 0 给出警告并阻止切分
- **标签泄露检查**：禁止使用未来信息（例如：未来用户行为）作为特征；特征工程脚本必须记录时间窗口与滑动窗口逻辑
- **Consent 筛选**：训练集仅包含已通过 `consent_id` 验证的样本

---

## 3. 训练细节（示例）

### 3.1 语音转换 (YourTTS) - 基线训练
- 预训练模型：使用 upstream pre-trained weights（记录 commit）
- Fine-tune：few-shot setup（5-30s sample）与 speaker embedding fusion
- 超参示例：batch_size 32, lr range [1e-4, 1e-3], scheduler: CosineAnnealing, weight_decay 1e-6
- Checkpoint：每 N=1000 step checkpoint；保留 5 最佳（by val loss + speaker-similarity）
- 随机种子：设置 seed = 42，并在 MLflow 中记录所有参数

### 3.2 视觉替换（First-Order / FaceShifter）
- 训练/微调使用 paired 或 unpaired 数据（视供应数据情况），冻结低层特征保留动画能力
- 损失项：perceptual loss (LPIPS), identity loss (ArcFace), temporal consistency (L1 across frames + flow-based loss)

### 3.3 同步（Wav2Lip）
- 使用 LSE-C / LSE-D 作为优化与评价指标
- 对出现 lip-sync failure 的样本，尝试两轮自动微调：调整时间轴微偏移（±50ms），若仍失败转人工复核

---

## 4. 评价计划（自动 + 主观）

### 自动指标
- speaker embedding cosine（目标 ≥ 0.8）
- LSE-C / LSE-D thresholds（根据 baseline 定义可接受值）
- PESQ / SI-SDR / MCD（音质）
- FID / LPIPS / SSIM（视觉质量）
- Watermark detection rate（目标 ≥ 99%）

### 主观评估
- MOS (1-5) 抽样，至少 200 个样本（overall）以得出稳定点估计；每个切片（语言/年龄）最少 50 样本
- 口型一致性人工打分（Likert）以及盲测 A/B 对比

### 统计要求
- A/B 测试设计需满足 power ≥ 0.8，α=0.05；提前计算 MDE 并列入实验 plan

---

## 5. 鲁棒性测试矩阵
- 场景：压缩 (H.264 4档), 带噪 (SNR 0-30 dB), 转码（mp4 → webm），部分遮挡、低光照
- 每个场景定义 test set，目标：在多数压缩/噪声条件下，speaker similarity decline ≤ 10% relative baseline
- 对抗样本：使用 simple perturbation / PGD style 测试 detection 与 model robustness

---

## 6. Watermark 与 Provenance 集成
- **嵌入点**：在后处理阶段（合成完音视频后）植入 learned watermark 与频域冗余
- **检测 API**：POST /internal/wm/check -> {detected, confidence, payload}
- **验收**：FN ≤ 1%，confidence < 0.6 标记人工复核
- **元数据**：为每条生成记录 {model_version, model_commit, data_snapshot_id, consent_id, generator_user_id, timestamp, signature}

---

## 7. Fail-safe 人工复核流程
- 触发条件：watermark 未检出、sync confidence < threshold、name in name-list（名人）、未成年相关、高-risk 内容类别
- 人工复核步骤：自动工单 -> QA 抽样 -> 人工决定（approve/reject/ask changes）
- SLAs：72 小时内完成复核；优先级高的 24 小时内

---

## 8. 监控与线上报警
- **Realtime / near-real-time 指标**：watermark detection rate, sync fail rate, avg inference latency(p50/p95/p99), user takedown ratio
- **Thresholds**：watermark < 99% -> P1; sync fail rate > 10% -> P2; takedown ratio > 0.1% -> P1
- **仪表盘**：Grafana + prometheus metrics; 报告发送到 #ml-ops and pager for P1

---

## 9. 复现性与记录
- 每次 experiment 记录：data_snapshot_id / model_commit / hyperparams / seed / artifact uri
- 版本化 model artifact（artifact registry）与 model card 生成（参照 doc-writer 模板）

---

## 10. 资源估算 & CI
- 训练：相对小规模 baseline fine-tune 每实验约 4–16 GPU-hours（4x A100）；大规模训练 ~100–500 GPU-hours
- 推理：single 30s video 估算 3–8 GPU-min（MVP pipeline），需约 500–1000 GPU-hours/月 以满足 10k/月产能
- CI：unit tests, integration tests, nightly GE checks, sample blind MOS pipeline（自动触发抽样测试）

---

## 11. 安全、隐私与合规注意事项
- 训练数据仅包含已授权样本（consent_id）；所有用于检测/审查的敏感信息以最小化原则保存并加密
- 如需启用 DP/合成数据替代，需额外分析对性能影响并在 Design Spec 中记录

---

## 12. Owners & Timeline
- Algorithm Designer: @data-scientist-algorithm-designer — 交付 Algorithm Spec 完整版（7 天）
- Data Owner: @data-engineer — 数据准备（10 天）
- QA/Manual Review: @data-scientist-evaluator — 评估计划（10 天）
- Infra: @infra-team — Watermark API 与监控（14 天）

---

## 13. Acceptance Criteria (pre-implementation)
- Baseline repo + tag recorded
- Data snapshot for training available and consented
- GE checks pass on training data
- Detection / watermark baseline test implemented with FN ≤ 1%

---

**Last Updated**: 2026-01-27
**Version**: v0.1
