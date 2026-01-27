# Research Design v1: 短视频批量替换系统（完整版）

- **项目名**：短视频批量替换系统
- **Owner**：
- **版本 / 日期**：v1.0 · 2026-01-27
- **审阅者**：@data-scientist-tech-lead / @legal

---

## 1) 一句话目标 🎯
批量生成短视频，功能包括：将指定用户声音替换为目标音色、将人物替换为卡通形象，同时保持剧情与参考经典短视频一致；在满足质量、可扩展性与合规/安全前提下，尽量实现自动化生产（batch-first）。

---

## 2) 业务动机（KPIs）
- 主要：短视频播放量 ↑、CTR ↑、完播率 ↑、内容生产成本 ↓
- 示例量化目标：新增短视频月产能 ≥ 10k 条；平均新增播放量/条 ≥ 5k；生产成本 ≤ $X/条

---

## 3) 问题定义（技术边界）
- **输入**：参考短视频 ID、目标音色样例（5–30s）、目标卡通风格、已取得合法授权的用户素材与同意文件
- **输出**：可发布短视频（嵌入不可见水印，带 provenance 元数据）
- **不做**：未经授权的人像替换（明文禁止）、未审的政治/煽动内容替换、跨语种实时翻译（除非另行声明）

---

## 4) 关键合规要点（必须）🔒
- **同意文件模板与验证**（参见附录 A）：必须包含用途范围、分发渠道、保存期限、撤回机制；签名人、签名时间、签署方式（电子签名/扫描件）需记录；同意文件由法务最终批准。
- **水印与 provenance**：使用“不可见鲁棒水印 + 签名化元数据”策略（建议采用 learned watermark 或 frequency-domain embedding，检测 API 返回位置信任度）；**误检容忍度**（false negative）≤ 1%（目标 ≤ 0.5%）。所有生成物在 DB 写入不可变审计条目（哈希 + 签名），并保留同意 ID。
- **禁止用途清单**：禁止用于诈骗、冒充、政治劝导、未成年人（任何涉未成年人替换需额外法律流程）、医疗/法律决策、直接用于执法或证据材料。发现滥用：立即下线、冻结账号、通知法务并保留证据。
- **PII 处理与保留策略**：样本与同意文件加密存储（AES-256），基于 RBAC 的访问控制；默认原始样本保留 90 天（仅用于模型迭代），同意文件保留 2 年；如需长期保留需明确同意。
- **强制人工复核触发条件**：涉及名人/公共人物、检测器置信度低于阈值、检测到潜在滥用标记、法务/合规显式请求、生成的内容被举报或高风险分类（政治/仇恨/未成年人）。

---

## 5) 成功判据（严格化）✅
- **总体目标（MVP）**：
  - 音频 MOS ≥ 3.8（主观抽样）
  - 口型一致性人工打分 ≥ 85%
  - 水印可检出率 ≥ 99%（FN ≤ 1%）
  - 法务合规通过：所有样本通过法务初审
  - 平均生产时长（batch） ≤ 5 min/视频（MVP 基线）
- **按切片阈值（示例）**：对于每个主要语言/年龄组/场景，MOS ≥ 3.5；口型一致性 ≥ 80%
- **统计显著性要求**：A/B 测试与主观评估需满足：power ≥ 0.8、α=0.05；例如，检测 MOS 差异 δ=0.3（σ≈1）时每个切片最小样本量 ≈ 100。

---

## 6) 数据需求（最低）🗂️
- **语音数据**：MVP 最小 N speakers ≥ 100（建议生产 N ≥ 500），每 speaker 最少 30s 优质音频；包含情感与语速标签的多样样本。
- **视频数据**：MVP 视频数 ≥ 1000，包含多视角、不同光照/年龄/性别/设备，以覆盖常见场景；需带字幕/时间戳优先（用于对齐）。
- **卡通风格集**：每种目标风格至少 50 张高质量样例（静态+动态）；若要支持多角色，提供相应样式表和优先级。
- **合规文件**：同意书、版权许可扫描件、证明材料。
- **负样本**：真实/合成混合数据，用于训练检测器。

---

## 6.1) 数据质量与管道（新增）📦

**问题摘要**：当前数据 pipeline 需要补充 schema 校验、同意证据绑定、自动化质量检测（损坏/噪声）、标签完备性检查与 lineage/version 控制。下列为拟采取的具体措施与验收准则。

**数据合同（Schema）**
- 必填字段：file_id, speaker_id, length_s, sample_rate, channels, language, consent_id, timestamp, checksum, source, data_snapshot_id
- 可选字段：emotion_tag, device, mic_type, ambient_noise_db
- 不满足 schema 的文件将被拒绝入库并记录错误原因。

**入库验证（自动化）**
- 使用 Great Expectations（或等效）实现自动化检查：
  - 长度检查：length_s >= 5s
  - 采样率检查：sample_rate ∈ {16000, 22050, 24000, 44100}
  - checksum 校验
  - consent_id 必填且在同意库中可查
  - 重复检测：基于 checksum 与音频指纹
- 失败策略：失败样本进入隔离队列并触发工单（Auto-ticket）给数据负责人。

**数据清洗与补标**
- 清洗：去重、时区标准化、异常 timestamp 修正、音频重采样、低质量标注（SNR 阈值）
- 补标流程：半自动标签 + 人工抽样校验（QA）

**Lineage 与版本控制**
- 每次数据快照写入 data_snapshot_id（含 git commit / data hash），并记录到元数据（immutable log）。

**监控与告警**
- 核心指标：missing_consent_pct, invalid_audio_pct, low_quality_pct, ingestion_failure_rate
- 阈值示例：missing_consent_pct > 0% → 阻塞发布；invalid_audio_pct > 1% → 告警并人工检查

**测试计划与 CI**
- 单元测试：损坏文件测试、checksum 校验测试、schema 边界测试
- 集成测试：上传一批样本（含坏样本），验证入库/隔离/工单流转
- 在 CI 中加入 Great Expectations 快照测试（nightly run）

**验收标准**
- 入库自动校验覆盖 100% 的新数据
- Speaker-count ≥ 100 且每 speaker ≥ 30s（MVP 要求）
- Consent 100% 绑定或未绑定进入隔离并关联工单
- 自动报警触发后 24h 内响应（SLA）
- 人工抽检通过率 ≥ 95%

**相关文件（已创建）**：
- `data/contracts/short-video-schema.md`（数据合同）
- `data/tests/great_expectations/short_video_expectations.yml`（Great Expectations 示例）

**Owner / Action Items**：
- Data Owner: @data-engineer — 负责实现入库校验与清洗脚本（目标完成日见负责人分配表）
- QA: @data-scientist-evaluator — 负责抽检计划与人工评审
- Infra: @infra-team — 负责监控报警与工单自动化

---

## 7) Baseline 算法与参考实现（明确库与版本）🔧
> 推荐基线实现（MVP 阶段使用开源/可复现）

- **说话人嵌入**：ECAPA-TDNN（pyannote/espnet 实现，参考版本：2025 年主分支）
- **语音转换 / TTS**：YourTTS（one/few-shot，参考 repo）、VITS + HiFi-GAN（v2）作为高质量 TTS + vocoder
- **口型同步**：Wav2Lip（参考 release 2023+） + SyncNet 用作检测
- **人脸驱动**：First-Order-Motion-Model（Siarohin 2019）/ PIRenderer（更好的人脸一致性）
- **风格化/卡通化**：CartoonGAN / U-GAT-IT（参考实现）或 StyleGAN2/3 风格迁移 pipeline
- **高保真替换**：FaceShifter / SimSwap（temporal consistency 增强）
- **检测器**：XceptionNet-based deepfake detector（参考 deepfake-detection-challenge stacks）
- **工具栈**：PyTorch 2.0, FAISS (检索), Triton 或 TorchServe（推理服）

> 注：提交实现时请在 Design Spec 中记录确切 repo + commit/tag

---

## 8) 模型训练与评估细节🧪
- **评估指标**：MOS（主观）、PESQ、SI-SDR、speaker embedding cosine、LSE-C/LSE-D（sync）、FID/LPIPS（视觉）、ArcFace 相似度（如需保真）
- **切片分析**：按年龄/性别/语言/场景/光照分别报告指标
- **检测器目标**：ROC AUC ≥ 0.98，召回（recall）≥ 95% 在 FPR ≤ 1% 时
- **对抗鲁棒性**：对抗扰动、压缩（H.264 常见压缩档）和流媒体场景进行稳健性测试

---

## 9) MVP 范围细化（4–6 周）📦
- **功能边界（MVP）**：单角色场景（1 个被替换角色）、视频长度 ≤ 60s、分辨率 ≤ 1080p、批处理模式（不要求实时）、自动嵌入水印、法务初审通过后才允许发布。
- **不包含（MVP）**：多人场景复杂遮挡、多语种实时字幕、低带宽实时流媒体替换

---

## 10) 产能估算 & 成本（初步）💸
- **估算（示例）**：每条 30–60s 视频（MVP pipeline）平均 GPU 时间约 3–8 GPU-min（A100 等级），取决于视觉替换复杂度；若使用 spot/GPU 池，云成本约 $0.5–$3/视频（取决于批量与压缩）。
- **容量规划**：目标 10k 条/月 → 约 500–1000 GPU-hour/month，建议按峰值/平均分配资源并做 autoscaling

---

## 11) 分阶段上线与回滚策略🚦
- **阶段**：内测（0% 对外）→ Pilot（1% 随机用户）→ 扩展（5% → 20%）→ 全量（100%）
- **回滚触发条件**：
  - MOS 抽样平均下降 > 10% 或任一主要切片下降 > 15%
  - 水印可检出率 < 99%
  - 滥用/投诉率 > 0.1% 日均（或连续 3 天上升趋势）
  - 法律/合规投诉或临时禁令

---

## 12) 监控 & 报警 🔍
- **核心监控指标**：watermark detection rate, MOS (抽样), lip-sync alarm rate (SyncNet), model inference latency (p50/p95/p99), user takedown ratio
- **报警阈值（示例）**：watermark rate < 99% → 高优先级告警；lip-sync fail rate > 10% → 手动抽检；user takedown ratio > 0.1% → 自动限流并复核

---

## 13) 风险与缓解（务必列明）⚠️
- 隐私/肖像权风险 → 要求书面同意、记录 provenance、提供撤回路径
- 法律纠纷风险 → 法务预审、限制高风险用途、保留审计证据
- 品质/品牌风险 → 分级发布 + 人工抽检 + 退回/下线机制
- 被滥用风险 → 强化检测、封禁与法律应对流程

---

## 14) 交付物与负责人📋
- **交付物**：Research Design、数据清单与许可证明、Algorithm Design Spec、MVP 测试报告（含人工评估）、法务合规报告
- **负责人（示例）**：
  - Research Design Owner: @data-scientist-research-lead — research@example.com
  - Data Owner: @data-engineer — data@example.com
  - Legal Approver: @legal — legal@example.com
  - QA / Human Review: @data-scientist-evaluator — qa@example.com
  - Deployment / SRE: @infra-team — infra@example.com

### 15) 负责人分配表（待确认）
以下表格为当前建议的责任人分配占位，请确认 Owner、联系方式与目标完成日期后标注 Status 并通知团队。

| Deliverable | Owner | Contact | Target due date | Status |
|-------------|-------|---------|----------------:|:------:|
| Dataset inventory | @data-engineer | data@example.com | 2026-02-02 | TODO |
| Baseline repo & tags | @data-scientist-algorithm-designer | algo@example.com | 2026-02-03 | TODO |
| Watermark implementation | @infra-team | infra@example.com | 2026-02-05 | TODO |
| Consent verification & storage | @legal | legal@example.com | 2026-01-31 | TODO |
| Detection test plan | @data-scientist-evaluator | qa@example.com | 2026-02-04 | TODO |
| Cost & capacity plan | @infra-team | infra@example.com | 2026-02-03 | TODO |
| Final legal sign-off | @legal | legal@example.com | 2026-02-03 | TODO |

请各 Owner 在文档中更新各自负责项并在完成后将 Status 标为 DONE。
---

## 15) 批准与审阅流程
- 提交给：@data-scientist-tech-lead、@legal、@data-engineer、@data-scientist-evaluator
- 审阅周期：3 个工作日（初审）
- 文档变更需记录版本日志（见附录 B）

---

## 16) 验收检查表（TICK-BOX）✅
- [ ] 同意书模板已完成并签署样本已上传（Owner: Legal）
- [ ] Baseline 模型列表（含 repo + tag/commit）记录完毕（Owner: Algorithm Designer）
- [ ] 最低数据集满足多样性与数量要求（Owner: Data Owner）
- [ ] 水印方法与检测 API 已实现并 FN ≤ 1%（Owner: Infra/ML）
- [ ] 人工复核流程与触发条件已定义（Owner: QA）
- [ ] 法务签字（最终批准）已完成（Owner: Legal）

---

## 附录 A：同意书样板（示例）
> **同意书（短视频替换使用授权）**

- **被授权人（签名）**：
- **授权用途**：允许使用本人/素材进行短视频音色替换/卡通形象替换并用于下列渠道：_______
- **有效期**：自签署日起至 YYYY-MM-DD（建议默认 2 年）
- **撤销与撤回权**：被授权人可书面撤销，但已发布内容的下架与追溯按平台政策处理
- **其他条款**：禁止用于诈骗、政治宣传与其他非法用途；平台有权保留审计记录
- **签名**：电子签名 / 手写签名扫描件
- **存证**：上传到公司合规仓库并记录签署证据 ID

---

## 附录 B：watermark & provenance 技术要点
- **水印方法**：推荐采用 learned robust watermark（训练嵌入器使水印对常见压缩/噪声鲁棒），同时在频域做冗余校验。对高风险内容采用多重水印（频域 + 空间域）。
- **检测 API**：POST /internal/wm/check → 返回 {detected: true/false, confidence: 0-1, details: {type, payload}}。阈值：confidence < 0.6 需人工复核。
- **元数据内容**：{model_version, model_commit, data_snapshot_id, consent_id, generator_user_id, timestamp, signature}
- **审计存储**：使用不可变 append-only 日志（DB + 签名），并将关键证据哈希上链或写入可查证存储（视公司合规要求）。

---

**Last Updated**: 2026-01-27  
**Version**: v1.0


*注：上述数值与阈值为建议，请在法务与 infra 团队一起确认最终值并记录在 Algorithm Design Spec 中。*
