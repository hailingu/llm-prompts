# Issue: Data Quality Remediation for Short-Video Replacement

**Summary**: 实施数据质量与管道改进（schema 校验、GE expectations、入库清洗、consent 绑定、监控告警、CI 测试）。

**Owners**:
- Data Owner: @data-engineer (data@example.com)
- QA: @data-scientist-evaluator (qa@example.com)
- Infra: @infra-team (infra@example.com)
- Legal: @legal (legal@example.com)

**Tasks**:
- [ ] 实现 `data/contracts/short-video-schema.md` 的 schema 校验（入库前） — @data-engineer
- [ ] 集成 `data/tests/great_expectations/short_video_expectations.yml` 到 CI，nightly run — @data-engineer
- [ ] 清洗脚本（去重、SNR 检测、时区标准化） — @data-engineer
- [ ] Consent 绑定流程与隔离队列实现（未绑定触发工单） — @data-engineer + @legal
- [ ] 监控面板（missing_consent_pct、invalid_audio_pct、low_quality_pct） — @infra-team
- [ ] 人工抽检计划与 QA 报告模板 — @data-scientist-evaluator

**Acceptance Criteria**:
- 100% 新数据走 schema 校验；失败样本进入隔离并触发工单
- Consent 100% 绑定或进入待审状态
- CI 包含 GE checks，nightly run 无关键失败
- 监控阈值与告警生效

**Notes**:
- 参照 `research-designs/short-video-replacement-v1.md` 中的 Data Quality & Pipeline 小节

**Suggested due date**: 2026-02-05
