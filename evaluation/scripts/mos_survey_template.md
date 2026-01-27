# MOS Survey Template

说明：使用 Blind A/B 或单轮 MOS 评分，随机化样本顺序。

字段：
- sample_id
- slice (language/age/device)
- gen_clip_url
- ref_clip_url (optional)

指示：
1. 请在安静环境使用耳机完成听感打分。  
2. 对每个片段，请给出 1-5 的评分（1: 非常差，5: 非常好），并可附加文本反馈。  
3. 在口型一致性题项，请判断口型是否与语音同步（是/否/不确定）。

输出表格格式（CSV）：
- rater_id, sample_id, mos_score, lip_sync, comments, timestamp

注意：确保每个样本被至少 3 位独立评审打分以降低偏差。