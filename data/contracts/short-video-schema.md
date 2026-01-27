# Short Video Data Schema (short-video)

说明：此 schema 用于语音与视频样本入库的最小字段定义。

必填字段：
- file_id: string (唯一标识)
- speaker_id: string
- length_s: float (秒)
- sample_rate: int
- channels: int
- language: string (e.g., zh-CN, en-US)
- consent_id: string (关联同意文档)
- timestamp: ISO-8601 datetime (UTC)
- checksum: string (sha256)
- source: string (采集来源，例如: upload, s3)
- data_snapshot_id: string (数据快照 id)

可选字段：
- emotion_tag: string (neutral, happy, sad, angry, etc.)
- device: string (mobile, desktop, studio)
- mic_type: string
- ambient_noise_db: float

示例 JSON 记录：

{
  "file_id": "audio_0001",
  "speaker_id": "spk_123",
  "length_s": 32.4,
  "sample_rate": 24000,
  "channels": 1,
  "language": "zh-CN",
  "consent_id": "consent_20260127_01",
  "timestamp": "2026-01-20T12:34:56Z",
  "checksum": "3a7bd3...",
  "source": "upload",
  "data_snapshot_id": "snapshot_20260127_v1",
  "emotion_tag": "neutral",
  "device": "mobile",
  "ambient_noise_db": 22.5
}

注：策略：入库前必须通过 schema 校验，任何缺失必填字段将被拒绝入库并产生工单。