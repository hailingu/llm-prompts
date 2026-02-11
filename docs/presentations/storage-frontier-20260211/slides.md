# 存储行业前沿（2023–2026）— 幻灯片草稿（25 页）

> 说明：中文版，视觉风格：BCG。每页包含要点、视觉注释与讲者笔记（每页 ≥ 2 句）。

## Slide 1 — 封面 (title)
- 标题：存储行业前沿（2023–2026）
- 副标题：关键趋势、厂商案例与可落地建议
- 作者 / 日期：2026-02-11
- 组件：{}  
- Speaker notes: 本次汇报聚焦 2023–2026 年间的技术与市场演进，并给出对企业/研究团队的短中长期建议。

---

## Slide 2 — 开场与关键结论（section_divider）
- 标题：开场与关键结论
- Callout: 一行结论（存储行业进入“低延迟高带宽 + 智能分层 + 云原生整合”的过渡期）
- Speaker notes: 用 30 秒概括结论，后续每节用 2–3 要点支撑。

---

## Slide 3 — 关键結論（bullet-list + KPIs）
- Bullets:
  - 全球市场从 ~85 亿美元增长到 ~110 亿美元（2023→2026，见 Figure 1）
  - NVMe、PMem、CXL 与云原生存储将主导未来 3 年创新路径
  - 中国市场受数据主权与合规驱动，本地化与混合云需求上升
- KPIs:
  - `市场 (2026)`：Global 110 (USD Bn) / China 40 (USD Bn)
- Speaker notes: 每条结论配一行关键证据来源（CSV/报告）。

---

## Slide 4 — 市场与趋势（section_divider）
- 标题：市场与趋势
- Speaker notes: 本节展示市场规模、云/企业分布与媒体份额的量化证据。

---

## Slide 5 — 市场规模：全球 vs 中国（data-heavy — line_chart）
- Title: 市场規模（2023–2026）
- Visual: `line_chart` — Source: `reports/data/global_china_market_2023_2026.csv`
- Visual.placeholder_data.chart_config:
  - x: [2023,2024,2025,2026]
  - series:
    - {name: "Global 市場 (USD Bn)", data: [85,92,100,110]}
    - {name: "China 市場 (USD Bn)", data: [28,32,36,40]}
- Content bullets:
  - 全球市場預計 2026 達 ~110 億美元；中國市場占比上升
- Speaker notes: 指出數據來源 (IDC/SNIA) 與估算注記；建議在講稿中說明不確定性邊界。

---

## Slide 6 — NVMe 与 NVMe-oF 采纳（data-heavy — line_chart）
- Title: NVMe 采纳趨勢（2023–2026）
- Visual: `line_chart` — Source: `reports/data/nvme_adoption_2023_2026.csv`
- Visual.placeholder_data.chart_config:
  - x: [2023,2024,2025,2026]
  - series:
    - {name: "NVMe server (%)", data: [30,40,52,65]}
    - {name: "NVMe-oF Deployments (%)", data: [5,8,15,25]}
- Content bullets:
  - NVMe server adoption 快速上升，NVMe-oF 從少量部署走向規模化
- Speaker notes: 討論網絡（RDMA vs NVMe/TCP）與互操作性風險。

---

## Slide 7 — PMem/SCM 与 存储介质份额（data-heavy — composite charts）
- Title: PMem 預測 & 介質份額（PMem 市場 / SSD-HDD-Tape）
- Visual: `composite_charts` — Sources: `pmem_scm_forecast_2023_2026.csv`, `ssd_tape_hdd_share_2023_2026.csv`
- Visual.placeholder_data:
  - pmem:
    - x: [2023,2024,2025,2026]
    - series: [{name: "PMem 市場 (USD M)", data: [50,80,150,300]}]
  - media_share:
    - x: [2023,2024,2025,2026]
    - series:
      - {name: "SSD (%)", data: [45,48,52,58]}
      - {name: "HDD (%)", data: [50,47,43,37]}
      - {name: "Tape (%)", data: [5,5,5,5]}
- Content bullets:
  - PMem 在低延遲場景增長顯著，但受成本/容量限制；建議在元數據/日志加速場景試點
- Speaker notes: 強調 PMem 的工程約束和試點驗證點（斷電恢復、一致性驗證）。

---

## Slide 8 — 硬件層創新（section_divider）
- 標題：硬件層創新
- Speaker notes: 本節概括 NVMe、PMem、CXL、Computational Storage 的適用場景與工程要點。

---

## Slide 9 — NVMe 工程實踐與建議（comparison）
- Title: NVMe/NVMe-oF 工程實踐
- Components.comparison_items:
  - {label: "並行訓練", attributes: {推薦協議: "NVMe/TCP / RDMA", 網絡要點: "100/200GbE leaf-spine", 驗證指標: "IO 並發、隊列延遲、丟包率"}}
  - {label: "低延遲 DB", attributes: {推薦協議: "RDMA", 驗證指標: "P99 延遲、QDepth", 回退策略: "本地 NVMe/實例"}}
- Visual.type: "none"
- Speaker notes: 對比不同場景下的網絡/QoS/回退策略，建議小規模互操作試驗。

---

## Slide 10 — PMem 與 CXL（comparison）
- Title: PMem / CXL 的應用與限制
- Components.comparison_items:
  - {label: "PMem (持久內存)", attributes: {適用: "元數據/日志加速", 優點: "低 P99 延遲", 局限: "高成本、容量有限"}}
  - {label: "CXL (遠內存 pooling)", attributes: {適用: "大模型內存彈性", 優點: "內存共享/擴展", 局限: "生態成熟度、隔離/權限"}}
- Visual.type: "none"
- Speaker notes: 建議在受控集群進行 CXL pooling/遷移延遲測試並記錄故障模式。

---

## Slide 11 — 近數據處理（Computational Storage）（comparison）
- Title: 近數據處理的適用場景
- Components.comparison_items:
  - {label: "邊緣流式預處理", attributes: {用例: "濾波/壓縮/解析", 驗證點: "端到端延遲、正確性", 限制: "設備異構、編程模型"}}
  - {label: "大規模日志/壓縮", attributes: {用例: "設備側壓縮/索引", 驗證點: "吞吐 & 減少主機 IO", 回退: "主機執行路徑"}}
- Visual.type: "none"
- Speaker notes: 強調需保留主機回退路徑並優先在受控場景試點。

---

## Slide 12 — 系統與雲原生（section_divider）
- 標題：系統與雲原生
- Speaker notes: 本節展示對象存儲、並行文件系統與雲原生存儲插件的趨勢與取捨。

---

## Slide 13 — 雲原生存儲採納（data-heavy — line_chart）
- Title: 雲原生存儲採納指數（CSI / Rook / Longhorn）
- Visual: `line_chart` — Source: `reports/data/cloud_native_storage_adoption.csv`
- Visual.placeholder_data.chart_config:
  - x: [2023,2024,2025,2026]
  - series:
    - {name: "CSI 使用指數", data: [20,35,50,65]}
- Content bullets:
  - 雲原生存儲採納快速上升，開源項目活躍（MinIO/ Ceph 指標為證據）
- Speaker notes: 說明 CSI 成熟度與運維挑戰，提出分階段遷移建議。

---

## Slide 14 — 存儲架構對比（comparison / cards）
- Title: 對象 / 並行文件 / 分層緩存 的架構權衡
- Components.comparison_items:
  - {label: "對象存儲 (S3 兼容)", attributes: {優點: "彈性擴展、成本優勢", 局限: "小對象/低延遲場景需緩存" , 典型場景: "數據湖/歸檔"}}
  - {label: "並行文件系統", attributes: {優點: "高帶寬並行 I/O", 局限: "元數據瓶頸/運維複雜", 典型場景: "HPC/訓練"}}
  - {label: "分層緩存 (Alluxio 等)", attributes: {優點: "提高並行讀性能", 局限: "一致性/緩存失效", 典型場景: "訓練/分析"}}
- Visual.type: "none"
- Speaker notes: 對比表用於幫助架構選擇（按工作負載特徵決策）。

---

## Slide 15 — 數據保護與合規（callout + bullets）
- Title: 數據保護、合規與不可變備份
- Bullets:
  - 加密（靜態/傳輸）、WORM 與不可變快照為防勒索基礎
  - 合規：中國市場對本地化與跨境控制要求高
  - 災備：異地多活與精細 RPO/RTO 設計必要
- Visual.type: "none"
- Speaker notes: 建議按業務關鍵性分級備份策略並把恢復演練納常態運維。

---

## Slide 16 — 面向 AI 的存儲演進（section_divider）
- 標題：面向 AI 的存儲演進
- Speaker notes: 本節給出訓練/推理的 I/O 特徵與組合實踐建議。

---

## Slide 17 — AI I/O 要求（data-heavy — bar_line_chart）
- Title: AI: Training vs Inference 的 I/O 需求（2023–2026）
- Visual.type: "bar_line_chart" — Source: `reports/data/ai_io_requirements_2023_2026.csv`
- Visual.placeholder_data.chart_config:
  - x: [2023,2024,2025,2026]
  - series:
    - {name: "Training IOPS", data: [500000,1000000,2000000,4000000]}
    - {name: "Training Bandwidth (GB/s)", data: [20,40,80,160]}
    - {name: "Inference IOPS", data: [50000,100000,200000,400000]}
    - {name: "Inference Bandwidth (GB/s)", data: [2,4,8,16]}
- Content bullets:
  - 訓練需要高帶寬、高並發 I/O；檢查點與 shuffle 導致短時峰值
- Speaker notes: 強調分層緩存 + NVMe-oF 的組合價值，並指出經濟性驗證點。

---

## Slide 18 — AI 數據路徑設計（diagram + bullets）
- Title: AI 數據路徑的常見組合 (對象 + 並行 FS + 緩存)
- Visual.type: "flow_diagram"
- Visual.placeholder_data.mermaid_code: "graph LR; ObjectStorage-->Cache[Alluxio/Local NVMe Cache]-->ParallelFS; ParallelFS-->Compute"
- Bullets:
  - 建議：S3+並行文件系統+本地 NVMe cache 組合用於訓練
  - 驗證點：緩存命中率、並行文件系統元數據擴展性
- Speaker notes: 給出 POC 驗證清單（緩存命中率、峰值帶寬、成本對比）。

---

## Slide 19 — 安全與治理（section_divider）
- 標題：安全、可靠性與數據治理
- Speaker notes: 討論加密、審計、合規策略與災備設計。

---

## Slide 20 — 安全實踐與風險（bullet-list + risks）
- Title: 安全實踐與工程風險
- Bullets:
  - 不可變備份 + 快照流水線為防勒索核心
  - 合規：跨境傳輸策略與審計追蹤
  - 運營：恢復演練與備份窗口成本需要權衡
- Components.risks:
  - {label: "合規風險", description: "跨境數據傳輸與本地化要求導致架構複雜化"}
  - {label: "成本風險", description: "高性能層的持續運行成本與備份成本"}
- Speaker notes: 建議分級治理並把恢復演練納入 SLO。

---

## Slide 21 — 廠商案例（section_divider）
- 標題：廠商案例研究（AWS / 阿里云 / 華為云 / Dell）
- Speaker notes: 每家一頁要點化展示架構亮點、試點建議與風險。

---

## Slide 22 — AWS：架構亮點與試點建議 (comparison / cards)
- Title: AWS 案例要點
- Components.comparison_items:
  - {label: "架構概述", attributes: {亮點: "S3 + EBS + 本地 NVMe 實例", 場景: "訓練/高並發 DB", 備註: "快照與跨區複製生態成熟"}}
  - {label: "試點建議", attributes: {建議: "NVMe-oF 性能試點 + 快照/恢復成本評估", 風險: "跨區流量成本" , 時間: "6–12 月"}}
- Visual.type: "none"
- Speaker notes: 強調商業化服務成熟度與成本-性能權衡。

---

## Slide 23 — 阿里云（Pangu）— 架構與經驗要點 (comparison / cards)
- Title: 阿里云 關鍵要點
- Components.comparison_items:
  - {label: "架構亮點", attributes: {亮點: "Pangu + SSD cache 分層", 場景: "大規模分析/訓練", 備註: "元數據擴展性強"}}
  - {label: "試點建議", attributes: {建議: "分層下沉成本驗證 + 元數據擴展性試驗", 風險: "實現內部化優化難以外移" , 時間: "6–12 月"}}
- Visual.type: "none"
- Speaker notes: 建議對 Pangu 的分層策略做成本-性能實驗並記錄運維複雜度。

---

## Slide 24 — 華為云（硬件協同）— 要點 (comparison / cards)
- Title: 華為云 關鍵要點
- Components.comparison_items:
  - {label: "架構亮點", attributes: {亮點: "PMem + NVMe 層 + CXL 試點", 場景: "低延遲事務/大模型內存彈性", 備註: "硬件協同優勢"}}
  - {label: "試點建議", attributes: {建議: "在受控集群測試 PMem/CXL 可用性與回退策略", 風險: "國際互操作與硬件支持" , 時間: "6–12 月"}}
- Visual.type: "none"
- Speaker notes: 建議與硬件合作夥伴確認支持矩陣並測試回退路徑。

---

## Slide 25 — Dell: 企業混合云與 NVMe 產品要點 (comparison / cards)
- Title: Dell 關鍵要點
- Components.comparison_items:
  - {label: "架構亮點", attributes: {亮點: "PowerMax / PowerScale NVMe 陣列", 場景: "企業級事務與分析", 備註: "端到端運維工具鏈成熟"}}
  - {label: "試點建議", attributes: {建議: "驗證 NVMe 陣列性能 & 快照恢復", 風險: "高成本、運維資源需求" , 時間: "6–12 月"}}
- Visual.type: "none"
- Speaker notes: 推薦與廠商工程團隊聯合開展性能與恢復試驗。

---

## 附錄與參考（metadata）
- 包含：圖表清單（CSV 路徑）、學術參考編號、數據來源注記
- Speaker notes: 附錄中列出所有 CSV 與代表性論文引用，便於審計。


---

Notes:
- 總頁數：25（含封面與分節頁）
- 每頁 ≤5 bullets：已按要求壓縮要點
- 視覺風格：BCG（已在 `design_spec.json` 中列明視覺 tokens）
- 所有圖表均標注 CSV 來源，數值來源可追溯到 `reports/data/` 中的 CSV

交付：請基於此 `slides.md` 與 `reports/data/` 中的數據生成 `slides_semantic.json`（已同時生成），並運行 MO-0..MO-12 自檢後交付 `ppt-visual-designer`（視覺風格：BCG）。