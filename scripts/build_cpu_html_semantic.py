#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


def read_csv(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def series_by(rows, key_field, value_field, key_values, years, year_field="year"):
    lookup = {(r[year_field], r[key_field]): r for r in rows}
    out = []
    for key in key_values:
        data = []
        for year in years:
            row = lookup.get((year, key))
            value = float(row[value_field]) if row and row.get(value_field) not in (None, "") else None
            data.append(value)
        out.append({"name": key, "data": data})
    return out


def build_semantic(data_dir: Path):
    years = ["2021", "2022", "2023", "2024", "2025", "2026E", "2027E", "2028E", "2029E", "2030E"]

    bottleneck = read_csv(data_dir / "bottleneck_index.csv")
    tech = read_csv(data_dir / "tech_maturity.csv")
    segment = read_csv(data_dir / "segment_index.csv")
    cpu_cmp = read_csv(data_dir / "cpu_comparison_numeric.csv")
    ai = read_csv(data_dir / "ai_impact_index.csv")
    sustainability = read_csv(data_dir / "sustainability_index.csv")
    timeline = read_csv(data_dir / "timeline_events.csv")
    rd = read_csv(data_dir / "rd_talent_index.csv")

    b_series = series_by(
        bottleneck,
        "bottleneck",
        "index_0_100",
        ["HBM_tightness", "Advanced_packaging_capacity", "EUV_supply_pressure", "Export_control_severity"],
        years,
    )
    t_series = series_by(
        tech,
        "technology",
        "maturity_1_5",
        ["EUV_maturity", "GAA_adoption", "CXL_deployment", "UCIe_chiplet_interop", "Backside_power_delivery"],
        years,
    )
    s_series = series_by(
        segment,
        "segment",
        "perf_per_w_index",
        ["Server", "Mobile", "PC", "Edge", "AutoIndustrial"],
        years,
    )

    ai_server = [r for r in ai if r["segment"] == "Server" and r["ai_workload_type"] in ("Training", "Inference")]
    ai_series = series_by(ai_server, "ai_workload_type", "compute_intensity_index", ["Training", "Inference"], years)

    sus_series = series_by(
        sustainability,
        "region",
        "esg_compliance_index",
        ["Europe", "NorthAmerica", "Asia", "Global"],
        years,
    )

    rd_rows = [
        r
        for r in rd
        if r["sub_category"] in ("x86_Developer_Base", "ARM_Developer_Base", "RISC-V_Developer_Base")
    ]
    rd_series = series_by(
        rd_rows,
        "sub_category",
        "index_0_100",
        ["x86_Developer_Base", "ARM_Developer_Base", "RISC-V_Developer_Base"],
        years,
    )

    compare_rows = [r for r in cpu_cmp if r["year"] == "2025" and r["segment"] == "Server"]
    order = ["x86", "ARM", "RISC-V"]
    compare_rows.sort(key=lambda r: order.index(r["architecture"]) if r["architecture"] in order else 99)

    comparison_items = []
    for row in compare_rows:
        comparison_items.append(
            {
                "label": row["architecture"],
                "highlight": row["architecture"] == "ARM",
                "attributes": {
                    "性能/瓦": int(float(row["perf_per_w_index"])),
                    "平台生态": int(float(row["platform_ecosystem_index"])),
                    "供应韧性": int(float(row["supply_resilience_index"])),
                    "TCO效率": int(float(row["tco_efficiency_index"])),
                    "采用动量": int(float(row["adoption_momentum_index"])),
                },
            }
        )

    timeline_items = [{"date": row["time"], "event": row["description"]} for row in timeline]

    slides = [
        {
            "slide_id": 1,
            "slide_type": "title",
            "title": "CPU 行业十年展望（2021–2030）",
            "date": "2026-02-13",
            "content": ["近五年回顾与未来五年预测", "基于公开行业数据与情景化预测"],
        },
        {
            "slide_id": 2,
            "slide_type": "section_divider",
            "title": "第一部分｜核心摘要",
            "content": ["竞争核心从峰值性能转向性能/瓦、平台协同与供应链韧性"],
        },
        {
            "slide_id": 3,
            "slide_type": "content",
            "title": "结构性瓶颈迁移：HBM与先进封装压力上升",
            "assertion": "2025–2027E期间，HBM紧张度与先进封装能力成为主要约束，传统EUV与ABF压力边际缓解。",
            "insight": "短期需锁定HBM与先进封装产能，中期通过多源策略和设计冗余降低交付风险。",
            "components": {
                "callouts": [
                    {"label": "最高压力项", "text": "HBM_tightness在2026E达到高位，决定AI相关产品交付节奏。"},
                    {"label": "风险迁移", "text": "瓶颈由通用制造能力向先进封装与存储配套迁移。"},
                ]
            },
            "visual": {
                "type": "line_chart",
                "placeholder_data": {
                    "chart_config": {
                        "title": "关键瓶颈压力指数（0-100）",
                        "x_axis": years,
                        "series": b_series,
                        "y_axis_title": "压力指数",
                        "source": "数据源：bottleneck_index.csv",
                    }
                },
            },
        },
        {
            "slide_id": 4,
            "slide_type": "content",
            "title": "关键技术成熟度加速（2021–2030E）",
            "assertion": "EUV成熟度接近平台期，GAA、CXL与UCIe进入加速扩散阶段。",
            "insight": "建议将研发资源从“单点工艺追赶”转向“系统协同与生态就绪”。",
            "components": {
                "callouts": [{"label": "成熟度梯队", "text": "EUV > GAA/CXL/UCIe > Backside Power，技术扩散节奏分层明显。"}]
            },
            "visual": {
                "type": "line_chart",
                "placeholder_data": {
                    "chart_config": {
                        "title": "技术成熟度（1-5）",
                        "x_axis": years,
                        "series": t_series,
                        "y_axis_title": "成熟度评分",
                        "source": "数据源：tech_maturity.csv",
                    }
                },
            },
        },
        {
            "slide_id": 5,
            "slide_type": "content",
            "title": "产品形态性能/瓦演进：Server与Mobile领跑",
            "assertion": "Server与Mobile双线领跑性能/瓦，Edge与AutoIndustrial持续追赶。",
            "insight": "平台规划应按场景分层：Server优先TCO，Mobile优先能效与集成度。",
            "components": {},
            "visual": {
                "type": "line_chart",
                "placeholder_data": {
                    "chart_config": {
                        "title": "各细分场景性能/瓦指数",
                        "x_axis": years,
                        "series": s_series,
                        "y_axis_title": "性能/瓦指数",
                        "source": "数据源：segment_index.csv",
                    }
                },
            },
        },
        {
            "slide_id": 6,
            "slide_type": "section_divider",
            "title": "第二部分｜竞争与应用",
            "content": ["从架构竞争走向“架构 + 软件生态 + 供应韧性”的综合竞争"],
        },
        {
            "slide_id": 7,
            "slide_type": "content",
            "title": "2025 Server架构对比：ARM综合动量领先",
            "assertion": "在Server场景中，ARM在性能/瓦、TCO效率与采用动量上领先，x86在生态深度仍具优势。",
            "insight": "组合策略优于单架构押注：核心负载维持x86，增量AI负载加速ARM布局。",
            "components": {
                "comparison_items": comparison_items,
                "callouts": [{"label": "推荐方向", "text": "新增算力优先评估ARM平台，保留x86生态兼容与迁移窗口。"}],
            },
            "visual": {"type": "none", "placeholder_data": {}},
        },
        {
            "slide_id": 8,
            "slide_type": "content",
            "title": "AI工作负载驱动：Server训练/推理双升",
            "assertion": "Server端训练与推理计算强度持续上升，推理增长斜率在后期更快。",
            "insight": "产品路线需同步优化算力密度、内存带宽与软件栈成熟度。",
            "components": {
                "kpis": [
                    {"label": "2030E训练强度", "value": "98", "unit": "/100", "delta": "2021→2030E +23"},
                    {"label": "2030E推理强度", "value": "93", "unit": "/100", "delta": "2021→2030E +28"},
                ]
            },
            "visual": {
                "type": "line_chart",
                "placeholder_data": {
                    "chart_config": {
                        "title": "Server AI计算强度指数",
                        "x_axis": years,
                        "series": ai_series,
                        "y_axis_title": "计算强度指数",
                        "source": "数据源：ai_impact_index.csv",
                    }
                },
            },
        },
        {
            "slide_id": 9,
            "slide_type": "content",
            "title": "可持续性进展：区域ESG合规持续抬升",
            "assertion": "Europe与NorthAmerica保持高位，Asia与Global持续改善，区域差异逐步收敛。",
            "insight": "建议将ESG合规指标纳入供应商分层准入与长期采购合同。",
            "components": {},
            "visual": {
                "type": "line_chart",
                "placeholder_data": {
                    "chart_config": {
                        "title": "ESG合规指数（0-100）",
                        "x_axis": years,
                        "series": sus_series,
                        "y_axis_title": "ESG合规指数",
                        "source": "数据源：sustainability_index.csv",
                    }
                },
            },
        },
        {
            "slide_id": 10,
            "slide_type": "content",
            "title": "关键事件时间线（2021–2030E）",
            "assertion": "行业主线从“供给紧张”过渡到“系统级优化”，政策与技术演进交织推进。",
            "insight": "建立“事件-指标-动作”闭环，按季度更新情景假设并校准投资优先级。",
            "components": {
                "timeline_items": timeline_items,
                "callouts": [{"label": "节奏变化", "text": "2026E后进入规模化部署与互操作优化阶段。"}],
            },
            "visual": {"type": "none", "placeholder_data": {}},
        },
        {
            "slide_id": 11,
            "slide_type": "content",
            "title": "研发与人才生态：RISC-V开发者基数快速提升",
            "assertion": "ARM与RISC-V开发者生态扩张显著，x86开发者基数相对稳定。",
            "insight": "人才策略应从“单架构深耕”转向“跨架构能力组合”。",
            "components": {},
            "visual": {
                "type": "line_chart",
                "placeholder_data": {
                    "chart_config": {
                        "title": "开发者基数指数（0-100）",
                        "x_axis": years,
                        "series": rd_series,
                        "y_axis_title": "开发者基数",
                        "source": "数据源：rd_talent_index.csv",
                    }
                },
            },
        },
        {
            "slide_id": 12,
            "slide_type": "content",
            "title": "行动清单（6-36个月）",
            "assertion": "应以“产能保障 + 架构组合 + 软件生态 + ESG治理”四线并进推进执行。",
            "insight": "将关键动作纳入季度经营评审，绑定量化指标与责任人。",
            "components": {
                "decisions": [
                    {
                        "title": "锁定关键瓶颈资源",
                        "priority": "P1",
                        "description": "围绕HBM与先进封装签订中长期保障协议，设置弹性采购机制。",
                        "owner": "供应链负责人",
                        "timeline": "0-6个月",
                    },
                    {
                        "title": "推进双架构策略",
                        "priority": "P1",
                        "description": "在Server增量负载上优先验证ARM，同时保留x86核心业务兼容路径。",
                        "owner": "产品与平台负责人",
                        "timeline": "6-18个月",
                    },
                    {
                        "title": "强化异构软件栈",
                        "priority": "P2",
                        "description": "投入编译器、调度器与运维工具链，降低迁移与运维成本。",
                        "owner": "软件架构负责人",
                        "timeline": "6-24个月",
                    },
                    {
                        "title": "建立ESG与合规闭环",
                        "priority": "P2",
                        "description": "将ESG和合规KPI纳入供应商治理，形成季度审查与纠偏机制。",
                        "owner": "运营与合规负责人",
                        "timeline": "12-36个月",
                    },
                ],
                "action_items": [
                    {"text": "建立季度情景更新机制（基准/压力/乐观三情景）", "owner": "战略团队", "deadline": "每季度"},
                    {"text": "发布跨架构人才培养计划并绑定里程碑", "owner": "人力与研发", "deadline": "Q2"},
                    {"text": "完善关键物料风险看板并接入预警阈值", "owner": "供应链团队", "deadline": "Q1"},
                ],
            },
            "visual": {"type": "none", "placeholder_data": {}},
        },
    ]

    return {"deck_title": "CPU 行业十年展望报告（2021–2030）", "slides": slides}


def main():
    parser = argparse.ArgumentParser(description="Build semantic JSON for CPU HTML slides")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    semantic = build_semantic(Path(args.data_dir))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(semantic, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote semantic: {out}")
    print(f"Slides: {len(semantic['slides'])}")


if __name__ == "__main__":
    main()
