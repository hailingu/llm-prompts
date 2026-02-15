#!/usr/bin/env python3
import argparse
import html
import json
from pathlib import Path


KPMG = {
    "blue": "#00338D",
    "light_blue": "#0091DA",
    "purple": "#483698",
    "teal": "#00A3A1",
    "pink": "#E31C79",
    "bg": "#F7F9FC",
    "text": "#1F2937",
    "muted": "#6B7280",
    "line": "#E5E7EB",
}


def esc(value):
    if value is None:
        return ""
    return html.escape(str(value))


def color_for_index(index):
    colors = [KPMG["blue"], KPMG["light_blue"], KPMG["purple"], KPMG["teal"], KPMG["pink"]]
    return colors[index % len(colors)]


def base_head(title):
    return f"""<!DOCTYPE html>
<html lang=\"zh-CN\">
<head>
  <meta charset=\"UTF-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
  <title>{esc(title)}</title>
  <link rel=\"stylesheet\" href=\"https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css\" />
  <link rel=\"stylesheet\" href=\"https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@300;400;500;700;900&display=swap\" />
  <script src=\"https://cdn.jsdelivr.net/npm/chart.js\"></script>
  <script type=\"module\">
    import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
    mermaid.initialize({{ startOnLoad: true, theme: 'default', securityLevel: 'loose' }});
  </script>
  <style>
    :root {{
      --kpmg-blue: {KPMG['blue']};
      --kpmg-light-blue: {KPMG['light_blue']};
      --kpmg-purple: {KPMG['purple']};
      --kpmg-teal: {KPMG['teal']};
      --kpmg-pink: {KPMG['pink']};
      --line: {KPMG['line']};
      --text: {KPMG['text']};
      --muted: {KPMG['muted']};
      --bg: {KPMG['bg']};
    }}
    body {{ margin:0; padding:0; background:#eaeef3; font-family:'Noto Sans SC',sans-serif; }}
    .slide {{ width:1280px; height:720px; margin:0 auto; background:#fff; display:flex; flex-direction:column; overflow:hidden; }}
    .topbar {{ height:8px; background:var(--kpmg-blue); }}
    .header {{ padding:26px 56px 16px; border-bottom:1px solid var(--line); display:flex; align-items:flex-end; justify-content:space-between; }}
    .title {{ font-size:34px; line-height:1.2; font-weight:800; color:var(--kpmg-blue); }}
    .subtitle {{ font-size:12px; letter-spacing:.08em; color:var(--muted); text-transform:uppercase; font-weight:700; }}
    .body {{ flex:1; background:var(--bg); padding:20px 56px; display:flex; gap:20px; }}
    .card {{ background:#fff; border:1px solid var(--line); border-radius:12px; padding:14px; }}
    .label {{ font-size:12px; color:var(--muted); }}
    .small {{ font-size:12px; color:var(--muted); }}
    .footer {{ border-top:1px solid var(--line); padding:10px 56px; font-size:11px; color:var(--muted); display:flex; justify-content:space-between; }}
    .pill {{ display:inline-block; font-size:11px; padding:2px 8px; border-radius:999px; background:#EDF4FF; color:var(--kpmg-blue); font-weight:700; }}
    .metric-value {{ font-size:28px; line-height:1; color:var(--kpmg-blue); font-weight:800; }}
    .mermaid-wrap {{ background:#fff; border:1px solid var(--line); border-radius:12px; padding:10px; overflow:auto; }}
    .flow-step {{ min-width:160px; border-radius:10px; padding:10px 12px; background:#fff; border:1px solid var(--line); }}
    .flow-arrow {{ font-size:18px; color:var(--kpmg-light-blue); font-weight:700; }}
  </style>
</head>"""


def render_cover(slide, deck_title):
    title = slide.get("title") or deck_title
    content = slide.get("content") or []
    subtitle = content[0] if content else ""
    extra = content[1] if len(content) > 1 else ""
    return f"""
<body>
  <div class=\"slide\" style=\"display:flex;flex-direction:row;background:#fff;\">
    <div style=\"width:340px;background:{KPMG['blue']};color:#fff;position:relative;padding:42px 32px;\">
      <div style=\"font-size:12px;opacity:.7;letter-spacing:.12em;text-transform:uppercase;\">KPMG Style Deck</div>
      <div style=\"position:absolute;left:32px;bottom:42px;\">
        <div style=\"font-size:24px;font-weight:800;\">CPU 行业报告</div>
        <div style=\"height:4px;width:72px;background:{KPMG['light_blue']};margin-top:12px;\"></div>
      </div>
    </div>
    <div style=\"flex:1;padding:72px 72px;display:flex;flex-direction:column;\">
      <div class=\"pill\" style=\"width:max-content;\">Industry Outlook</div>
      <h1 style=\"margin-top:18px;font-size:56px;line-height:1.1;font-weight:900;color:{KPMG['blue']};\">{esc(title)}</h1>
      <p style=\"margin-top:14px;font-size:28px;color:#4B5563;\">{esc(subtitle)}</p>
      <p style=\"margin-top:8px;font-size:20px;color:#6B7280;\">{esc(extra)}</p>
      <div style=\"margin-top:auto;color:#9CA3AF;font-size:14px;\">{esc(slide.get('date') or '2026-02-12')}</div>
    </div>
  </div>
</body>
</html>
"""


def render_section_divider(slide, page_no, total):
    content = (slide.get("content") or [""])[0]
    return f"""
<body>
  <div class=\"slide\" style=\"background:{KPMG['blue']};color:#fff;\">
    <div style=\"flex:1;display:flex;flex-direction:column;justify-content:center;align-items:center;padding:0 120px;text-align:center;\">
      <div style=\"font-size:54px;font-weight:900;line-height:1.15;\">{esc(slide.get('title'))}</div>
      <div style=\"margin-top:24px;font-size:28px;color:#D1E8FF;\">{esc(content)}</div>
    </div>
    <div class=\"footer\" style=\"border-top:1px solid rgba(255,255,255,.2);color:#D1D5DB;background:transparent;\">
      <div>CPU 行业十年展望</div><div>{page_no} / {total}</div>
    </div>
  </div>
</body>
</html>
"""


def render_kpis(kpis):
    if not kpis:
        return ""
    cards = []
    for idx, item in enumerate(kpis):
        cards.append(
            f"""
<div class=\"card\" style=\"flex:1;border-top:4px solid {color_for_index(idx)};\">
  <div class=\"label\">{esc(item.get('label'))}</div>
  <div class=\"metric-value\">{esc(item.get('value'))}{esc(item.get('unit') or '')}</div>
  <div class=\"small\" style=\"margin-top:6px;\">{esc(item.get('delta') or '')}</div>
</div>
"""
        )
    return f"<div style=\"display:flex;gap:12px;\">{''.join(cards)}</div>"


def render_callouts(callouts):
    if not callouts:
        return ""
    blocks = []
    for idx, c in enumerate(callouts):
        label = c.get("label") or "要点"
        text = c.get("text") or ""
        blocks.append(
            f"""
<div class=\"card\" style=\"border-left:4px solid {color_for_index(idx)};\">
  <div class=\"label\" style=\"font-weight:700;color:{KPMG['blue']};\">{esc(label)}</div>
  <div style=\"margin-top:6px;font-size:14px;color:{KPMG['text']};line-height:1.5;\">{esc(text)}</div>
</div>
"""
        )
    return "".join(blocks)


def render_decisions(items):
    if not items:
        return ""
    blocks = []
    for idx, item in enumerate(items):
        blocks.append(
            f"""
<div class=\"card\" style=\"border-top:4px solid {color_for_index(idx)};\">
  <div style=\"display:flex;justify-content:space-between;gap:10px;align-items:flex-start;\">
    <div style=\"font-size:20px;font-weight:800;color:{KPMG['blue']};\">{esc(item.get('title'))}</div>
    <span class=\"pill\">{esc(item.get('priority') or 'P1')}</span>
  </div>
  <div style=\"margin-top:8px;font-size:14px;line-height:1.6;color:{KPMG['text']};\">{esc(item.get('description') or '')}</div>
  <div style=\"margin-top:10px;font-size:12px;color:{KPMG['muted']};\">Owner: {esc(item.get('owner') or '')} · Timeline: {esc(item.get('timeline') or '')}</div>
</div>
"""
        )
    return f"<div style=\"display:grid;grid-template-columns:1fr;gap:12px;\">{''.join(blocks)}</div>"


def render_comparison(items):
    if not items:
        return ""
    cards = []
    for idx, item in enumerate(items):
        attrs = item.get("attributes") or {}
        rows = "".join(
            f"<div style=\"display:flex;justify-content:space-between;gap:8px;border-top:1px dashed #e5e7eb;padding-top:6px;margin-top:6px;\"><span class=\"small\">{esc(k)}</span><span style=\"font-size:13px;color:{KPMG['text']};font-weight:600;text-align:right;\">{esc(v)}</span></div>"
            for k, v in attrs.items()
        )
        highlight = item.get("highlight")
        cards.append(
            f"""
<div class=\"card\" style=\"border:{'2px solid ' + KPMG['light_blue'] if highlight else '1px solid ' + KPMG['line']};\">
  <div style=\"display:flex;justify-content:space-between;align-items:center;\">
    <div style=\"font-size:22px;font-weight:800;color:{KPMG['blue']};\">{esc(item.get('label'))}</div>
    {('<span class=\"pill\">推荐</span>' if highlight else '')}
  </div>
  {rows}
</div>
"""
        )
    return f"<div style=\"display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;\">{''.join(cards)}</div>"


def render_risks(items):
    if not items:
        return ""
    header = """
<div class=\"card\" style=\"padding:0;overflow:hidden;\">
  <table style=\"width:100%;border-collapse:collapse;font-size:13px;\">
    <thead><tr style=\"background:#EEF4FF;color:#1D4ED8;text-align:left;\"><th style=\"padding:10px;\">风险</th><th style=\"padding:10px;\">概率</th><th style=\"padding:10px;\">影响</th><th style=\"padding:10px;\">缓释</th><th style=\"padding:10px;\">责任人</th></tr></thead>
    <tbody>
"""
    rows = []
    for item in items:
        rows.append(
            f"<tr style=\"border-top:1px solid #E5E7EB;\"><td style=\"padding:10px;font-weight:600;\">{esc(item.get('risk'))}</td><td style=\"padding:10px;\">{esc(item.get('probability'))}</td><td style=\"padding:10px;\">{esc(item.get('impact'))}</td><td style=\"padding:10px;\">{esc(item.get('mitigation'))}</td><td style=\"padding:10px;\">{esc(item.get('owner'))}</td></tr>"
        )
    return header + "".join(rows) + "</tbody></table></div>"


def render_timeline(items):
    if not items:
        return ""
    nodes = []
    for idx, item in enumerate(items):
        nodes.append(
            f"""
<div class=\"card\" style=\"min-width:220px;border-top:4px solid {color_for_index(idx)};\">
  <div class=\"label\">{esc(item.get('date') or item.get('time') or '')}</div>
  <div style=\"margin-top:6px;font-size:15px;color:{KPMG['text']};font-weight:700;\">{esc(item.get('event') or item.get('title') or item.get('text') or '')}</div>
</div>
"""
        )
    return f"<div style=\"display:flex;gap:10px;overflow:auto;padding-bottom:4px;\">{''.join(nodes)}</div>"


def render_actions(items):
    if not items:
        return ""
    rows = []
    for idx, item in enumerate(items):
        rows.append(
            f"""
<div class=\"card\" style=\"display:flex;justify-content:space-between;align-items:flex-start;gap:12px;border-left:4px solid {color_for_index(idx)};\">
  <div>
    <div style=\"font-size:16px;color:{KPMG['text']};font-weight:700;\">{esc(item.get('text'))}</div>
    <div class=\"small\" style=\"margin-top:6px;\">Owner: {esc(item.get('owner') or '')}</div>
  </div>
  <span class=\"pill\">{esc(item.get('deadline') or '')}</span>
</div>
"""
        )
    return "".join(rows)


def render_flow_diagram(flow_data):
    if not flow_data:
        return ""
    steps = flow_data.get("steps") or []
    transitions = {(t.get("from"), t.get("to")): t for t in (flow_data.get("transitions") or [])}
    chunks = []
    for idx, step in enumerate(steps):
        chunks.append(f"<div class=\"flow-step\"><div class=\"label\">STEP {idx+1}</div><div style=\"font-weight:700;color:{KPMG['text']};margin-top:4px;\">{esc(step.get('label'))}</div></div>")
        if idx < len(steps) - 1:
            t = transitions.get((step.get("id"), steps[idx + 1].get("id")), {})
            note = esc(t.get("label") or t.get("condition") or "")
            chunks.append(f"<div style=\"display:flex;flex-direction:column;align-items:center;justify-content:center;min-width:72px;\"><div class=\"flow-arrow\">→</div><div class=\"small\">{note}</div></div>")
    return f"<div class=\"card\" style=\"overflow:auto;\"><div style=\"display:flex;align-items:stretch;gap:8px;\">{''.join(chunks)}</div></div>"


def render_mermaid(mermaid_code):
    if not mermaid_code:
        return ""
    return f"""
<div class=\"mermaid-wrap\">
  <div class=\"mermaid\">{esc(mermaid_code)}</div>
</div>
"""


def render_line_chart(slide_id, chart_config):
    if not chart_config:
        return "", ""
    canvas_id = f"chart-{slide_id}"
    labels = chart_config.get("x_axis") or []
    series = chart_config.get("series") or []
    datasets = []
    for idx, s in enumerate(series):
        datasets.append(
            {
                "label": s.get("name", f"Series {idx+1}"),
                "data": s.get("data", []),
                "borderColor": color_for_index(idx),
                "backgroundColor": color_for_index(idx),
                "fill": False,
                "tension": 0.25,
                "pointRadius": 3,
            }
        )
    chart_html = f"""
<div class=\"card\" style=\"height:100%;\">
  <div style=\"font-size:16px;font-weight:700;color:{KPMG['text']};margin-bottom:8px;\">{esc(chart_config.get('title') or '趋势图')}</div>
  <div style=\"height:420px;\"><canvas id=\"{canvas_id}\"></canvas></div>
  <div class=\"small\" style=\"margin-top:8px;\">{esc(chart_config.get('source') or '')}</div>
</div>
"""
    js = f"""
<script>
new Chart(document.getElementById('{canvas_id}'), {{
  type: 'line',
  data: {{ labels: {json.dumps(labels, ensure_ascii=False)}, datasets: {json.dumps(datasets, ensure_ascii=False)} }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{ legend: {{ position: 'bottom' }} }},
    scales: {{ y: {{ title: {{ display: true, text: {json.dumps(chart_config.get('y_axis_title') or 'Index')} }} }} }}
  }}
}});
</script>
"""
    return chart_html, js


def render_generic_slide(slide, page_no, total, deck_title):
    components = slide.get("components") or {}
    visual = slide.get("visual") or {}

    left_blocks = []
    right_blocks = []
    scripts = []

    if components.get("kpis"):
        left_blocks.append(render_kpis(components.get("kpis")))
    if components.get("decisions"):
        left_blocks.append(render_decisions(components.get("decisions")))
    if components.get("comparison_items"):
        left_blocks.append(render_comparison(components.get("comparison_items")))
    if components.get("timeline_items"):
        left_blocks.append(render_timeline(components.get("timeline_items")))
    if components.get("risks"):
        left_blocks.append(render_risks(components.get("risks")))
    if components.get("action_items"):
        left_blocks.append(render_actions(components.get("action_items")))

    vt = visual.get("type", "none")
    placeholder = visual.get("placeholder_data") or {}
    if vt == "line_chart":
        cfg = placeholder.get("chart_config")
        if isinstance(cfg, list):
            cfg = cfg[0] if cfg else None
        chart_html, chart_js = render_line_chart(slide.get("slide_id"), cfg)
        if chart_html:
            left_blocks.append(chart_html)
        if chart_js:
            scripts.append(chart_js)
    elif vt == "flow_diagram":
        left_blocks.append(render_flow_diagram(placeholder.get("flow_data") or {}))
    elif vt in {"architecture_diagram", "decision_tree"}:
        left_blocks.append(render_mermaid(placeholder.get("mermaid_code") or ""))

    if components.get("callouts"):
        right_blocks.append(render_callouts(components.get("callouts")))

    if slide.get("assertion"):
        right_blocks.insert(
            0,
            f"<div class=\"card\" style=\"border-left:4px solid {KPMG['blue']};\"><div class=\"label\" style=\"font-weight:700;\">核心结论</div><div style=\"font-size:16px;color:{KPMG['text']};font-weight:700;margin-top:6px;line-height:1.45;\">{esc(slide.get('assertion'))}</div></div>",
        )
    if slide.get("insight"):
        right_blocks.append(
            f"<div class=\"card\" style=\"border-left:4px solid {KPMG['light_blue']};\"><div class=\"label\" style=\"font-weight:700;\">行动建议</div><div style=\"font-size:15px;color:{KPMG['text']};margin-top:6px;line-height:1.5;\">{esc(slide.get('insight'))}</div></div>"
        )

    if not left_blocks and slide.get("content"):
        left_blocks.append(
            "<div class='card'>"
            + "".join(f"<div style='font-size:16px;color:{KPMG['text']};line-height:1.65;margin-bottom:8px;'>• {esc(c)}</div>" for c in slide.get("content"))
            + "</div>"
        )

    left_html = "".join(left_blocks) if left_blocks else "<div class='card'>暂无内容</div>"
    right_html = "".join(right_blocks)
    if not right_html:
        right_html = "<div class='card'><div class='small'>本页无侧栏补充。</div></div>"

    return f"""
<body>
  <div class=\"slide\">
    <div class=\"topbar\"></div>
    <div class=\"header\">
      <div>
        <div class=\"subtitle\">{esc(slide.get('slide_type') or 'slide')}</div>
        <div class=\"title\">{esc(slide.get('title') or deck_title)}</div>
      </div>
      <div class=\"small\">{page_no} / {total}</div>
    </div>
    <div class=\"body\">
      <div style=\"flex:7;display:flex;flex-direction:column;gap:12px;min-width:0;\">{left_html}</div>
      <div style=\"flex:4;display:flex;flex-direction:column;gap:12px;min-width:0;\">{right_html}</div>
    </div>
    <div class=\"footer\"><div>{esc(deck_title)}</div><div>Slide {page_no}</div></div>
  </div>
  {''.join(scripts)}
</body>
</html>
"""


def render_slide(slide, page_no, total, deck_title):
    slide_type = slide.get("slide_type")
    head = base_head(slide.get("title") or deck_title)
    if slide_type == "title":
        return head + render_cover(slide, deck_title)
    if slide_type == "section_divider":
        return head + render_section_divider(slide, page_no, total)
    return head + render_generic_slide(slide, page_no, total, deck_title)


def write_presentation_index(output_dir, deck_title, total):
    links = []
    for i in range(1, total + 1):
        links.append(
            f"<a class='item' href='slide-{i}.html' target='viewer'>第{i}页</a>"
        )
    index_html = f"""<!DOCTYPE html>
<html lang=\"zh-CN\">
<head>
  <meta charset=\"UTF-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
  <title>{esc(deck_title)} - HTML 预览</title>
  <style>
    body {{ margin:0; font-family:Arial, sans-serif; background:#f3f4f6; }}
    .wrap {{ display:flex; height:100vh; }}
    .nav {{ width:220px; background:#00338D; color:#fff; padding:16px; overflow:auto; }}
    .title {{ font-weight:700; margin-bottom:14px; line-height:1.4; }}
    .item {{ display:block; color:#dbeafe; text-decoration:none; padding:8px 10px; border-radius:8px; margin-bottom:6px; }}
    .item:hover {{ background:#1d4ed8; color:#fff; }}
    iframe {{ border:0; width:100%; height:100%; background:#d1d5db; }}
  </style>
</head>
<body>
  <div class=\"wrap\">
    <aside class=\"nav\">
      <div class=\"title\">{esc(deck_title)}<br/>HTML 演示 ({total}页)</div>
      {''.join(links)}
    </aside>
    <main style=\"flex:1;\"><iframe name=\"viewer\" src=\"slide-1.html\"></iframe></main>
  </div>
</body>
</html>
"""
    (output_dir / "presentation.html").write_text(index_html, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Generate HTML slides from semantic JSON")
    parser.add_argument("--semantic", required=True, help="Path to slides_semantic_v2.json")
    parser.add_argument("--output-dir", required=True, help="Output directory for slide HTML files")
    args = parser.parse_args()

    semantic_path = Path(args.semantic)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    doc = json.loads(semantic_path.read_text(encoding="utf-8"))
    slides = sorted(doc.get("slides", []), key=lambda s: s.get("slide_id", s.get("id", 0)))
    deck_title = doc.get("deck_title", "Presentation")
    total = len(slides)

    for index, slide in enumerate(slides, start=1):
        html_text = render_slide(slide, index, total, deck_title)
        (output_dir / f"slide-{index}.html").write_text(html_text, encoding="utf-8")

    write_presentation_index(output_dir, deck_title, total)
    print(f"Generated {total} slides in {output_dir}")


if __name__ == "__main__":
    main()
