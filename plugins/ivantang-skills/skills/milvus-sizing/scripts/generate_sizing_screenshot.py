#!/usr/bin/env python3
"""
Generate an HTML report matching the Milvus sizing tool layout and screenshot it.

Usage:
  python generate_sizing_screenshot.py --vectors 1000000 --dim 1536
  python generate_sizing_screenshot.py --vectors 1e9 --dim 768 --hnsw-m 16
  python generate_sizing_screenshot.py --vectors 1000000 --dim 1536 --out /tmp/report.png

Requires: playwright CLI at /opt/node22/bin/playwright  (Chromium)
"""

import argparse
import math
import os
import subprocess
import sys
import tempfile

# ---------------------------------------------------------------------------
# Reuse sizing logic from milvus_sizing.py (same directory)
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import milvus_sizing as ms


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Milvus Sizing Tool — Report</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #f5f7fa;
    color: #1a1a2e;
    padding: 32px;
  }}
  .card {{
    background: #fff;
    border-radius: 12px;
    box-shadow: 0 2px 12px rgba(0,0,0,.08);
    max-width: 960px;
    margin: 0 auto;
    overflow: hidden;
  }}
  .header {{
    background: linear-gradient(135deg, #06b6d4 0%, #3b82f6 100%);
    color: #fff;
    padding: 28px 36px 24px;
    display: flex;
    align-items: center;
    gap: 14px;
  }}
  .header svg {{ flex-shrink: 0; }}
  .header h1 {{ font-size: 22px; font-weight: 700; letter-spacing: -.3px; }}
  .header p  {{ font-size: 13px; opacity: .85; margin-top: 4px; }}
  .source-badge {{
    margin-left: auto; background: rgba(255,255,255,.2);
    border-radius: 6px; padding: 4px 10px; font-size: 11px; white-space: nowrap;
  }}

  .body {{ padding: 28px 36px 32px; }}

  /* Input params */
  .params {{
    display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 28px;
  }}
  .param-chip {{
    background: #f0f9ff; border: 1px solid #bae6fd; border-radius: 999px;
    padding: 5px 14px; font-size: 13px; color: #0369a1; font-weight: 500;
  }}

  /* Two-column grid */
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }}
  @media (max-width: 680px) {{ .grid {{ grid-template-columns: 1fr; }} }}

  .section {{
    background: #f8fafc; border-radius: 10px; padding: 18px 20px;
    border: 1px solid #e2e8f0;
  }}
  .section h2 {{
    font-size: 12px; font-weight: 600; text-transform: uppercase;
    letter-spacing: .8px; color: #64748b; margin-bottom: 14px;
  }}

  /* Metric rows */
  .metric {{ display: flex; align-items: baseline; justify-content: space-between; padding: 5px 0; }}
  .metric + .metric {{ border-top: 1px solid #e2e8f0; }}
  .metric-label {{ font-size: 13px; color: #475569; }}
  .metric-value {{ font-size: 14px; font-weight: 600; color: #0f172a; text-align: right; }}
  .metric-sub  {{ font-size: 11px; color: #94a3b8; display: block; }}

  /* Component table */
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th {{ background: #f1f5f9; font-weight: 600; font-size: 11px; text-transform: uppercase;
       letter-spacing: .6px; color: #64748b; padding: 8px 10px; text-align: left; }}
  td {{ padding: 9px 10px; border-top: 1px solid #e2e8f0; vertical-align: middle; }}
  tr:hover td {{ background: #f8fafc; }}
  .count-badge {{
    display: inline-block; background: #dbeafe; color: #1d4ed8;
    border-radius: 999px; padding: 2px 9px; font-size: 11px; font-weight: 600;
  }}

  /* Dependency section */
  .dep-row {{ display: flex; gap: 16px; flex-wrap: wrap; }}
  .dep-card {{
    flex: 1; min-width: 160px; background: #fff; border: 1px solid #e2e8f0;
    border-radius: 8px; padding: 12px 14px;
  }}
  .dep-card h3 {{ font-size: 11px; font-weight: 600; color: #64748b;
                  text-transform: uppercase; letter-spacing: .6px; margin-bottom: 8px; }}
  .dep-kv {{ display: flex; justify-content: space-between; font-size: 12px; padding: 3px 0; }}
  .dep-kv span:last-child {{ font-weight: 600; color: #0f172a; }}

  /* Summary bar */
  .summary {{
    background: linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 100%);
    border: 1px solid #86efac; border-radius: 10px;
    padding: 16px 20px; display: flex; gap: 32px; flex-wrap: wrap;
    margin-top: 20px;
  }}
  .sum-item {{ text-align: center; }}
  .sum-num {{ font-size: 24px; font-weight: 700; color: #15803d; }}
  .sum-label {{ font-size: 11px; color: #4ade80; font-weight: 500; margin-top: 2px; }}

  /* Formula note */
  .formula-note {{
    margin-top: 20px; background: #fefce8; border: 1px solid #fde68a;
    border-radius: 8px; padding: 12px 16px; font-size: 12px; color: #713f12;
  }}
  .formula-note strong {{ display: block; margin-bottom: 4px; font-size: 11px;
                           text-transform: uppercase; letter-spacing: .5px; color: #92400e; }}
  code {{ background: #fef3c7; border-radius: 3px; padding: 1px 5px; font-family: monospace; }}

  .footer {{
    text-align: center; font-size: 11px; color: #94a3b8;
    margin-top: 24px; padding-top: 16px; border-top: 1px solid #e2e8f0;
  }}
  .footer a {{ color: #3b82f6; text-decoration: none; }}
</style>
</head>
<body>
<div class="card">
  <div class="header">
    <svg width="36" height="36" viewBox="0 0 36 36" fill="none">
      <rect width="36" height="36" rx="8" fill="rgba(255,255,255,.2)"/>
      <path d="M18 8 L28 13.5 V22.5 L18 28 L8 22.5 V13.5 Z" stroke="white" stroke-width="1.8" fill="none"/>
      <circle cx="18" cy="18" r="4" fill="white"/>
    </svg>
    <div>
      <h1>Milvus Sizing Tool</h1>
      <p>Resource estimate for your vector collection — Distributed Mode</p>
    </div>
    <div class="source-badge">Formulas: milvus-io/milvus.io sizingTool.ts</div>
  </div>

  <div class="body">
    <div class="params">
      {param_chips}
    </div>

    <div class="grid">
      <div class="section">
        <h2>Data Size</h2>
        {data_size_metrics}
      </div>
      <div class="section">
        <h2>Index Memory</h2>
        {index_metrics}
      </div>
    </div>

    <div class="section" style="margin-bottom:20px">
      <h2>Distributed Components</h2>
      <table>
        <thead>
          <tr><th>Component</th><th>Count</th><th>vCPU</th><th>RAM (GiB)</th><th>Role</th></tr>
        </thead>
        <tbody>
          {component_rows}
        </tbody>
      </table>
    </div>

    <div class="section">
      <h2>Dependencies</h2>
      <div class="dep-row">
        {dep_cards}
      </div>
    </div>

    <div class="summary">
      {summary_items}
    </div>

    <div class="formula-note">
      <strong>Formula used ({index_label})</strong>
      {formula_html}
      Loading memory = <code>(index_memory + 2 × segment_size) × 1.15</code><br/>
      Node tiers from <code>clusterNodesConfigCalculator()</code> · MinIO = <code>max(⌈(raw + loading) GiB⌉, 30)</code>
    </div>

    <div class="footer">
      Cross-check at <a href="https://milvus.io/tools/sizing">milvus.io/tools/sizing</a> ·
      Formulas sourced from
      <a href="https://github.com/milvus-io/milvus.io/blob/master/src/utils/sizingTool.ts">
        github.com/milvus-io/milvus.io
      </a>
    </div>
  </div>
</div>
</body>
</html>
"""


def chip(label, value):
    return f'<span class="param-chip"><b>{label}</b>: {value}</span>'


def metric(label, value, sub=None):
    sub_html = f'<span class="metric-sub">{sub}</span>' if sub else ""
    return f"""
    <div class="metric">
      <span class="metric-label">{label}</span>
      <span class="metric-value">{value}{sub_html}</span>
    </div>"""


def comp_row(name, count, cpu, mem, role):
    return f"""
    <tr>
      <td>{name}</td>
      <td><span class="count-badge">×{count}</span></td>
      <td>{cpu}</td>
      <td>{mem}</td>
      <td style="color:#64748b;font-size:12px">{role}</td>
    </tr>"""


def dep_card(title, kvs):
    rows = "".join(f'<div class="dep-kv"><span>{k}</span><span>{v}</span></div>' for k, v in kvs)
    return f'<div class="dep-card"><h3>{title}</h3>{rows}</div>'


def sum_item(num, label):
    return f'<div class="sum-item"><div class="sum-num">{num}</div><div class="sum-label">{label}</div></div>'


def generate_html(r):
    n = r["nodes"]
    d = r["deps"]
    idx = r["idx"].upper()
    num_v = r["num_vectors"]
    dim = r["dim"]

    # --- param chips ---
    chips = [
        chip("Vectors", f"{num_v:,}"),
        chip("Dimensions", str(dim)),
        chip("Index", idx + (f" ({r['params_str']})" if r["params_str"] else "")),
        chip("Mode", "Distributed"),
        chip("Segment size", f"{r['segment_gib']*1024:.0f} MiB"),
    ]

    # --- data size metrics ---
    row_size = dim * ms.BYTES_PER_FLOAT32
    data_metrics = [
        metric("Raw data size", ms.fmt(r["raw"]),
               f"{num_v:,} × {dim} × 4 bytes"),
        metric("Row size (1 vector)", ms.fmt(row_size)),
    ]

    # --- index metrics ---
    idx_metrics = [
        metric("Formula", f"<code style='font-size:12px'>{r['formula_str']}</code>"),
        metric("Index memory", ms.fmt(r["index_mem"])),
        metric("Loading memory", ms.fmt(r["load_mem"]),
               "incl. 2×segment buffer ×1.15"),
    ]
    if r["index_disk"]:
        idx_metrics.append(metric("Index disk (DiskANN)", ms.fmt(r["index_disk"])))

    # --- component rows ---
    rows = [
        comp_row("Query Node", n["qn"], n["qn_cpu"], n["qn_mem"],
                 "Serves search queries, holds index in memory"),
        comp_row("Data Node", n["dn"], n["dn_cpu"], n["dn_mem"],
                 "Ingests data, manages growing segments"),
        comp_row("mixCoord", 1, n["coord_cpu"], n["coord_mem"],
                 "Coordination (root/data/query/index coord)"),
        comp_row("Proxy", 1, n["proxy_cpu"], n["proxy_mem"],
                 "Client gateway, load balancing"),
    ]

    # --- dep cards ---
    cards = [
        dep_card("MinIO (Object Storage)", [
            ("PVC", f"{d['minio_gib']} GiB"),
            ("Formula", "max(⌈(raw+loading) GiB⌉, 30)"),
        ]),
        dep_card("Pulsar (Message Broker)", [
            ("Ledgers PVC", f"{d['pulsar_ledgers_gib']} GiB"),
            ("Journal PVC", f"{d['pulsar_journal_gib']} GiB"),
        ]),
        dep_card("etcd (Metadata, HA×3)", [
            ("Disk / node", f"{d['etcd_gib']} GiB SSD"),
            ("Total", f"{d['etcd_gib'] * 3} GiB"),
        ]),
    ]

    # --- summary ---
    total_nodes = n["qn"] + n["dn"] + 2
    total_ram = n["qn"] * n["qn_mem"] + n["dn"] * n["dn_mem"] + n["coord_mem"] + n["proxy_mem"]
    items = [
        sum_item(total_nodes, "Milvus Nodes"),
        sum_item(f"{total_ram} GiB", "Total RAM"),
        sum_item(f"{d['minio_gib']} GiB", "Object Storage"),
        sum_item(f"{d['pulsar_ledgers_gib']+d['pulsar_journal_gib']} GiB", "Pulsar PVC"),
    ]

    # --- formula ---
    formula_html = f"<code>{r['formula_str']}</code><br/>"

    return HTML.format(
        param_chips="".join(chips),
        data_size_metrics="".join(data_metrics),
        index_metrics="".join(idx_metrics),
        component_rows="".join(rows),
        dep_cards="".join(cards),
        summary_items="".join(items),
        index_label=idx,
        formula_html=formula_html,
    )


# ---------------------------------------------------------------------------
# Playwright screenshot (via CLI)
# ---------------------------------------------------------------------------

def screenshot(html_path, out_path, width=1020):
    playwright = os.environ.get("PLAYWRIGHT_BIN", "/opt/node22/bin/playwright")
    chromium = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"

    # playwright screenshot CLI: playwright screenshot [options] <url> <file>
    cmd = [
        playwright, "screenshot",
        "--browser", "chromium",
        "--viewport-size", f"{width},900",
        "--full-page",
        f"file://{html_path}",
        out_path,
    ]
    env = os.environ.copy()
    env["PLAYWRIGHT_BROWSERS_PATH"] = "/opt/pw-browsers"

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        # Fallback: use Node inline script
        node_script = f"""
const {{ chromium }} = require('playwright');
(async () => {{
  const b = await chromium.launch({{
    executablePath: '{chromium}',
    headless: true,
    args: ['--no-sandbox']
  }});
  const p = await b.newPage();
  await p.setViewportSize({{ width: {width}, height: 900 }});
  await p.goto('file://{html_path}', {{ waitUntil: 'networkidle' }});
  await p.screenshot({{ path: '{out_path}', fullPage: true }});
  await b.close();
}})().catch(e => {{ console.error(e.message); process.exit(1); }});
"""
        node = "/opt/node22/bin/node"
        node_env = env.copy()
        node_env["NODE_PATH"] = "/opt/node22/lib/node_modules"
        r2 = subprocess.run([node, "-e", node_script],
                            capture_output=True, text=True, env=node_env)
        if r2.returncode != 0:
            print(f"Screenshot error:\n{r2.stderr}", file=sys.stderr)
            sys.exit(1)

    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vectors", type=float, required=True)
    parser.add_argument("--dim", type=int, required=True)
    parser.add_argument("--index", type=str, default="hnsw",
                        choices=["hnsw", "flat", "ivf_flat", "ivf_sq8", "ivf_pq",
                                 "scann", "diskann", "ivf_rabitq"])
    parser.add_argument("--hnsw-m", type=int, default=30)
    parser.add_argument("--nlist", type=int, default=128)
    parser.add_argument("--ivfpq-m", type=int, default=None)
    parser.add_argument("--ivfpq-nbits", type=int, default=8)
    parser.add_argument("--diskann-max-degree", type=int, default=56)
    parser.add_argument("--scann-with-raw-data", action="store_true")
    parser.add_argument("--rabitq-refine-type", type=str, default="SQ8")
    parser.add_argument("--segment-size-mb", type=int, default=512)
    parser.add_argument("--out", type=str, default="/tmp/milvus_sizing.png")
    args = parser.parse_args()

    result = ms.compute(args)
    ms.print_report(result)

    html_content = generate_html(result)

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
        f.write(html_content)
        html_path = f.name

    print(f"Generating screenshot → {args.out}")
    screenshot(html_path, args.out)
    os.unlink(html_path)
    print(f"Screenshot saved: {args.out}")


if __name__ == "__main__":
    main()
