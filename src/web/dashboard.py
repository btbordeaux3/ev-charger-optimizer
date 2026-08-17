"""Build the self-contained local HTML dashboard."""
from __future__ import annotations

import base64
import html
import os
from pathlib import Path

import pandas as pd


def _b64_or_none(path: str) -> str | None:
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def _table_html(csv_path: str, *, percent_columns: set[str] | None = None, max_rows: int = 0) -> str:
    if not os.path.exists(csv_path):
        return "<p class='muted'>No results generated.</p>"
    df = pd.read_csv(csv_path)
    if df.empty:
        return "<p class='muted'>No rows generated.</p>"
    truncated = False
    total_rows = len(df)
    if max_rows and total_rows > max_rows:
        df = df.head(max_rows).copy()
        truncated = True
    percent_columns = percent_columns or set()
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    rows = []
    for _, r in df.iterrows():
        vals = []
        for c in df.columns:
            value = r[c]
            if c in percent_columns and pd.notna(value):
                text = f"{float(value) * 100:.1f}%"
            elif isinstance(value, float) and pd.notna(value):
                text = f"{value:,.3f}".rstrip("0").rstrip(".")
            else:
                text = str(value)
            vals.append(f"<td>{html.escape(text)}</td>")
        rows.append("<tr>" + "".join(vals) + "</tr>")
    table = f"<div class='table-wrap'><table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    if truncated:
        table += f"<p class='muted'>Showing first {len(df):,} of {total_rows:,} rows. Full audit is in the CSV.</p>"
    return table


def _cards_html(metrics_opt: dict, metrics_greedy: dict, coverage_csv: str, regulatory_csv: str = "") -> str:
    cards = []

    def card(label, value, sub=""):
        cards.append(
            f"<div class='card'><div class='label'>{html.escape(label)}</div>"
            f"<div class='value'>{html.escape(str(value))}</div>"
            f"<div class='sub'>{html.escape(str(sub))}</div></div>"
        )

    card(
        "Demand units served",
        f"{metrics_opt.get('demand_units_served', 0):,.0f}",
        f"greedy {metrics_greedy.get('demand_units_served', 0):,.0f}",
    )
    card(
        "Distinct sites",
        int(metrics_opt.get("n_sites_used", 0)),
        f"{int(metrics_opt.get('n_chargers_total', 0))} chargers total",
    )
    card(
        "Budget spent",
        f"${metrics_opt.get('budget_spent', 0):,.0f}",
        f"solver {metrics_opt.get('solver_used', 'n/a')}",
    )

    if os.path.exists(coverage_csv):
        cov = pd.read_csv(coverage_csv)
        row = cov[(cov["mode"] == "either") & (cov["threshold_min"].astype(float) == 10.0)]
        if not row.empty:
            r = row.iloc[0]
            card(
                "10-min combined access",
                f"{float(r['weighted_demand_fraction']) * 100:.1f}%",
                f"population {float(r['population_fraction']) * 100:.1f}% · land {float(r['land_area_fraction']) * 100:.1f}%",
            )
    if regulatory_csv and os.path.exists(regulatory_csv):
        reg = pd.read_csv(regulatory_csv)
        if not reg.empty:
            matched = reg.get("parcel_id", pd.Series([""] * len(reg))).fillna("").astype(str).str.len().gt(0).sum()
            review = reg.get("reg_manual_review", pd.Series([False] * len(reg))).astype(str).str.lower().isin(["true", "1"]).sum()
            card(
                "Regulatory context",
                f"{matched}/{len(reg)} matched",
                f"{review} candidate(s) flagged for manual review",
            )
    return "".join(cards)


def _chart_html(path: str, alt: str) -> str:
    src = _b64_or_none(path)
    if src is None:
        return f"<div class='chart-missing'>No {html.escape(alt)} chart available for this run.</div>"
    return f'<figure><img src="{src}" alt="{html.escape(alt)}"></figure>'


_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EV Charger Placement · {region}</title>
<style>
:root{{--bg:#0b1220;--panel:#111b2e;--line:#273449;--txt:#cbd5e1;--hi:#f8fafc;--accent:#facc15;}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--txt);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
header{{padding:24px 32px;border-bottom:1px solid var(--line);background:#0f192a}} h1{{margin:0;color:var(--hi);font-size:23px}} header p{{margin:6px 0 0;opacity:.72;font-size:13px}}
main{{max-width:1280px;margin:auto;padding:24px 30px}} .cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(205px,1fr));gap:12px;margin-bottom:26px}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px}} .label{{font-size:11px;letter-spacing:.07em;text-transform:uppercase;opacity:.7}} .value{{font-size:28px;font-weight:750;color:var(--hi);margin-top:3px}} .sub{{font-size:12px;margin-top:5px;opacity:.65}}
section{{margin:0 0 28px}} h2{{font-size:16px;color:var(--hi);padding-bottom:7px;border-bottom:1px solid var(--line)}} .map{{height:600px;border:1px solid var(--line);border-radius:12px;overflow:hidden}} iframe{{width:100%;height:100%;border:0}}
.notice{{background:#172033;border:1px solid #3b4a63;border-radius:10px;padding:12px 14px;font-size:12px;line-height:1.5;margin:0 0 12px}}
.chart-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:14px}} figure{{margin:0}} figure img,.chart-missing{{width:100%;background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:8px}} .chart-missing{{min-height:130px;display:grid;place-items:center;opacity:.65}}
.table-wrap{{overflow:auto;border:1px solid var(--line);border-radius:12px}} table{{width:100%;border-collapse:collapse;background:var(--panel)}} th,td{{padding:9px 11px;text-align:left;white-space:nowrap;font-size:12px}} th{{background:#18253a;color:var(--hi);font-size:10px;text-transform:uppercase;letter-spacing:.06em}} td{{border-top:1px solid var(--line)}} .muted{{opacity:.65}}
footer{{padding:22px;text-align:center;font-size:11px;opacity:.55}}
</style>
</head>
<body>
<header><h1>⚡ EV Charger Placement · {region}</h1><p>Network-time accessibility · L2 walk {l2:g} min · DCFC drive {dcfc:g} min · routing {routing_backend} · optimization {solver}</p></header>
<main>
<div class="cards">{cards}</div>
<section><h2>Interactive accessibility map</h2><div class="map"><iframe src="map.html"></iframe></div></section>
<section><h2>Plan comparison</h2><div class="chart-grid">{chart_ba}{chart_mix}{chart_equity}</div></section>
<section><h2>Accessibility coverage</h2>{coverage_table}</section>
{regulatory_section}
<section><h2>Recommended sites</h2>{results_table}</section>
</main>
<footer>Generated locally by EV Charger Optimizer V3 · OpenStreetMap contributors · public transportation/energy/regulatory datasets</footer>
</body></html>"""


def build_dashboard(
    out_html: str,
    map_html: str,
    charts_dir: str,
    results_csv: str,
    coverage_csv: str,
    regulatory_csv: str = "",
    *,
    region: str,
    metrics_opt: dict,
    metrics_greedy: dict,
    l2_walk_time_min: float,
    dcfc_drive_time_min: float,
    solver: str,
    routing_backend: str,
    budget: float,
    regulations_enabled: bool = False,
    regulations_jurisdiction: str = "",
    regulations_as_of: str = "",
) -> str:
    """Assemble and write the local static dashboard."""
    out = Path(out_html)
    out.parent.mkdir(parents=True, exist_ok=True)
    map_target = out.parent / "map.html"
    map_target.write_text(Path(map_html).read_text(encoding="utf-8"), encoding="utf-8")

    percent_cols = {
        "weighted_demand_fraction", "optimization_demand_fraction",
        "population_fraction", "land_area_fraction",
    }
    if regulations_enabled and regulatory_csv and os.path.exists(regulatory_csv):
        jurisdiction = html.escape(regulations_jurisdiction or "configured jurisdiction")
        as_of = html.escape(regulations_as_of or "configured source date")
        regulatory_section = (
            "<section><h2>Regulatory screening</h2>"
            f"<div class='notice'><b>{jurisdiction}</b> · {as_of}<br>"
            "Planning-screening only: parcel/zoning matches and configured ordinance rules can flag or cap candidates, "
            "but this dashboard is not a permit or legal determination. Conditional provisions remain manual-review flags.</div>"
            + _table_html(regulatory_csv, max_rows=250) + "</section>"
        )
    else:
        regulatory_section = ""

    page = _TEMPLATE.format(
        region=html.escape(region),
        budget=budget,
        l2=l2_walk_time_min,
        dcfc=dcfc_drive_time_min,
        solver=html.escape(str(solver)),
        routing_backend=html.escape(str(routing_backend)),
        cards=_cards_html(metrics_opt, metrics_greedy, coverage_csv, regulatory_csv),
        chart_ba=_chart_html(str(Path(charts_dir) / "before_after.png"), "optimized versus greedy"),
        chart_mix=_chart_html(str(Path(charts_dir) / "budget_mix.png"), "charger mix"),
        chart_equity=_chart_html(str(Path(charts_dir) / "equity.png"), "equity profile"),
        coverage_table=_table_html(coverage_csv, percent_columns=percent_cols),
        regulatory_section=regulatory_section,
        results_table=_table_html(results_csv),
    )
    out.write_text(page, encoding="utf-8")
    return str(out)
