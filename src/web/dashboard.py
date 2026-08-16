"""Build a static, locally-hostable dashboard for the EV charger results.

Produces a single self-contained `index.html` that embeds:
  - the folium interactive map (from the saved HTML file, in an iframe)
  - the optimization vs. greedy before/after chart
  - the gamma-sensitivity chart
  - the equity income-profile chart
  - a table of recommended sites
  - a summary stat panel

Host it locally with the included stdlib server:
    python -m src.web.serve 8000
"""
from __future__ import annotations

import base64
import os
from pathlib import Path

import pandas as pd

# Summary metric cards shown at the top of the dashboard.
# Budget model (primary): demand units served, chargers built, sites used, spend.
_BUDGET_METRIC_LABELS = [
    ("demand_units_served", "Demand served (units)"),
    ("n_chargers_total", "Chargers built"),
    ("n_sites_used", "Sites used"),
    ("budget_spent", "Budget spent"),
]
_BUDGET_CARDS = {
    "demand_units_served": lambda v: f"{v:,.0f}",
    "n_chargers_total": lambda v: f"{int(v)}",
    "n_sites_used": lambda v: f"{int(v)}",
    "budget_spent": lambda v: f"${v:,.0f}",
}

# Legacy MCLP model.
_METRIC_LABELS = [
    ("k_sites", "Chargers sited"),
    ("frac_weight_covered", "Demand covered"),
    ("duplicated_coverage", "Redundant overlaps"),
    ("mean_coverage_per_covered", "Avg coverage / cell"),
]

_CARDS = {
    "frac_weight_covered": lambda v: f"{v*100:.1f}%",
    "duplicated_coverage": lambda v: f"{int(v)}",
    "mean_coverage_per_covered": lambda v: f"{v:.2f}",
    "k_sites": lambda v: f"{int(v)}",
}


def _b64(path: str) -> str:
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def _results_html(csv_path: str) -> str:
    if not os.path.exists(csv_path):
        return "<p>No results.</p>"
    df = pd.read_csv(csv_path)
    rows = ""
    for _, r in df.iterrows():
        cols = "".join(f"<td>{r[c]}</td>" for c in df.columns)
        rows += f"<tr>{cols}</tr>"
    head = "".join(f"<th>{c}</th>" for c in df.columns)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>"


def _cards_html(metrics_opt: dict, metrics_greedy: dict) -> str:
    budget_style = "demand_units_served" in metrics_opt
    if budget_style:
        labels, cards = _BUDGET_METRIC_LABELS, _BUDGET_CARDS
    else:
        labels, cards = _METRIC_LABELS, _CARDS
    out = ""
    for key, label in labels:
        v_opt = metrics_opt.get(key, 0)
        v_gr = metrics_greedy.get(key, 0)
        fmt = cards.get(key, str)
        try:
            opt_s = fmt(v_opt)
        except Exception:
            opt_s = str(v_opt)
        try:
            gr_s = fmt(v_gr)
        except Exception:
            gr_s = str(v_gr)
        if key == "budget_spent":
            cls = ""
        else:
            try:
                better = (v_opt >= v_gr) if key in {
                    "demand_units_served", "n_chargers_total", "n_sites_used",
                    "frac_weight_covered"} else (v_opt <= v_gr)
            except Exception:
                better = True
            cls = "good" if better else "bad"
        out += f"""
        <div class="card {cls}">
          <div class="label">{label}</div>
          <div class="value">{opt_s}</div>
          <div class="sub">greedy: {gr_s}</div>
        </div>"""
    return out


_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EV Charger Placement · {region}</title>
<style>
  :root {{
    --bg:#0f1720; --card:#17222e; --line:#2b3a4b;
    --good:#2ecc71; --bad:#e74c3c; --txt:#cbd5e1; --hi:#f1f5f9;
  }}
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ background:var(--bg); color:var(--txt); font-family:-apple-system,
         "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }}
  header {{ padding:24px 32px; border-bottom:1px solid var(--line);
           background:linear-gradient(180deg,#14202c,#0f1720); }}
  header h1 {{ color:var(--hi); font-size:22px; }}
  header p {{ margin-top:4px; font-size:13px; opacity:.75; }}
  main {{ max-width:1200px; margin:0 auto; padding:24px 32px; }}
  .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr));
           gap:14px; margin-bottom:28px; }}
  .card {{ background:var(--card); border:1px solid var(--line); border-radius:10px;
          padding:16px; }}
  .card .label {{ font-size:12px; text-transform:uppercase; letter-spacing:.5px;
                 opacity:.7; }}
  .card .value {{ font-size:28px; font-weight:700; color:var(--hi); }}
  .card.good .value {{ color:var(--good); }}
  .card.bad .value {{ color:var(--bad); }}
  .card .sub {{ font-size:12px; margin-top:6px; opacity:.6; }}
  section {{ margin-bottom:28px; }}
  h2 {{ color:var(--hi); font-size:16px; margin-bottom:10px;
       padding-bottom:6px; border-bottom:1px solid var(--line); }}
  .map-frame {{ width:100%; height:560px; border:1px solid var(--line);
                border-radius:10px; overflow:hidden; }}
  .map-frame iframe {{ width:100%; height:100%; border:0; }}
  .chart-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(340px,1fr));
                gap:14px; }}
  .chart-grid img {{ width:100%; background:var(--card); border:1px solid var(--line);
                     border-radius:10px; padding:8px; }}
  table {{ width:100%; border-collapse:collapse; background:var(--card);
          border:1px solid var(--line); border-radius:10px; overflow:hidden; }}
  th,td {{ text-align:left; padding:10px 12px; font-size:13px; }}
  th {{ background:#1d2b3a; color:var(--hi); text-transform:uppercase;
       font-size:11px; letter-spacing:.5px; }}
  td {{ border-top:1px solid var(--line); }}
  .foot {{ color:#7f8fa6; font-size:12px; padding:20px 32px; text-align:center; }}
</style>
</head>
<body>
<header>
  <h1>⚡ EV Charger Placement · {region}</h1>
  <p>Equity-aware capacitated charger placement · budget ${budget:,.0f} ·
     L2 walk {l2:.0f} min · DCFC drive {dcfc:.0f} min · {solver}</p>
</header>
<main>
  <div class="cards">{cards}</div>

  <section>
    <h2>Interactive map</h2>
    <div class="map-frame">{map_iframe}</div>
  </section>

  <section>
    <h2>Charts</h2>
    <div class="chart-grid">
      <figure><img src="{b64_ba}" alt="before/after"></figure>
      <figure><img src="{b64_mix}" alt="charger mix"></figure>
      <figure><img src="{b64_equity}" alt="equity profile"></figure>
    </div>
  </section>

  <section>
    <h2>Recommended sites</h2>
    {results_table}
  </section>
</main>
<div class="foot">Generated by the EV Charger Placement pipeline ·
   OpenStreetMap © contributors · NCDOT · US Census ACS · NREL/RPL</div>
</body>
</html>
"""


def build_dashboard(
    out_html: str,
    map_html: str,
    charts_dir: str,
    results_csv: str,
    region: str = "Durham-Orange NC",
    metrics_opt: dict = None,
    metrics_greedy: dict = None,
    k_sites: int = 5,
    l2_walk_time_min: float = 20.0,
    dcfc_drive_time_min: float = 20.0,
    gamma: float = 0.5,
    solver: str = "auto",
    budget: float = 1_500_000.0,
) -> str:
    """Assemble and write the static dashboard to `out_html`."""
    metrics_opt = metrics_opt or {}
    metrics_greedy = metrics_greedy or {}

    # Copy the folium map HTML as a sibling file so an iframe can load it via
    # a relative src (folium HTML contains quotes/script that break inline srcdoc).
    with open(map_html) as f:
        map_src = f.read()
    os.makedirs(os.path.dirname(out_html) or ".", exist_ok=True)
    map_path_out = os.path.join(os.path.dirname(out_html) or ".", "map.html")
    with open(map_path_out, "w") as f:
        f.write(map_src)

    html = _TEMPLATE.format(
        region=region,
        k=k_sites,
        budget=budget,
        l2=l2_walk_time_min,
        dcfc=dcfc_drive_time_min,
        gamma=gamma,
        solver=solver,
        cards=_cards_html(metrics_opt, metrics_greedy),
        map_iframe=(f'<iframe src="map.html" id="map"></iframe>'),
        b64_ba=_b64(os.path.join(charts_dir, "before_after_durham.png")),
        b64_mix=_b64(os.path.join(charts_dir, "budget_mix_durham.png")),
        b64_equity=_b64(os.path.join(charts_dir, "equity_durham.png")),
        results_table=_results_html(results_csv),
    )
    with open(out_html, "w") as f:
        f.write(html)
    return out_html