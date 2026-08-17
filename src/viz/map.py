"""Interactive map and comparison charts for V3."""
from __future__ import annotations

import html

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import mapping


def _site_points(
    sites: gpd.GeoDataFrame,
    site_types: list[tuple[int, str]] | list[tuple[int, str, int]],
) -> gpd.GeoDataFrame:
    """Expand selected (site,type,count) tuples while preserving site metadata."""
    if not site_types:
        return gpd.GeoDataFrame(
            {"site_index": [], "charger_type": [], "charger_count": []},
            geometry=gpd.GeoSeries([], crs=sites.crs),
            crs=sites.crs,
        )
    rows = []
    for entry in site_types:
        j, typ = int(entry[0]), str(entry[1])
        count = int(entry[2]) if len(entry) > 2 else 1
        source = sites.iloc[j]
        row = {k: v for k, v in source.items() if k != "geometry"}
        row.update({
            "site_index": j,
            "charger_type": typ,
            "charger_count": count,
            "geometry": source.geometry,
        })
        rows.append(row)
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=sites.crs).reset_index(drop=True)


def _center(region_geom):
    gs = gpd.GeoSeries([region_geom], crs="EPSG:4326").to_crs("EPSG:3857")
    pt = gpd.GeoSeries(gs.centroid, crs="EPSG:3857").to_crs("EPSG:4326").iloc[0]
    return [float(pt.y), float(pt.x)]


def _fmt(v, decimals=0):
    try:
        if pd.isna(v):
            return "n/a"
        return f"{float(v):,.{decimals}f}"
    except Exception:
        return html.escape(str(v))


def folium_map(
    region_name: str,
    demand: gpd.GeoDataFrame,
    sites: gpd.GeoDataFrame,
    recommended_sites: gpd.GeoDataFrame,
    *,
    region_geom,
    existing: gpd.GeoDataFrame | None = None,
    service_layers: dict[str, object] | None = None,
    save_path: str | None = None,
) -> object:
    """Build a Cities-Skylines-like accessibility map.

    The service polygons are unions of demand-grid cells reachable within each
    network-time threshold. They are deliberately labeled as service areas,
    rather than pretending the cell union is a continuous exact isochrone.
    """
    import folium
    from folium.plugins import MarkerCluster

    m = folium.Map(location=_center(region_geom), zoom_start=11, tiles="CartoDB positron")

    folium.GeoJson(
        mapping(region_geom),
        name="Study boundary",
        style_function=lambda _f: {"color": "#111827", "weight": 3, "fillOpacity": 0.0},
        show=True,
    ).add_to(m)

    # Accessibility/service-area layers. Show combined 10-minute coverage by
    # default; all other thresholds remain one click away in LayerControl.
    layer_colors = {"l2_walk": "#2563eb", "dcfc_drive": "#ef4444", "either": "#16a34a"}
    service_layers = service_layers or {}
    for key, geom in sorted(service_layers.items(), key=lambda kv: kv[0]):
        mode, t = key.split("|", 1)
        label = {
            "l2_walk": f"L2 walk service ≤ {t} min",
            "dcfc_drive": f"DCFC drive service ≤ {t} min",
            "either": f"Combined service ≤ {t} min",
        }.get(mode, f"{mode} ≤ {t} min")
        show = mode == "either" and abs(float(t) - 10.0) < 1e-9
        fg = folium.FeatureGroup(name=label, show=show)
        folium.GeoJson(
            mapping(geom),
            style_function=lambda _f, c=layer_colors.get(mode, "#16a34a"): {
                "color": c, "weight": 1, "fillColor": c, "fillOpacity": 0.20
            },
        ).add_to(fg)
        fg.add_to(m)

    # Demand cells are useful diagnostically but visually noisy, so off by default.
    demand_fg = folium.FeatureGroup(name="Weighted demand grid", show=False)
    if not demand.empty and "demand" in demand.columns:
        vals = pd.to_numeric(demand["demand"], errors="coerce").fillna(0).to_numpy(float)
        vmax = max(float(np.nanmax(vals)) if len(vals) else 0.0, 1e-9)
        for _, row in demand.to_crs("EPSG:4326").iterrows():
            frac = max(0.0, min(1.0, float(row.get("demand", 0.0)) / vmax))
            opacity = 0.08 + 0.52 * frac
            popup = (
                f"Demand: {_fmt(row.get('demand', 0), 3)}<br>"
                f"Traffic score: {_fmt(row.get('traffic_norm', 0), 3)}<br>"
                f"Equity score: {_fmt(row.get('equity_norm', 0), 3)}<br>"
                f"Existing-served: {bool(row.get('already_served_existing', False))}"
            )
            folium.GeoJson(
                mapping(row.geometry),
                style_function=lambda _f, op=opacity: {
                    "color": "#7c3aed", "weight": 0.25,
                    "fillColor": "#7c3aed", "fillOpacity": op,
                },
                tooltip=popup,
            ).add_to(demand_fg)
    demand_fg.add_to(m)

    # Candidate sites: visible for debugging the exact-boundary fix.
    cand_fg = folium.FeatureGroup(name="Parking candidates (post-screening)", show=False)
    for idx, r in sites.to_crs("EPSG:4326").iterrows():
        reg_status = str(r.get("regulatory_status", "not configured") or "not configured")
        zoning = str(r.get("zoning_code", "") or "n/a")
        popup = (
            f"Candidate #{idx}<br>{html.escape(str(r.get('name', '') or 'unnamed parking'))}<br>"
            f"Estimated spaces: {_fmt(r.get('estimated_spaces', 0))}<br>"
            f"Physical cap: {_fmt(r.get('site_max_pre_regulation', r.get('site_max', 0)))}<br>"
            f"Final site cap: {_fmt(r.get('site_max', 0))}<br>"
            f"Zoning: {html.escape(zoning)}<br>"
            f"Regulatory status: {html.escape(reg_status)}"
        )
        color = "#9ca3af" if int(r.get("site_max", 0) or 0) > 0 else "#dc2626"
        folium.CircleMarker(
            [r.geometry.y, r.geometry.x], radius=2.5, color=color,
            fill=True, fill_opacity=0.55, weight=1, popup=popup,
        ).add_to(cand_fg)
    cand_fg.add_to(m)

    if "reg_manual_review" in sites.columns:
        review_fg = folium.FeatureGroup(name="Regulatory manual-review candidates", show=False)
        for idx, r in sites.to_crs("EPSG:4326").iterrows():
            if not bool(r.get("reg_manual_review", False)):
                continue
            popup = (
                f"Candidate #{idx}<br>Zoning: {html.escape(str(r.get('zoning_code', '') or 'n/a'))}<br>"
                f"Rules: {html.escape(str(r.get('reg_rule_names', '') or 'n/a'))}<br>"
                f"Notes: {html.escape(str(r.get('reg_notes', '') or 'manual review required'))}"
            )
            folium.CircleMarker(
                [r.geometry.y, r.geometry.x], radius=5, color="#d97706",
                fill=True, fill_color="#f59e0b", fill_opacity=0.45, weight=2, popup=popup,
            ).add_to(review_fg)
        review_fg.add_to(m)

    # Recommended sites are grouped by physical site so L2+DCFC at one parking
    # facility do not create overlapping markers.
    rec_fg = folium.FeatureGroup(name="Recommended sites", show=True)
    if recommended_sites is not None and not recommended_sites.empty:
        rec = recommended_sites.to_crs("EPSG:4326")
        for site_idx, group in rec.groupby("site_index", sort=True):
            first = group.iloc[0]
            l2 = int(group.loc[group["charger_type"] == "l2", "charger_count"].sum())
            dc = int(group.loc[group["charger_type"] == "dcfc", "charger_count"].sum())
            total = l2 + dc
            name = str(first.get("name", "") or f"Candidate #{site_idx}")
            popup = (
                f"<b>{html.escape(name)}</b><br>Candidate #{site_idx}<br>"
                f"Level 2: {l2}<br>DC fast: {dc}<br>Total chargers: {total}<br>"
                f"Estimated parking spaces: {_fmt(first.get('estimated_spaces', 0))}<br>"
                f"Site cap: {_fmt(first.get('site_max', 0))}<br>"
                f"Zoning: {html.escape(str(first.get('zoning_code', '') or 'n/a'))}<br>"
                f"Regulatory status: {html.escape(str(first.get('regulatory_status', '') or 'not configured'))}"
            )
            folium.Marker(
                [first.geometry.y, first.geometry.x],
                icon=folium.DivIcon(html=(
                    "<div style='width:30px;height:30px;border-radius:50%;"
                    "background:#111827;color:white;border:3px solid #facc15;"
                    "text-align:center;line-height:24px;font-weight:800;font-size:12px;"
                    "box-shadow:0 1px 4px rgba(0,0,0,.35)'>"
                    f"{total}</div>"
                )),
                popup=popup,
                tooltip=f"{name}: {total} charger(s)",
            ).add_to(rec_fg)
    rec_fg.add_to(m)

    if existing is not None and not existing.empty:
        ec = MarkerCluster(name="Existing public chargers", show=False)
        for _, r in existing.to_crs("EPSG:4326").iterrows():
            label = r.get("station_name", r.get("name", "existing charger"))
            l2 = int(r.get("ev_level2_evse_num", 0) or 0)
            dc = int(r.get("ev_dc_fast_num", 0) or 0)
            folium.CircleMarker(
                [r.geometry.y, r.geometry.x], radius=3, color="#374151",
                fill=True, fill_opacity=0.55,
                popup=f"{html.escape(str(label))}<br>L2: {l2}<br>DCFC: {dc}",
            ).add_to(ec)
        ec.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    if save_path:
        m.save(save_path)
    return m

def compare_charts(
    metrics_optimized: dict,
    metrics_greedy: dict,
    save_dir: str = "charts",
    suffix: str = "",
) -> None:
    """Before/after bar chart comparing optimized vs greedy plans.

    Handles both the budget model (demand units served / chargers / spend) and
    the legacy MCLP model (fraction of cells / weight covered / duplication).
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    budget_style = "demand_units_served" in metrics_optimized
    if budget_style:
        labels = ["Demand units served", "Chargers built", "Sites used"]
        opt_vals = [
            metrics_optimized["demand_units_served"],
            metrics_optimized["n_chargers_total"],
            metrics_optimized["n_sites_used"],
        ]
        greedy_vals = [
            metrics_greedy["demand_units_served"],
            metrics_greedy["n_chargers_total"],
            metrics_greedy["n_sites_used"],
        ]
        opt_label = "Optimized (budget model)"
    else:
        labels = ["Fraction of cells covered", "Weight covered", "Duplicated coverage"]
        opt_vals = [
            metrics_optimized["frac_cells_covered"],
            metrics_optimized["frac_weight_covered"],
            metrics_optimized["duplicated_coverage"],
        ]
        greedy_vals = [
            metrics_greedy["frac_cells_covered"],
            metrics_greedy["frac_weight_covered"],
            metrics_greedy["duplicated_coverage"],
        ]
        opt_label = "Optimized MCLP"

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w / 2, greedy_vals, w, label="Greedy baseline", color="#999999")
    ax.bar(x + w / 2, opt_vals, w, label=opt_label, color="#1a9850")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Optimized vs greedy: plan comparison")
    ax.legend()
    for xi, v in zip(x - w / 2, greedy_vals):
        ax.text(xi, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    for xi, v in zip(x + w / 2, opt_vals):
        ax.text(xi, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{save_dir}/before_after{suffix}.png", dpi=150)
    plt.close(fig)


def budget_chart(
    budget: float,
    metrics_optimized: dict,
    metrics_greedy: dict,
    cost_l2: float,
    cost_dcfc: float,
    save_dir: str = "charts",
    suffix: str = "",
) -> None:
    """Chart the charger mix / spend allocation for the budget model."""
    import os
    os.makedirs(save_dir, exist_ok=True)

    def _spend(by_type):
        return (by_type.get("l2", 0) * cost_l2
                + by_type.get("dcfc", 0) * cost_dcfc)

    labels = ["Level 2", "DC fast"]
    opt_l2 = metrics_optimized.get("chargers_by_type", {}).get("l2", 0)
    opt_dcfc = metrics_optimized.get("chargers_by_type", {}).get("dcfc", 0)
    gd_l2 = metrics_greedy.get("chargers_by_type", {}).get("l2", 0)
    gd_dcfc = metrics_greedy.get("chargers_by_type", {}).get("dcfc", 0)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(labels, [opt_l2, opt_dcfc], color=["#1a9850", "#d73027"])
    axes[0].set_title(f"Optimized plan (${_spend(metrics_optimized.get('chargers_by_type', {})):,.0f} of ${budget:,.0f})")
    axes[0].set_ylabel("Chargers")
    axes[1].bar(labels, [gd_l2, gd_dcfc], color=["#1a9850", "#d73027"])
    axes[1].set_title(f"Greedy baseline (${_spend(metrics_greedy.get('chargers_by_type', {})):,.0f} of ${budget:,.0f})")
    axes[1].set_ylabel("Chargers")
    for ax in axes:
        for xi, v in zip(labels, ax.patches):
            ax.text(v.get_x() + v.get_width() / 2, v.get_height(),
                    f"{int(v.get_height())}", ha="center", va="bottom")
        ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{save_dir}/budget_mix{suffix}.png", dpi=150)
    plt.close(fig)


def gamma_sensitivity_chart(
    gamma_values: list[float],
    coverage_fracs: list[float],
    duplicated: list[float],
    save_dir: str = "charts",
    suffix: str = "",
) -> None:
    """Plot how gamma (spreading penalty) trades coverage vs duplication."""
    import os
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(gamma_values, coverage_fracs, "o-", label="Fraction of weight covered",
            color="#1a9850")
    ax.plot(gamma_values, duplicated, "s--", label="Duplicated coverage",
            color="#d73027")
    ax.set_xlabel("gamma (spreading penalty)")
    ax.set_ylabel("value")
    ax.set_title("Sensitivity to gamma")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{save_dir}/gamma_sensitivity{suffix}.png", dpi=150)
    plt.close(fig)


def equity_chart(
    recommended: gpd.GeoDataFrame,
    tracts: gpd.GeoDataFrame,
    save_dir: str = "charts",
    suffix: str = "",
) -> None:
    """Simple equity check: what's the median income near recommended sites vs county."""
    import os
    os.makedirs(save_dir, exist_ok=True)
    if recommended.empty or tracts.empty or "median_income" not in tracts.columns:
        return

    joined = gpd.sjoin(
        recommended[["charger_type", "geometry"]],
        tracts[["geometry", "median_income"]],
        how="left", predicate="within",
    )
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(["L2", "DCFC"],
           [joined.loc[joined["charger_type"] == "l2", "median_income"].median(),
            joined.loc[joined["charger_type"] == "dcfc", "median_income"].median()],
           color=["#1a9850", "#d73027"])
    county_median = tracts["median_income"].median()
    ax.axhline(county_median, color="#333", linestyle="--", label="County median")
    ax.set_ylabel("Median income of served tracts ($)")
    ax.set_title("Equity: income profile near recommended sites")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{save_dir}/equity{suffix}.png", dpi=150)
    plt.close(fig)