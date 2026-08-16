"""Visualization: interactive folium map + matplotlib comparison charts.

The map shows demand (chloropleth/heat) with the recommended sites overlaid.
The charts quantify the before/after story:
  - recommended vs greedy coverage fraction (bar)
  - gamma sensitivity: how much spreading sites out costs in raw coverage
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt


def _site_points(
    sites: gpd.GeoDataFrame,
    site_types: list[tuple[int, str]] | list[tuple[int, str, int]],
) -> gpd.GeoDataFrame:
    """Return the recommended sites as a GeoDataFrame.

    Accepts either ``[(j, 'l2'), ...]`` (one charger per site) or
    ``[(j, 'l2', n), ...]`` (cluster model: n chargers at site j). A site with
    multiple chargers gets one row per charger, tagged with a count for display.
    """
    cols = ["name", "charger_type", "charger_count", "geometry"]
    if not site_types:
        return gpd.GeoDataFrame(geometry=[],
                                columns=cols,
                                crs=sites.crs if sites.crs else None)
    rows = []
    for entry in site_types:
        j, t = entry[0], entry[1]
        n = entry[2] if len(entry) > 2 else 1
        rows.append((sites.iloc[j]["name"] if "name" in sites.columns else "",
                     t, n, sites.iloc[j].geometry))
    gdf = gpd.GeoDataFrame(rows, columns=cols, crs=sites.crs if sites.crs else None)
    return gdf


def _budget_metrics_or(metrics: dict) -> dict:
    """Return metrics dict; if the budget-style keys are absent, synthesize them
    so both MCLP and budget pipelines render the same charts."""
    if "demand_units_served" in metrics:
        return metrics
    total = metrics.get("weight_covered", 0.0)
    return {
        "demand_units_served": total,
        "frac_demand_served": metrics.get("frac_weight_covered", 0.0),
        "budget_spent": None,
        "n_chargers_total": metrics.get("n_sites", 0),
        "n_sites_used": metrics.get("n_sites", 0),
        "chargers_by_type": metrics.get("chargers_by_type", {}),
        "n_demand": metrics.get("n_demand", 0),
    }


def folium_map(
    region_name: str,
    demand: gpd.GeoDataFrame,
    sites: gpd.GeoDataFrame,
    recommended_sites: gpd.GeoDataFrame,
    existing: gpd.GeoDataFrame | None = None,
    save_path: str | None = None,
) -> object:
    """Build an interactive folium map of demand + recommended charger sites.

    demand           : cells with a 'demand' weight column
    recommended_sites: selected sites with 'charger_type' in {l2, dcfc}
    existing         : existing public chargers (optional, from NREL)
    """
    import folium
    from folium.plugins import MarkerCluster

    m = folium.Map(location=[demand.geometry.centroid.y.mean(),
                             demand.geometry.centroid.x.mean()], zoom_start=11)

    # Demand layer: color cells by weight
    norm_demand = demand["demand"].values
    vmin, vmax = float(np.nanmin(norm_demand)), float(np.nanmax(norm_demand))
    if not (np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin):
        vmin, vmax = 0.0, 1.0

    def _color(v):
        frac = (v - vmin) / (vmax - vmin)
        # blue -> red
        r = int(255 * frac)
        b = int(255 * (1 - frac))
        return f"#{r:02x}00{b:02x}"

    for _, row in demand.iterrows():
        c = row.geometry.centroid.coords[0]
        folium.Rectangle(
            bounds=[(row.geometry.bounds[1], row.geometry.bounds[0]),
                    (row.geometry.bounds[3], row.geometry.bounds[2])],
            color=_color(row["demand"]),
            fill=True, fill_opacity=0.45, weight=0,
            tooltip=f"demand={row['demand']:.3f}",
        ).add_to(m)

    # Recommended sites (a site can host a cluster: multiple chargers)
    for _, r in recommended_sites.iterrows():
        lon, lat = r.geometry.x, r.geometry.y
        kind = r["charger_type"]
        count = int(r.get("charger_count", 1) or 1)
        color = "#1a9850" if kind == "l2" else "#d73027"
        icon = "plug" if kind == "l2" else "flash"
        name = r.get("name", "site")
        popup = f"{name}<br>{count} x {kind}"
        folium.Marker(
            [lat, lon],
            icon=folium.DivIcon(html=(
                f"<div style='position:relative;width:26px;height:26px;"
                f"border-radius:50%;background:{color};color:white;"
                f"text-align:center;line-height:26px;font-weight:bold;"
                f"font-size:12px;border:2px solid #fff'>"
                f"{count}</div>")),
            popup=popup,
        ).add_to(m)

    # Existing chargers (smaller, semi-transparent)
    if existing is not None and not existing.empty:
        ec = MarkerCluster(name="Existing chargers")
        for _, r in existing.iterrows():
            folium.CircleMarker(
                [r.geometry.y, r.geometry.x], radius=3,
                color="gray", fill=True, fill_opacity=0.5,
                popup=r.get("name", "existing"),
            ).add_to(ec)
        ec.add_to(m)

    folium.LayerControl().add_to(m)
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