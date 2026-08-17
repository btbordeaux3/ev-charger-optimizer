"""End-to-end EV charger siting pipeline (V3).

The pipeline deliberately separates four ideas that were conflated in V1:
1. exact study boundary (candidate/demand eligibility),
2. buffered walk/drive routing networks,
3. capacitated siting optimization, and
4. post-solve accessibility/isochrone-style reporting.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src.analysis.baseline import greedy_budget
from src.config import RegionConfig, load_config
from src.model.coverage import CoverageMatrix
from src.model.demand import build_demand
from src.model.optimize import solve_budget
from src.viz.map import budget_chart, compare_charts, equity_chart, folium_map

log = logging.getLogger(__name__)


def _empty_gdf() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs="EPSG:4326"), crs="EPSG:4326")


def _region_geometry(cfg: RegionConfig):
    """Return exact county-union geometry in EPSG:4326."""
    import pygris

    gdf = pygris.counties(state=cfg.state_fips, year=2024, cache=True).to_crs("EPSG:4326")
    wanted = {c.county_fips.zfill(3) for c in cfg.counties}
    selected = gdf[gdf["COUNTYFP"].astype(str).str.zfill(3).isin(wanted)].copy()
    if selected.empty:
        raise RuntimeError(f"No county geometry matched FIPS {sorted(wanted)}")
    if len(selected) != len(wanted):
        found = set(selected["COUNTYFP"].astype(str).str.zfill(3))
        raise RuntimeError(f"Missing county geometry for FIPS {sorted(wanted - found)}")
    region = selected.geometry.union_all() if hasattr(selected.geometry, "union_all") else selected.geometry.unary_union
    return region


def _load_env_keys(cfg: RegionConfig) -> tuple[str, str]:
    from dotenv import load_dotenv

    load_dotenv()
    return (
        cfg.fetcher.nrel_api_key or os.environ.get("NREL_API_KEY", ""),
        cfg.fetcher.census_api_key or os.environ.get("CENSUS_API_KEY", ""),
    )


def _clip_points(gdf: gpd.GeoDataFrame, region) -> gpd.GeoDataFrame:
    if gdf is None or gdf.empty:
        return _empty_gdf()
    gdf = gdf.to_crs("EPSG:4326").copy()
    return gdf[gdf.geometry.apply(region.covers)].reset_index(drop=True)


def fetch_all(
    cfg: RegionConfig,
    *,
    region,
    metric_crs,
    namespace: str,
    nrel_key: str,
    census_key: str,
    force_refresh: bool = False,
) -> dict:
    """Fetch/cache all spatial inputs for one exact region."""
    out: dict = {}
    west, south, east, north = region.bounds

    # Existing chargers: context + explicit gap-prioritization weight.
    if nrel_key:
        try:
            from src.fetch.nrel import fetch_stations

            existing = fetch_stations(state=cfg.state_abbr, api_key=nrel_key)
            existing = _clip_points(existing, region)
            if cfg.existing.public_only and "access_code" in existing.columns:
                public = existing["access_code"].astype(str).str.contains("public", case=False, na=False)
                existing = existing[public].reset_index(drop=True)
            out["existing"] = existing
        except Exception as e:
            log.warning("Existing-charger fetch failed; continuing without it: %s", e)
            out["existing"] = _empty_gdf()
    else:
        log.info("No NREL/NLR key: existing chargers will be map/context-neutral.")
        out["existing"] = _empty_gdf()

    # NCDOT AADT is intentionally NC-only. Other states still run, but with the
    # traffic component at zero unless another traffic source is added.
    if cfg.state_abbr.upper() == "NC":
        try:
            from src.fetch.ndot import fetch_aadt

            out["aadt"] = _clip_points(fetch_aadt(counties=[c.name for c in cfg.counties]), region)
        except Exception as e:
            log.warning("AADT fetch failed; traffic demand will be zero: %s", e)
            out["aadt"] = _empty_gdf()
    else:
        log.warning("No built-in AADT adapter for %s; traffic demand will be zero.", cfg.state_abbr)
        out["aadt"] = _empty_gdf()

    from src.fetch.network import fetch_routing_graphs
    from src.fetch.osm import add_site_capacity_columns, fetch_candidate_sites

    sites = fetch_candidate_sites(
        region,
        metric_crs=metric_crs,
        cache_dir=cfg.fetcher.cache_dir,
        namespace=namespace,
        tile_deg=cfg.fetcher.candidate_tile_deg,
        min_parking_m2=cfg.fetcher.min_parking_m2,
        min_capacity=cfg.fetcher.min_parking_capacity,
        force_refresh=force_refresh,
    )
    sites = add_site_capacity_columns(
        sites,
        parking_m2_per_space=cfg.budget.parking_m2_per_space,
        parking_spaces_per_charger=cfg.budget.parking_spaces_per_charger,
        absolute_site_max=cfg.budget.site_max,
        min_site_max=cfg.budget.min_site_max,
    )

    # V3 regulatory screening: attach authoritative parcel/zoning context and
    # apply config-driven parking/EV rules. Unknown context is explicit; it is
    # never silently treated as approval.
    if cfg.regulations.enabled:
        from src.fetch.regulations import attach_parcel_context, apply_regulatory_rules

        sites = attach_parcel_context(
            sites,
            cfg.regulations,
            cache_dir=cfg.fetcher.cache_dir,
            namespace=namespace,
            force_refresh=force_refresh,
        )
        sites = apply_regulatory_rules(sites, cfg.regulations)
    out["sites"] = sites

    walk_graph, drive_graph = fetch_routing_graphs(
        region,
        metric_crs=metric_crs,
        cache_dir=cfg.fetcher.cache_dir,
        namespace=namespace,
        walk_buffer_m=cfg.fetcher.walk_network_buffer_m,
        drive_buffer_m=cfg.fetcher.drive_network_buffer_m,
        walk_speed_kph=cfg.coverage.walk_speed_kph,
        drive_default_speed_kph=cfg.coverage.drive_default_speed_kph,
        force_refresh=force_refresh,
    )
    out["walk_graph"] = walk_graph
    out["drive_graph"] = drive_graph

    try:
        from src.fetch.acs import fetch_tracts

        if not census_key:
            raise RuntimeError("no Census API key configured")
        out["tracts"] = fetch_tracts(cfg.state_fips, cfg.county_fips_list, api_key=census_key, year=2024)
    except Exception as e:
        log.warning("ACS fetch failed; equity demand will be zero: %s", e)
        out["tracts"] = _empty_gdf()

    from src.fetch.grid import fetch_grid_capacity
    from src.fetch.zoning import fetch_zoning

    out["zoning"] = fetch_zoning(
        cfg.zoning.layers,
        cfg.zoning.multifamily_column,
        cfg.zoning.multifamily_values,
        bbox=(west, south, east, north),
    )
    out["grid"] = fetch_grid_capacity(
        cfg.grid.layers,
        cfg.grid.capacity_column,
        bbox=(west, south, east, north),
    )
    return out


def _spacing_pairs(sites: gpd.GeoDataFrame, metric_crs, min_spacing_m: float) -> list[tuple[int, int]]:
    if min_spacing_m <= 0 or len(sites) < 2:
        return []
    sm = sites.to_crs(metric_crs)
    xy = np.column_stack([sm.geometry.x.to_numpy(), sm.geometry.y.to_numpy()])
    pairs = cKDTree(xy).query_pairs(float(min_spacing_m), output_type="set")
    return sorted((int(a), int(b)) for a, b in pairs)


def _existing_coverage_mask(
    existing: gpd.GeoDataFrame,
    *,
    walk_graph,
    drive_graph,
    demand,
    metric_crs,
    cfg: RegionConfig,
) -> np.ndarray:
    """Cells already accessible to at least one current charger."""
    mask = np.zeros(len(demand), dtype=bool)
    if existing is None or existing.empty or len(demand) == 0:
        return mask

    cm = CoverageMatrix(
        walk_graph=walk_graph,
        drive_graph=drive_graph,
        sites=existing,
        demand=demand,
        metric_crs=metric_crs,
        l2_walk_time_min=cfg.coverage.l2_walk_time_min,
        dcfc_drive_time_min=cfg.coverage.dcfc_drive_time_min,
        dcfc_min_aadt=0.0,
        direction=cfg.coverage.direction,
        backend=cfg.coverage.routing_backend,
        n_workers=cfg.coverage.routing_workers,
        chunk_size=cfg.coverage.routing_chunk_size,
    )
    A_l2, A_dc = cm.build()
    l2_ports = pd.to_numeric(existing.get("ev_level2_evse_num", 0), errors="coerce").fillna(0).to_numpy() > 0
    dc_ports = pd.to_numeric(existing.get("ev_dc_fast_num", 0), errors="coerce").fillna(0).to_numpy() > 0
    if l2_ports.any():
        mask |= A_l2[l2_ports].any(axis=0)
    if dc_ports.any():
        mask |= A_dc[dc_ports].any(axis=0)
    return mask


def _coverage_outputs(
    cm: CoverageMatrix,
    demand: gpd.GeoDataFrame,
    site_chargers: list[tuple[int, str, int]],
    thresholds: list[float],
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Create accessibility metrics and unioned demand-cell service layers."""
    thresholds = sorted({float(t) for t in thresholds if float(t) > 0})
    if not thresholds:
        return pd.DataFrame(), {}
    max_t = max(thresholds)
    l2_sites = sorted({j for j, typ, n in site_chargers if typ == "l2" and n > 0})
    dc_sites = sorted({j for j, typ, n in site_chargers if typ == "dcfc" and n > 0})
    min_l2 = cm.min_times_for_selected("l2", l2_sites, max_t) / 60.0
    min_dc = cm.min_times_for_selected("dcfc", dc_sites, max_t) / 60.0

    raw_w = pd.to_numeric(demand["demand"], errors="coerce").fillna(0).to_numpy(float)
    opt_w = pd.to_numeric(demand.get("optimization_demand", demand["demand"]), errors="coerce").fillna(0).to_numpy(float)
    pop = pd.to_numeric(demand.get("population_est", 0), errors="coerce").fillna(0).to_numpy(float)
    area = pd.to_numeric(demand.get("cell_area_m2", 0), errors="coerce").fillna(0).to_numpy(float)

    def frac(values, mask):
        den = float(np.nansum(values))
        return float(np.nansum(values[mask]) / den) if den > 0 else 0.0

    rows = []
    layers: dict[str, object] = {}
    for t in thresholds:
        mode_masks = {
            "l2_walk": np.isfinite(min_l2) & (min_l2 <= t),
            "dcfc_drive": np.isfinite(min_dc) & (min_dc <= t),
        }
        mode_masks["either"] = mode_masks["l2_walk"] | mode_masks["dcfc_drive"]
        for mode, mask in mode_masks.items():
            rows.append({
                "mode": mode,
                "threshold_min": t,
                "cells_covered": int(mask.sum()),
                "weighted_demand_fraction": frac(raw_w, mask),
                "optimization_demand_fraction": frac(opt_w, mask),
                "population_fraction": frac(pop, mask),
                "land_area_fraction": frac(area, mask),
            })
            if mask.any():
                geom = demand.loc[mask, "geometry"]
                layers[f"{mode}|{t:g}"] = geom.union_all() if hasattr(geom, "union_all") else geom.unary_union
    return pd.DataFrame(rows), layers


def _run_solver(cfg, demand_w, A_l2, A_dcfc, site_max_arr, spacing_pairs, initial_solution):
    b, o = cfg.budget, cfg.optimization
    return solve_budget(
        demand_w,
        A_l2,
        A_dcfc,
        budget=b.budget,
        cost={"l2": b.cost_l2, "dcfc": b.cost_dcfc},
        capacity={"l2": b.cap_l2, "dcfc": b.cap_dcfc},
        site_max=site_max_arr,
        solver=o.solver,
        time_limit_s=o.time_limit_s,
        mip_gap=o.mip_gap,
        initial_solution=initial_solution,
        min_l2=o.min_l2,
        min_dcfc=o.min_dcfc,
        min_sites_used=o.min_sites_used,
        max_sites_used=o.max_sites_used,
        spacing_pairs=spacing_pairs,
        coverage_bonus=o.coverage_bonus,
    )


def run(cfg_path: str, *, force_refresh: bool = False, **overrides) -> dict:
    cfg = load_config(cfg_path)
    for k, v in overrides.items():
        if v is None:
            continue
        if hasattr(cfg.optimization, k):
            setattr(cfg.optimization, k, v)
        elif hasattr(cfg.demand, k):
            setattr(cfg.demand, k, v)
        elif hasattr(cfg.budget, k):
            setattr(cfg.budget, k, v)
        elif hasattr(cfg.coverage, k):
            setattr(cfg.coverage, k, v)

    from src.fetch.network import estimate_metric_crs, graph_summary, region_cache_key

    nrel_key, census_key = _load_env_keys(cfg)
    region = _region_geometry(cfg)
    metric_crs = estimate_metric_crs(region)
    namespace = region_cache_key(cfg.name, region)
    west, south, east, north = region.bounds
    log.info("Region: %s; bounds=%s; metric CRS=%s", cfg.name, (west, south, east, north), metric_crs)
    log.info("Cache namespace: %s", namespace)

    data = fetch_all(
        cfg,
        region=region,
        metric_crs=metric_crs,
        namespace=namespace,
        nrel_key=nrel_key,
        census_key=census_key,
        force_refresh=force_refresh,
    )
    sites = data["sites"]
    if sites.empty:
        raise RuntimeError(
            "No eligible parking candidate sites were found inside the study boundary. "
            "Try lowering min_parking_m2/min_parking_capacity or inspect OSM coverage."
        )
    usable_sites = int((pd.to_numeric(sites.get("site_max", 0), errors="coerce").fillna(0) > 0).sum())
    if usable_sites < cfg.optimization.min_sites_used:
        raise RuntimeError(
            f"Only {usable_sites} usable candidate sites remain after parking/regulatory screening, but "
            f"min_sites_used={cfg.optimization.min_sites_used}. Inspect regulatory_audit.csv, "
            "lower candidate thresholds, or revise the jurisdiction profile rather than hiding the issue."
        )

    demand = build_demand(
        bounds=(west, south, east, north),
        aadt=data["aadt"],
        tracts=data["tracts"],
        cell_size_m=cfg.demand.grid_resolution_m,
        alpha=cfg.demand.alpha_traffic,
        beta=cfg.demand.beta_equity,
        income_weighted=cfg.demand.income_weighted_equity,
        region_geom=region,
        zoning=data["zoning"],
        zoning_multiplier=cfg.zoning.cell_multiplier,
        metric_crs=metric_crs,
        traffic_influence_m=cfg.demand.traffic_influence_m,
    )
    if demand.empty:
        raise RuntimeError("Demand grid is empty after clipping to the region.")
    log.info("Demand cells=%d; candidate sites=%d", len(demand), len(sites))
    log.info("Walk graph: %s", graph_summary(data["walk_graph"]))
    log.info("Drive graph: %s", graph_summary(data["drive_graph"]))

    cm = CoverageMatrix(
        walk_graph=data["walk_graph"],
        drive_graph=data["drive_graph"],
        sites=sites,
        demand=demand,
        metric_crs=metric_crs,
        l2_walk_time_min=cfg.coverage.l2_walk_time_min,
        dcfc_drive_time_min=cfg.coverage.dcfc_drive_time_min,
        dcfc_min_aadt=cfg.coverage.dcfc_min_aadt,
        direction=cfg.coverage.direction,
        backend=cfg.coverage.routing_backend,
        n_workers=cfg.coverage.routing_workers,
        chunk_size=cfg.coverage.routing_chunk_size,
    )
    A_l2, A_dcfc = cm.build()
    log.info("Routing backend=%s; coverage matrices=%s/%s", cm.backend_used, A_l2.shape, A_dcfc.shape)

    # Existing infrastructure reduces, but does not erase, demand weight in
    # already-served cells. This makes "fill gaps" behavior explicit.
    existing_mask = _existing_coverage_mask(
        data["existing"],
        walk_graph=data["walk_graph"],
        drive_graph=data["drive_graph"],
        demand=demand,
        metric_crs=metric_crs,
        cfg=cfg,
    )
    raw_w = pd.to_numeric(demand["demand"], errors="coerce").fillna(0).to_numpy(float)
    opt_w = raw_w.copy()
    opt_w[existing_mask] *= float(cfg.existing.covered_demand_multiplier)
    demand["already_served_existing"] = existing_mask
    demand["optimization_demand"] = opt_w

    if float(opt_w.sum()) <= 0:
        raise RuntimeError(
            "All demand weights are zero. For Durham, verify the NCDOT fetch and activate "
            "your Census API key so the traffic/equity inputs are actually populated."
        )

    site_max_arr = pd.to_numeric(sites["site_max"], errors="coerce").fillna(cfg.budget.min_site_max).to_numpy(int)
    spacing_pairs = _spacing_pairs(sites, metric_crs, cfg.optimization.min_site_spacing_m)

    cost = {"l2": cfg.budget.cost_l2, "dcfc": cfg.budget.cost_dcfc}
    capacity = {"l2": cfg.budget.cap_l2, "dcfc": cfg.budget.cap_dcfc}
    greedy = greedy_budget(
        opt_w,
        A_l2,
        A_dcfc,
        budget=cfg.budget.budget,
        cost=cost,
        capacity=capacity,
        site_max=site_max_arr,
    )
    opt = _run_solver(cfg, opt_w, A_l2, A_dcfc, site_max_arr, spacing_pairs, greedy["site_chargers"])
    if not opt.get("metrics"):
        raise RuntimeError(f"Optimizer returned no feasible solution (status={opt.get('status')}).")

    # Optional grid-feasibility re-solves. Never relax the parking-derived cap.
    grid_status = {"enabled": bool(cfg.grid.enabled), "iterations": 0, "violations": [], "constrained": False, "final_feasible": False}
    grid_gdf = data["grid"]
    if cfg.grid.enabled and grid_gdf is not None and not grid_gdf.empty:
        from src.model.grid_check import feasible_site_max, site_capacity

        grid_cap = site_capacity(sites, grid_gdf, cfg.grid.capacity_column, metric_crs=metric_crs)
        allowed = site_max_arr.astype(float).copy()
        violations = [1]  # enter loop once to evaluate plan
        it = 0
        while it < cfg.grid.max_iterations:
            candidate_allowed, violations = feasible_site_max(
                opt["site_chargers"],
                grid_cap,
                power_kw=cfg.grid.charger_power_kw,
                simultaneity=cfg.grid.simultaneity,
                margin=cfg.grid.margin,
                default_site_max=int(site_max_arr.max()) if len(site_max_arr) else 0,
                previous=allowed,
            )
            candidate_allowed = np.minimum(candidate_allowed, site_max_arr)
            if not violations:
                allowed = candidate_allowed
                break
            it += 1
            allowed = candidate_allowed
            log.warning("Grid feasibility iteration %d: %d overloaded site(s)", it, len(violations))
            opt = _run_solver(cfg, opt_w, A_l2, A_dcfc, allowed, spacing_pairs, opt["site_chargers"])
            if not opt.get("metrics"):
                raise RuntimeError("Grid-constrained re-solve became infeasible.")
        grid_status.update({
            "iterations": it,
            "violations": violations,
            "constrained": it > 0,
            "final_feasible": not violations,
        })

    coverage_summary, service_layers = _coverage_outputs(
        cm, demand, opt["site_chargers"], cfg.coverage.map_thresholds_min
    )

    Path(cfg.output.out_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.output.charts_dir).mkdir(parents=True, exist_ok=True)
    recommended = _site_points(sites, opt["site_chargers"])
    write_results(recommended, cfg.output.results_full)
    coverage_summary.to_csv(cfg.output.coverage_full, index=False)
    if cfg.regulations.enabled:
        from src.fetch.regulations import regulatory_audit_frame

        regulatory_audit_frame(sites).to_csv(cfg.output.regulatory_full, index=False)

    folium_map(
        cfg.name,
        demand,
        sites,
        recommended,
        region_geom=region,
        existing=data["existing"],
        service_layers=service_layers,
        save_path=cfg.output.map_full,
    )
    compare_charts(opt["metrics"], greedy["metrics"], save_dir=cfg.output.charts_dir)
    budget_chart(
        cfg.budget.budget,
        opt["metrics"],
        greedy["metrics"],
        cost_l2=cfg.budget.cost_l2,
        cost_dcfc=cfg.budget.cost_dcfc,
        save_dir=cfg.output.charts_dir,
    )
    equity_chart(recommended, data["tracts"], save_dir=cfg.output.charts_dir)

    from src.web.dashboard import build_dashboard

    build_dashboard(
        out_html=str(Path(cfg.output.out_dir) / "index.html"),
        map_html=cfg.output.map_full,
        charts_dir=cfg.output.charts_dir,
        results_csv=cfg.output.results_full,
        coverage_csv=cfg.output.coverage_full,
        regulatory_csv=cfg.output.regulatory_full if cfg.regulations.enabled else "",
        region=cfg.name,
        metrics_opt=opt["metrics"],
        metrics_greedy=greedy["metrics"],
        l2_walk_time_min=cfg.coverage.l2_walk_time_min,
        dcfc_drive_time_min=cfg.coverage.dcfc_drive_time_min,
        solver=opt["metrics"].get("solver_used") or cfg.optimization.solver,
        routing_backend=cm.backend_used,
        budget=cfg.budget.budget,
        regulations_enabled=cfg.regulations.enabled,
        regulations_jurisdiction=cfg.regulations.jurisdiction,
        regulations_as_of=cfg.regulations.as_of_date,
    )

    log.info("Optimized: %s", opt["metrics"])
    log.info("Greedy: %s", greedy["metrics"])
    log.info("Recommended sites -> %s", cfg.output.results_full)
    log.info("Coverage summary -> %s", cfg.output.coverage_full)
    if cfg.regulations.enabled:
        log.info("Regulatory audit -> %s", cfg.output.regulatory_full)
    log.info("Dashboard -> %s", Path(cfg.output.out_dir) / "index.html")

    return {
        "config": cfg,
        "region": region,
        "metric_crs": metric_crs,
        "demand": demand,
        "sites": sites,
        "recommended": recommended,
        "existing": data["existing"],
        "A_l2": A_l2,
        "A_dcfc": A_dcfc,
        "optimized": opt,
        "greedy": greedy,
        "coverage_summary": coverage_summary,
        "regulatory_audit": (
            __import__("src.fetch.regulations", fromlist=["regulatory_audit_frame"]).regulatory_audit_frame(sites)
            if cfg.regulations.enabled else pd.DataFrame()
        ),
        "grid": grid_status,
    }


def _site_points(sites, site_types):
    from src.viz.map import _site_points as _viz_site_points

    return _viz_site_points(sites, site_types)


def write_results(recommended: gpd.GeoDataFrame, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if recommended.empty:
        pd.DataFrame(columns=["site_index", "name", "charger_type", "charger_count", "lon", "lat"]).to_csv(path, index=False)
        return
    out = recommended.drop(columns="geometry").copy()
    points = recommended.to_crs("EPSG:4326").geometry
    out["lon"] = points.x.to_numpy()
    out["lat"] = points.y.to_numpy()
    preferred = [
        "site_index", "name", "charger_type", "charger_count", "estimated_spaces",
        "site_max_pre_regulation", "planning_ev_share_cap", "site_max",
        "zoning_code", "land_class", "parcel_id", "parcel_address",
        "reg_parking_min_spaces", "reg_parking_max_factor", "reg_ev_min_installed",
        "regulatory_status", "reg_manual_review", "reg_rule_names", "reg_notes",
        "parking_capacity", "parking_area_m2", "osm_type", "osm_id", "lon", "lat",
    ]
    cols = [c for c in preferred if c in out.columns] + [c for c in out.columns if c not in preferred]
    out[cols].to_csv(path, index=False)


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description="EV charger placement + accessibility pipeline")
    ap.add_argument("config", help="path to region config YAML")
    ap.add_argument("--budget", type=float, default=None, help="override total budget ($)")
    ap.add_argument("--solver", choices=["auto", "gurobi", "cbc"], default=None)
    ap.add_argument("--routing-backend", choices=["auto", "scipy", "cugraph"], default=None)
    ap.add_argument("--refresh", action="store_true", help="refetch OSM candidates and routing graphs")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    run(
        args.config,
        force_refresh=args.refresh,
        budget=args.budget,
        solver=args.solver,
        routing_backend=args.routing_backend,
    )


if __name__ == "__main__":
    main()
