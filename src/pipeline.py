"""End-to-end pipeline: region config in, recommended plan out.

Flow:
    config YAML -> fetch data -> demand grid -> coverage matrix
                -> MCLP solve (Gurobi/CBC) + greedy baseline
                -> map, charts, ranked site CSV
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.ops import unary_union

from src.config import load_config, RegionConfig
from src.model.demand import build_demand
from src.model.coverage import CoverageMatrix
from src.model.optimize import solve_budget
from src.analysis.baseline import greedy_budget
from src.viz.map import (
    folium_map,
    compare_charts,
    budget_chart,
    equity_chart,
)

log = logging.getLogger(__name__)


def _region_bounds(cfg: RegionConfig) -> tuple[float, float, float, float, object]:
    """Union of county geometries -> (west, south, east, north, region_geom)."""
    import pygris
    gdf = pygris.counties(state=cfg.state_fips, year=2022, cache=True)
    gdf = gdf.to_crs("EPSG:4326")
    wanted = {c.county_fips for c in cfg.counties}
    gdf = gdf[gdf["COUNTYFP"].astype(str).str.zfill(3).isin(wanted)]
    region = gdf.geometry.unary_union
    west, south, east, north = region.bounds
    return west, south, east, north, region


def _load_env_keys(cfg: RegionConfig):
    from dotenv import load_dotenv
    load_dotenv()
    nrel_key = cfg.fetcher.nrel_api_key or os.environ.get("NREL_API_KEY", "")
    census_key = cfg.fetcher.census_api_key or os.environ.get("CENSUS_API_KEY", "")
    return nrel_key, census_key


def fetch_all(
    cfg: RegionConfig,
    west: float, south: float, east: float, north: float,
    nrel_key: str,
    census_key: str,
) -> dict:
    """Fetch every data source; returns a dict of GeoDataFrames."""
    out = {}

    # 1. Existing chargers (NREL) - for context on the map + over-clustering
    if nrel_key:
        from src.fetch.nrel import fetch_stations
        os.environ.setdefault("NREL_API_KEY", nrel_key)
        try:
            stations = fetch_stations(state=cfg.state_abbr)
            if not stations.empty:
                region = _region_bounds(cfg)[4]
                stations = stations[
                    stations.geometry.within(gpd.GeoSeries([region], crs="EPSG:4326").iloc[0])
                ]
            out["existing"] = stations
        except Exception as e:
            log.warning("NREL fetch failed: %s", e)
            out["existing"] = gpd.GeoDataFrame({"geometry": []}, crs="EPSG:4326")
    else:
        out["existing"] = gpd.GeoDataFrame({"geometry": []}, crs="EPSG:4326")

    # 2. AADT traffic (NCDOT)
    from src.fetch.ndot import fetch_aadt
    try:
        aadt = fetch_aadt(counties=[c.name for c in cfg.counties])
        out["aadt"] = aadt
    except Exception as e:
        log.warning("AADT fetch failed: %s", e)
        out["aadt"] = gpd.GeoDataFrame({"geometry": []}, crs="EPSG:4326")

    # 3. OSM road network + candidate sites
    from src.fetch.osm import fetch_network_gdf, fetch_candidate_sites
    cache_dir = cfg.fetcher.cache_dir
    out["network"] = fetch_network_gdf(north, south, east, west, cache_dir=cache_dir)
    out["sites"] = fetch_candidate_sites(
        north, south, east, west,
        cache_dir=cache_dir,
        tile_deg=cfg.fetcher.candidate_tile_deg,
        min_parking_m2=cfg.fetcher.min_parking_m2,
        min_capacity=cfg.fetcher.min_parking_capacity,
    )

    # 4. ACS tracts (equity)
    try:
        from src.fetch.acs import fetch_tracts
        tracts = fetch_tracts(
            cfg.state_fips, cfg.county_fips_list,
            api_key=census_key or os.environ.get("CENSUS_API_KEY"),
        )
        out["tracts"] = tracts
    except Exception as e:
        log.warning("ACS fetch failed: %s", e)
        out["tracts"] = gpd.GeoDataFrame({"geometry": []}, crs="EPSG:4326")

    return out


def run(cfg_path: str, **overrides) -> dict:
    """Run the whole pipeline for a region config file.

    overrides can override any config field (e.g. k_sites=8).
    """
    cfg = load_config(cfg_path)
    for k, v in overrides.items():
        if hasattr(cfg.optimization, k):
            setattr(cfg.optimization, k, v)
        elif hasattr(cfg.demand, k):
            setattr(cfg.demand, k, v)
        elif hasattr(cfg.budget, k):
            setattr(cfg.budget, k, v)

    nrel_key, census_key = _load_env_keys(cfg)
    west, south, east, north, region = _region_bounds(cfg)
    log.info("Region bounds: %s", (west, south, east, north))

    data = fetch_all(cfg, west, south, east, north, nrel_key, census_key)
    network = data["network"]
    sites = data["sites"]
    aadt = data["aadt"]
    tracts = data["tracts"]
    existing = data["existing"]

    # Build demand grid
    demand = build_demand(
        bounds=(west, south, east, north),
        aadt=aadt,
        tracts=tracts,
        cell_size_m=cfg.demand.grid_resolution_m,
        alpha=cfg.demand.alpha_traffic,
        beta=cfg.demand.beta_equity,
        income_weighted=cfg.demand.income_weighted_equity,
        region_geom=region,
    )
    log.info("Demand cells: %d", len(demand))

    # Build coverage matrix on the road network
    from src.fetch.network import build_graph
    G = build_graph(network)
    log.info("Graph nodes: %d, edges: %d", len(G.nodes), len(G.edges))

    cm = CoverageMatrix(
        G=G,
        sites=sites,
        demand=demand,
        radius_l2_m=cfg.coverage.radius_l2_m,
        radius_dcfc_m=cfg.coverage.radius_dcfc_m,
        dcfc_min_aadt=cfg.coverage.dcfc_min_aadt,
    )
    A_l2, A_dcfc = cm.build()
    log.info("Coverage matrix: %s / %s", A_l2.shape, A_dcfc.shape)

    # Solve the capacitated budget model + greedy baseline
    w = demand["demand"].values.astype(float)
    b = cfg.budget
    cost = {"l2": b.cost_l2, "dcfc": b.cost_dcfc}
    capacity = {"l2": b.cap_l2, "dcfc": b.cap_dcfc}

    greedy = greedy_budget(w, A_l2, A_dcfc, budget=b.budget, cost=cost,
                           capacity=capacity, site_max=b.site_max)

    opt = solve_budget(
        w, A_l2, A_dcfc,
        budget=b.budget,
        cost=cost,
        capacity=capacity,
        site_max=b.site_max,
        solver=cfg.optimization.solver,
        time_limit_s=cfg.optimization.time_limit_s,
        mip_gap=cfg.optimization.mip_gap,
        initial_solution=greedy["site_chargers"],
    )

    # ---- outputs ----
    os.makedirs(cfg.output.out_dir, exist_ok=True)
    os.makedirs(cfg.output.charts_dir, exist_ok=True)

    recommended = _site_points(sites, opt["site_chargers"])
    write_results(recommended, cfg.output.results_full, opt)

    folium_map(
        cfg.name, demand, sites, recommended, existing,
        save_path=cfg.output.map_full,
    )

    compare_charts(opt["metrics"], greedy["metrics"],
                   save_dir=cfg.output.charts_dir, suffix="_durham")
    budget_chart(
        b.budget, opt["metrics"], greedy["metrics"],
        cost_l2=b.cost_l2, cost_dcfc=b.cost_dcfc,
        save_dir=cfg.output.charts_dir, suffix="_durham",
    )
    equity_chart(recommended, tracts, save_dir=cfg.output.charts_dir,
                 suffix="_durham")

    # ---- build the locally-hostable dashboard ----
    from src.web.dashboard import build_dashboard
    build_dashboard(
        out_html=f"{cfg.output.out_dir}/index.html",
        map_html=cfg.output.map_full,
        charts_dir=cfg.output.charts_dir,
        results_csv=cfg.output.results_full,
        region=cfg.name,
        metrics_opt=opt["metrics"],
        metrics_greedy=greedy["metrics"],
        k_sites=opt["metrics"]["n_chargers_total"],
        radius_l2_m=cfg.coverage.radius_l2_m,
        radius_dcfc_m=cfg.coverage.radius_dcfc_m,
        gamma=cfg.optimization.gamma,
        solver=cfg.optimization.solver,
        budget=b.budget,
    )

    log.info("Optimized: %s", opt["metrics"])
    log.info("Greedy   : %s", greedy["metrics"])
    log.info("Map -> %s", cfg.output.map_full)
    log.info("CSV -> %s", cfg.output.results_full)
    log.info("Dashboard -> %s/index.html  (serve: python -m src.web.serve)", cfg.output.out_dir)

    return {
        "config": cfg,
        "demand": demand,
        "sites": sites,
        "recommended": recommended,
        "existing": existing,
        "A_l2": A_l2,
        "A_dcfc": A_dcfc,
        "optimized": opt,
        "greedy": greedy,
    }


def _site_points(sites, site_types):
    from src.viz.map import _site_points
    return _site_points(sites, site_types)


def write_results(recommended: gpd.GeoDataFrame, path: str, opt: dict) -> None:
    if recommended.empty:
        log.warning("No sites selected - empty result.")
        return
    df = recommended[["geometry", "charger_type"]].copy()
    df["lon"] = recommended.geometry.x
    df["lat"] = recommended.geometry.y
    if "name" in recommended.columns:
        df["name"] = recommended["name"]
    else:
        df["name"] = ""
    if "charger_count" in recommended.columns:
        df["charger_count"] = recommended["charger_count"]
    else:
        df["charger_count"] = 1
    df = df[["name", "charger_type", "charger_count", "lon", "lat"]]
    df.to_csv(path, index=False)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="EV charger placement pipeline")
    ap.add_argument("config", help="path to region config YAML")
    ap.add_argument("--budget", type=float, default=None,
                    help="override the total budget ($)")
    ap.add_argument("--solver", default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    overrides = {}
    if args.budget is not None:
        overrides["budget"] = args.budget
    if args.solver is not None:
        overrides["solver"] = args.solver

    run(args.config, **overrides)


if __name__ == "__main__":
    main()
