"""Build the equity-aware demand grid.

V2 fixes two important spatial issues from v1:
1. grid cells are clipped to the exact study polygon in a local metric CRS;
2. tract household counts are converted to density and allocated by cell area,
   instead of copying a tract's full household count into every grid cell.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box


def make_grid(
    bounds: tuple[float, float, float, float],
    cell_size_m: int = 400,
    region_geom=None,
    metric_crs=None,
) -> gpd.GeoDataFrame:
    if region_geom is None:
        west, south, east, north = bounds
        region_geom = box(west, south, east, north)
    if metric_crs is None:
        metric_crs = gpd.GeoSeries([region_geom], crs="EPSG:4326").estimate_utm_crs()

    region_m = gpd.GeoSeries([region_geom], crs="EPSG:4326").to_crs(metric_crs).iloc[0]
    minx, miny, maxx, maxy = region_m.bounds

    xs = np.arange(minx, maxx, float(cell_size_m))
    ys = np.arange(miny, maxy, float(cell_size_m))
    polys = [
        box(x, y, x + cell_size_m, y + cell_size_m)
        for x in xs for y in ys
    ]
    cells_m = gpd.GeoDataFrame({"geometry": polys}, crs=metric_crs)
    cells_m = cells_m[cells_m.intersects(region_m)].copy()
    cells_m["geometry"] = cells_m.geometry.intersection(region_m)
    cells_m = cells_m[~cells_m.geometry.is_empty].copy()
    cells_m["cell_area_m2"] = cells_m.geometry.area
    cells_m["cell_id"] = np.arange(len(cells_m), dtype=int)

    cells = cells_m.to_crs("EPSG:4326")
    return cells.reset_index(drop=True)


def _cell_points_metric(cells: gpd.GeoDataFrame, metric_crs):
    cm = cells.to_crs(metric_crs)
    return gpd.GeoDataFrame(
        {"cell_id": cells["cell_id"].to_numpy()},
        geometry=cm.geometry.representative_point(),
        crs=metric_crs,
        index=cells.index,
    )


def assign_traffic(
    cells: gpd.GeoDataFrame,
    aadt: gpd.GeoDataFrame,
    *,
    metric_crs,
    influence_m: float = 600.0,
) -> gpd.GeoDataFrame:
    cells = cells.copy()
    cells["traffic_count"] = 0.0
    if aadt is None or aadt.empty or "aadt" not in aadt.columns:
        return cells

    pts = _cell_points_metric(cells, metric_crs)
    buffers = gpd.GeoDataFrame(
        {"cell_id": pts["cell_id"].to_numpy()},
        geometry=pts.geometry.buffer(float(influence_m)),
        crs=metric_crs,
    )
    traffic = aadt.to_crs(metric_crs)[["aadt", "geometry"]].copy()
    traffic["aadt"] = pd.to_numeric(traffic["aadt"], errors="coerce").fillna(0.0)

    joined = gpd.sjoin(traffic, buffers, how="inner", predicate="within")
    if joined.empty:
        return cells
    sums = joined.groupby("cell_id")["aadt"].sum()
    cells["traffic_count"] = cells["cell_id"].map(sums).fillna(0.0).astype(float)
    return cells


def assign_equity(
    cells: gpd.GeoDataFrame,
    tracts: gpd.GeoDataFrame,
    *,
    metric_crs,
    income_weighted: bool = False,
) -> gpd.GeoDataFrame:
    cells = cells.copy()
    cells["equity_count"] = 0.0
    cells["population_est"] = 0.0
    cells["median_income"] = np.nan
    if tracts is None or tracts.empty or "no_garage_households" not in tracts.columns:
        return cells

    tr = tracts.to_crs(metric_crs).copy()
    tr["tract_area_m2"] = tr.geometry.area.clip(lower=1.0)
    tr["eq_density"] = (
        pd.to_numeric(tr["no_garage_households"], errors="coerce").fillna(0.0)
        / tr["tract_area_m2"]
    )
    pop = pd.to_numeric(tr.get("total_population", 0), errors="coerce").fillna(0.0)
    tr["pop_density"] = pop / tr["tract_area_m2"]
    tr["median_income"] = pd.to_numeric(tr.get("median_income", np.nan), errors="coerce")

    pts = _cell_points_metric(cells, metric_crs)
    joined = gpd.sjoin(
        pts,
        tr[["eq_density", "pop_density", "median_income", "geometry"]],
        how="left",
        predicate="within",
    )
    # If a point lands exactly on a tract boundary, keep one deterministic row.
    joined = joined[~joined.index.duplicated(keep="first")]
    eq_density = joined["eq_density"].reindex(cells.index).fillna(0.0).to_numpy()
    pop_density = joined["pop_density"].reindex(cells.index).fillna(0.0).to_numpy()
    income = joined["median_income"].reindex(cells.index).to_numpy()
    area = pd.to_numeric(cells["cell_area_m2"], errors="coerce").fillna(0.0).to_numpy()

    equity = eq_density * area
    if income_weighted:
        valid = income[np.isfinite(income) & (income > 0)]
        p25 = float(np.percentile(valid, 25)) if valid.size else 1.0
        factor = np.where(np.isfinite(income) & (income > 0), p25 / np.maximum(income, 1), 1.0)
        # Do not allow the income modifier to erase demand; cap extreme values.
        factor = np.clip(factor, 0.5, 2.5)
        equity = equity * factor

    cells["equity_count"] = equity
    cells["population_est"] = pop_density * area
    cells["median_income"] = income
    return cells


def assign_zoning(cells, zoning, *, metric_crs):
    cells = cells.copy()
    cells["zoning_multifamily"] = False
    if zoning is None or zoning.empty or "is_multifamily" not in zoning.columns:
        return cells
    mf = zoning[zoning["is_multifamily"]]
    if mf.empty:
        return cells
    pts = _cell_points_metric(cells, metric_crs)
    joined = gpd.sjoin(pts, mf.to_crs(metric_crs)[["geometry"]], how="left", predicate="within")
    inside = set(joined.loc[joined["index_right"].notna(), "cell_id"].astype(int))
    cells["zoning_multifamily"] = cells["cell_id"].isin(inside)
    return cells


def compute_demand(cells, alpha=1.0, beta=1.0, zoning_multiplier=1.5):
    cells = cells.copy()

    def norm(series):
        s = pd.to_numeric(series, errors="coerce").fillna(0.0).clip(lower=0.0)
        # p99 scaling prevents one anomalous traffic station from flattening the
        # rest of the city while retaining a [0,1] score.
        positive = s[s > 0]
        scale = float(np.percentile(positive, 99)) if len(positive) else 0.0
        if scale <= 0:
            return np.zeros(len(s), dtype=float)
        return np.clip((s / scale).to_numpy(dtype=float), 0.0, 1.0)

    traffic_norm = norm(cells.get("traffic_count", 0.0))
    equity_norm = norm(cells.get("equity_count", 0.0))
    if "zoning_multifamily" in cells.columns:
        boost = cells["zoning_multifamily"].fillna(False).to_numpy()
        equity_norm = np.where(boost, equity_norm * float(zoning_multiplier), equity_norm)

    cells["traffic_norm"] = traffic_norm
    cells["equity_norm"] = equity_norm
    cells["demand"] = float(alpha) * traffic_norm + float(beta) * equity_norm
    return cells


def build_demand(
    bounds,
    aadt,
    tracts,
    cell_size_m=400,
    alpha=1.0,
    beta=1.0,
    income_weighted=False,
    region_geom=None,
    zoning=None,
    zoning_multiplier=1.5,
    metric_crs=None,
    traffic_influence_m=600.0,
):
    cells = make_grid(bounds, cell_size_m, region_geom, metric_crs)
    cells = assign_traffic(
        cells,
        aadt if aadt is not None else gpd.GeoDataFrame(),
        metric_crs=metric_crs,
        influence_m=traffic_influence_m,
    )
    cells = assign_equity(
        cells,
        tracts if tracts is not None else gpd.GeoDataFrame(),
        metric_crs=metric_crs,
        income_weighted=income_weighted,
    )
    cells = assign_zoning(cells, zoning, metric_crs=metric_crs)
    return compute_demand(cells, alpha, beta, zoning_multiplier)
