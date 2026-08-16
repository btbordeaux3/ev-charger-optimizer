"""Build the weighted demand grid for maximal coverage.

Demand cells are a regular grid over the region. Each cell gets a composite
weight:
    demand_i = alpha * norm(traffic_i) + beta * norm(equity_i)

traffic_i : aggregate AADT of NCDOT stations near the cell
equity_i   : multifamily / no-garage households (from ACS tracts); optionally
             upweighted by a low-income factor
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box
from shapely.ops import unary_union


def make_grid(
    bounds: tuple[float, float, float, float],
    cell_size_m: int = 400,
    region_geom=None,
) -> gpd.GeoDataFrame:
    """Create a GeoDataFrame of square cells covering the region bounds.

    bounds: (west, south, east, north) in EPSG:4326.
    Cells are created on a metric grid then a region polygon clips them.
    """
    # Convert bounds to a metric CRS for a clean square grid (UTM 17N default)
    west, south, east, north = bounds
    bbox = box(west, south, east, north)
    bbox_gdf = gpd.GeoDataFrame(geometry=[bbox], crs="EPSG:4326")
    bbox_m = bbox_gdf.to_crs(32617)
    minx, miny, maxx, maxy = bbox_m.total_bounds

    xs = np.arange(minx, maxx, cell_size_m)
    ys = np.arange(miny, maxy, cell_size_m)
    polygons = []
    for x in xs:
        for y in ys:
            polygons.append(box(x, y, x + cell_size_m, y + cell_size_m))
    cells = gpd.GeoDataFrame(
        {"geometry": polygons}, crs=32617
    ).to_crs("EPSG:4326")

    if region_geom is not None:
        if isinstance(region_geom, gpd.GeoDataFrame):
            region = unary_union(region_geom.geometry)
        else:
            region = region_geom
        cells = cells[cells.geometry.centroid.within(region)].copy()

    # Cell id and centroid (in projected CRS for accurate geometry ops)
    cells["cell_id"] = range(len(cells))
    cells_m = cells.to_crs(32617)
    cells["centroid_proj"] = cells_m.geometry.centroid
    return cells


def assign_traffic(
    cells: gpd.GeoDataFrame,
    aadt: gpd.GeoDataFrame,
    radius_deg: float = 0.003,
) -> gpd.GeoDataFrame:
    """Assign AADT to each cell by summing traffic of stations in a radius.

    Using a spatial index on the AADT stations for efficiency.
    """
    if aadt.empty:
        cells["traffic_count"] = 0.0
        return cells
    if aadt.crs and cells.crs and str(aadt.crs) != str(cells.crs):
        aadt = aadt.to_crs(cells.crs)

    idx = aadt.sindex
    vals = np.zeros(len(cells))
    # radius in degrees is only an approximation for the sindex prefilter;
    # exactness is not required since sjoin/traffic uses the buffer as-is.
    lon_d = radius_deg
    lat_d = radius_deg
    c_lon = cells.geometry.centroid.x.values
    c_lat = cells.geometry.centroid.y.values
    for i in range(len(cells)):
        disp = box(c_lon[i] - lon_d, c_lat[i] - lat_d, c_lon[i] + lon_d, c_lat[i] + lat_d)
        possible = list(idx.intersection(disp.bounds))
        if possible:
            vals[i] = float(aadt.iloc[possible]["aadt"].fillna(0).sum())
    cells["traffic_count"] = vals
    return cells


def assign_equity(
    cells: gpd.GeoDataFrame,
    tracts: gpd.GeoDataFrame,
    income_weighted: bool = False,
) -> gpd.GeoDataFrame:
    """Assign equity (multifamily / no-garage households) to each cell.

    Each cell takes the value of the tract it falls in (via centroid).
    If income_weighted, multiply by a low-income factor.
    """
    if tracts.empty:
        cells["equity_count"] = 0.0
        return cells
    if "no_garage_households" not in tracts.columns:
        cells["equity_count"] = 0.0
        return cells

    # Spatial join cell centroid (projected 32617) to tract
    pts = cells["centroid_proj"]
    tracts_proj = (
        tracts.to_crs(32617) if str(tracts.crs) != "EPSG:32617" else tracts
    )
    joined = gpd.sjoin(
        gpd.GeoDataFrame(geometry=pts, crs=32617),
        tracts_proj[["geometry", "no_garage_households", "median_income"]],
        how="left",
        predicate="within",
    )
    eq = joined["no_garage_households"].fillna(0.0).values
    if income_weighted:
        inc = joined["median_income"].fillna(0.0).values
        # factor = 1 for low income (<= 25th pct), decaying with income
        p25 = np.percentile(inc[inc > 0], 25) if (inc > 0).any() else 1.0
        factor = np.where(inc > 0, p25 / np.maximum(inc, 1), 1.0)
        eq = eq * factor
    cells["equity_count"] = eq
    return cells


def assign_zoning(
    cells: gpd.GeoDataFrame,
    zoning: gpd.GeoDataFrame | None,
) -> gpd.GeoDataFrame:
    """Flag cells whose centroid falls inside a multifamily-designated zone.

    Census tract/block-group counts give *how many* apartment households but not
    *where*. A zoning layer (with a boolean ``is_multifamily`` column) pinpoints
    which parcels are actually designated for apartments, so cells inside those
    zones are upweighted when the demand is combined (see ``compute_demand``).
    """
    cells["zoning_multifamily"] = False
    if zoning is None or zoning.empty or "is_multifamily" not in zoning.columns:
        return cells
    mf = zoning[zoning["is_multifamily"]]
    if mf.empty:
        return cells
    pts = gpd.GeoDataFrame(geometry=cells["centroid_proj"], crs=32617)
    mf_proj = mf.to_crs(32617)
    joined = gpd.sjoin(pts, mf_proj[["geometry"]], how="left", predicate="within")
    inside = joined.index[joined["index_right"].notna()]
    cells["zoning_multifamily"] = cells.index.isin(inside)
    return cells


def compute_demand(
    cells: gpd.GeoDataFrame,
    alpha: float = 1.0,
    beta: float = 1.0,
    zoning_multiplier: float = 1.5,
) -> gpd.GeoDataFrame:
    """Compute the composite demand weight per cell.

    Normalises traffic and equity to [0, 1] then combines with alpha/beta.
    If zoning data is present, cells inside multifamily-designated zones get
    their equity term multiplied by ``zoning_multiplier`` (census + zoning
    combined).
    """
    def norm(s):
        s = pd.to_numeric(s, errors="coerce").fillna(0.0)
        mx = s.max()
        if mx and mx > 0:
            return (s / mx).values
        return np.zeros(len(s))

    traffic_norm = norm(cells["traffic_count"])
    equity_norm = norm(cells["equity_count"])
    if "zoning_multifamily" in cells.columns:
        mf = cells["zoning_multifamily"].fillna(False).to_numpy()
        equity_norm = np.where(mf, equity_norm * zoning_multiplier, equity_norm)
    cells["traffic_norm"] = traffic_norm
    cells["equity_norm"] = equity_norm
    cells["demand"] = alpha * traffic_norm + beta * equity_norm
    return cells


def build_demand(
    bounds: tuple[float, float, float, float],
    aadt: gpd.GeoDataFrame | None,
    tracts: gpd.GeoDataFrame | None,
    cell_size_m: int = 400,
    alpha: float = 1.0,
    beta: float = 1.0,
    income_weighted: bool = False,
    region_geom=None,
    zoning: gpd.GeoDataFrame | None = None,
    zoning_multiplier: float = 1.5,
) -> gpd.GeoDataFrame:
    """End-to-end demand construction for a region."""
    cells = make_grid(bounds, cell_size_m, region_geom)
    if aadt is not None:
        cells = assign_traffic(cells, aadt)
    if tracts is not None:
        cells = assign_equity(cells, tracts, income_weighted)
    else:
        cells["traffic_count"] = cells.get("traffic_count", 0.0)
        cells["equity_count"] = 0.0
    cells = assign_zoning(cells, zoning)
    cells = compute_demand(cells, alpha, beta, zoning_multiplier)
    return cells