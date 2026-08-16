"""Grid-feasibility check that runs AFTER optimization, then resolves.

The budget MILP (``src.model.optimize.solve_budget``) optimizes siting and
sizing of chargers ignoring the distribution grid. This module checks the
proposed plan against per-site hosting capacity and computes a per-site charger
cap so the plan can be re-solved grid-feasibly:

    for each site j:
        load_j   = sum_t chargers[j][t] * power_kw[t] * simultaneity   (kVA)
        if load_j > capacity_j * margin:
            violation; cut site j back proportionally to the overload:
                allowed_j = floor(total_chargers_j * capacity_j * margin / load_j)
        else:
            allowed_j = parking site_max (unchanged)

The pipeline then re-solves the MILP with ``site_max = allowed``; freed budget
flows to other sites. The loop repeats until no violations remain or the
iteration budget is exhausted.
"""
from __future__ import annotations

import numpy as np
import geopandas as gpd
import pandas as pd
from scipy.spatial import cKDTree


def site_capacity(
    sites: gpd.GeoDataFrame,
    grid: gpd.GeoDataFrame,
    capacity_column: str,
    utm_epsg: int = 32617,
) -> np.ndarray:
    """Return per-site available capacity (kVA).

    A site takes the capacity of the grid feature that *contains* its centroid;
    sites outside every feature fall back to the nearest feature (cKDTree over
    feature centroids). Missing/non-numeric capacity becomes ``nan`` (treated as
    "unknown" = unconstrained downstream).
    """
    n = len(sites)
    cap = np.full(n, np.nan, dtype=float)
    if grid is None or grid.empty or capacity_column not in grid.columns:
        return cap
    if "capacity" not in grid.columns:
        grid = grid.copy()
        grid["capacity"] = pd.to_numeric(grid[capacity_column], errors="coerce")

    grid_proj = grid.to_crs(f"EPSG:{utm_epsg}") if str(grid.crs) != f"EPSG:{utm_epsg}" else grid
    site_pts = sites.geometry.to_crs(f"EPSG:{utm_epsg}").centroid

    pts = gpd.GeoDataFrame(geometry=site_pts, crs=f"EPSG:{utm_epsg}")
    joined = gpd.sjoin(
        pts, grid_proj[["geometry", "capacity"]], how="left", predicate="within"
    )
    cap = joined["capacity"].to_numpy(dtype=float)

    # nearest-feature fallback for sites without a containing feature
    missing = np.isnan(cap)
    if missing.any():
        reps = np.array([(p.x, p.y) for p in grid_proj.representative_point()])
        tree = cKDTree(reps)
        site_coords = np.array([(p.x, p.y) for p in site_pts])
        for idx in np.where(missing)[0]:
            _, ni = tree.query(site_coords[idx])
            cap[idx] = grid_proj["capacity"].iloc[int(ni)]
    return cap


def plan_loads(
    chargers: list[tuple[int, str, int]],
    power_kw: dict,
    simultaneity: float = 1.0,
    n_sites: int | None = None,
) -> np.ndarray:
    """Per-site aggregate load (kVA) from a charger plan.

    chargers: [(site_idx, 'l2'|'dcfc', count), ...]
    """
    n = n_sites or (max((c[0] for c in chargers), default=-1) + 1)
    loads = np.zeros(n, dtype=float)
    for j, t, cnt in chargers:
        loads[j] += cnt * float(power_kw.get(t, 0.0)) * simultaneity
    return loads


def check_feasibility(
    chargers: list[tuple[int, str, int]],
    capacity: np.ndarray,
    power_kw: dict,
    simultaneity: float = 1.0,
    margin: float = 0.85,
) -> tuple[list[dict], np.ndarray]:
    """Return (violations, loads).

    violations: [{site, load_kva, capacity_kva, chargers}, ...] where
    load > capacity * margin and capacity is known (finite).
    """
    loads = plan_loads(chargers, power_kw, simultaneity, len(capacity))
    violations = []
    for j, load in enumerate(loads):
        c = capacity[j]
        if load <= 0:
            continue
        if np.isfinite(c) and load > c * margin:
            violations.append({
                "site": int(j),
                "load_kva": float(load),
                "capacity_kva": float(c),
                "n_chargers": _chargers_at(chargers, j),
            })
    return violations, loads


def _chargers_at(chargers, j) -> int:
    return sum(cnt for (sj, _t, cnt) in chargers if sj == j)


def feasible_site_max(
    chargers: list[tuple[int, str, int]],
    capacity: np.ndarray,
    power_kw: dict,
    simultaneity: float = 1.0,
    margin: float = 0.85,
    default_site_max: int = 12,
    previous: np.ndarray | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """Per-site charger cap that fits the grid, plus the violations list.

    For overloaded sites the cap is cut proportionally to the overload, so the
    optimizer keeps the best mix it can while staying within capacity. Sites
    with unknown capacity keep the parking ``default_site_max``.

    ``previous`` is the cap array from an earlier iteration; caps are only ever
    tightened (element-wise min), never raised, which guarantees the pipeline's
    re-solve loop converges instead of oscillating.
    """
    n = len(capacity)
    allowed = np.full(n, float(default_site_max), dtype=float)
    if previous is not None:
        allowed = np.minimum(allowed, np.asarray(previous, dtype=float))
    loads = plan_loads(chargers, power_kw, simultaneity, n)
    violations = []
    for j in range(n):
        total = _chargers_at(chargers, j)
        c = capacity[j]
        load = loads[j]
        if total <= 0:
            continue
        if np.isfinite(c) and load > c * margin:
            frac = c * margin / load
            allowed[j] = max(0.0, np.floor(total * frac))
            violations.append({
                "site": int(j),
                "load_kva": float(load),
                "capacity_kva": float(c),
                "n_chargers": total,
                "allowed": int(allowed[j]),
            })
    return allowed, violations
