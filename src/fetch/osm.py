"""OpenStreetMap candidate parking-site fetcher.

Road routing is handled by :mod:`src.fetch.network` with OSMnx. This module
uses small tiled Overpass queries for parking candidates because they are much
lighter than a full feature download. Crucially, V2 clips candidate points to
THE ACTUAL REGION POLYGON, not merely its bounding box.
"""
from __future__ import annotations

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import Polygon, shape

log = logging.getLogger(__name__)

_OVERPASS_ENDPOINTS = [
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass-api.de/api/interpreter",
]
_TILE_WORKERS = 3

DEFAULT_CANDIDATE_TAGS = {
    "amenity": ["parking", "parking_multi_storey", "park_and_ride"],
    "parking": ["surface", "multi-storey", "underground", "garage", "park_and_ride"],
}


class OverpassError(RuntimeError):
    pass


def geocode_place(place: str, timeout: int = 30) -> tuple[float, float, float, float]:
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": place, "format": "json", "limit": 1, "polygon_geojson": 0}
    r = requests.get(
        url,
        params=params,
        timeout=timeout,
        headers={"User-Agent": "ev-charger-opt/2.0 (research)"},
    )
    r.raise_for_status()
    results = r.json()
    if not results:
        raise OverpassError(f"Could not geocode place: {place}")
    south, north, west, east = map(float, results[0]["boundingbox"])
    return north, south, east, west


def bbox_str(north: float, south: float, east: float, west: float) -> str:
    return f"{south},{west},{north},{east}"


def _post(data: str, max_retries: int = 3) -> dict:
    last_err = None
    for attempt in range(max_retries):
        for ep in _OVERPASS_ENDPOINTS:
            try:
                r = requests.post(
                    ep,
                    data={"data": data},
                    timeout=120,
                    headers={"User-Agent": "ev-charger-opt/2.0 (research)"},
                )
                if r.status_code == 200:
                    return r.json()
                last_err = OverpassError(f"{ep} -> HTTP {r.status_code}")
            except Exception as e:
                last_err = e
        time.sleep(1.0 + attempt)
    raise last_err or OverpassError("Overpass query failed")


def _tiles(north, south, east, west, tile_deg):
    lat = south
    while lat < north:
        lon = west
        while lon < east:
            yield {
                "south": lat,
                "west": lon,
                "north": min(lat + tile_deg, north),
                "east": min(lon + tile_deg, east),
            }
            lon += tile_deg
        lat += tile_deg


def _int_capacity(value) -> int:
    if not value:
        return 0
    m = re.search(r"\d+", str(value).replace(",", ""))
    return int(m.group()) if m else 0


def _build_tile_query(tags: dict, bb: str) -> str:
    sub = []
    for key, values in tags.items():
        if not values:
            continue
        val_pattern = "^(" + "|".join(re.escape(v) for v in values) + ")$"
        sub.append(f'  node["{key}"~"{val_pattern}"]({bb});\n')
        sub.append(f'  way["{key}"~"{val_pattern}"]({bb});\n')
    return "[out:json][timeout:120];\n(\n" + "".join(sub) + ");\nout body center geom;\n"


def _way_polygon(el) -> Polygon | None:
    coords = [(g["lon"], g["lat"]) for g in el.get("geometry", [])]
    if len(coords) < 3:
        return None
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    try:
        poly = Polygon(coords)
        if not poly.is_valid:
            poly = poly.buffer(0)
        return poly if not poly.is_empty else None
    except Exception:
        return None


def _cache_path(cache_dir, namespace, min_parking_m2, min_capacity):
    root = Path(cache_dir) / namespace
    root.mkdir(parents=True, exist_ok=True)
    return root / (
        f"candidate_parking_m2-{int(round(min_parking_m2))}"
        f"_cap-{int(min_capacity)}.geojson"
    )


def fetch_candidate_sites(
    region_geom,
    *,
    metric_crs,
    cache_dir: str = "data/raw",
    namespace: str = "region",
    tile_deg: float = 0.1,
    min_parking_m2: float = 1000.0,
    min_capacity: int = 20,
    tags: dict | None = None,
    force_refresh: bool = False,
) -> gpd.GeoDataFrame:
    """Return parking-rich candidate site points INSIDE ``region_geom``.

    Ways qualify by parking area; nodes qualify by explicit capacity. Candidate
    metadata includes a conservative estimated parking-space count that the
    optimizer later converts into a per-site charger cap.
    """
    path = _cache_path(cache_dir, namespace, min_parking_m2, min_capacity)
    if path.exists() and not force_refresh:
        return gpd.read_file(path)

    tags = tags or DEFAULT_CANDIDATE_TAGS
    west, south, east, north = region_geom.bounds
    tiles = list(_tiles(north, south, east, west, tile_deg))

    def fetch_tile(t):
        bb = bbox_str(t["north"], t["south"], t["east"], t["west"])
        try:
            data = _post(_build_tile_query(tags, bb))
        except Exception as e:
            log.warning("Candidate tile failed (%s): %s", bb, e)
            return []

        rows = []
        for el in data.get("elements", []):
            typ = el.get("type")
            tags_el = el.get("tags", {})
            cap = _int_capacity(tags_el.get("capacity"))
            if typ == "node":
                rows.append({
                    "osm_type": "node",
                    "osm_id": str(el.get("id", "")),
                    "lon": float(el["lon"]),
                    "lat": float(el["lat"]),
                    "name": tags_el.get("name", ""),
                    "amenity": tags_el.get("amenity", ""),
                    "parking": tags_el.get("parking", ""),
                    "parking_capacity": cap,
                    "parking_area_m2": 0.0,
                    "tags": str(tags_el),
                    "_poly": None,
                })
            elif typ == "way":
                poly = _way_polygon(el)
                if poly is None:
                    continue
                rp = poly.representative_point()
                rows.append({
                    "osm_type": "way",
                    "osm_id": str(el.get("id", "")),
                    "lon": float(rp.x),
                    "lat": float(rp.y),
                    "name": tags_el.get("name", ""),
                    "amenity": tags_el.get("amenity", ""),
                    "parking": tags_el.get("parking", ""),
                    "parking_capacity": cap,
                    "parking_area_m2": 0.0,
                    "tags": str(tags_el),
                    "_poly": poly,
                })
        return rows

    rows = []
    with ThreadPoolExecutor(max_workers=_TILE_WORKERS) as ex:
        for chunk in ex.map(fetch_tile, tiles):
            rows.extend(chunk)

    if not rows:
        return _empty_sites()

    df = pd.DataFrame(rows)

    # Compute polygon areas in one vectorized projected operation.
    way_mask = df["_poly"].notna()
    if way_mask.any():
        way_geom = gpd.GeoSeries(df.loc[way_mask, "_poly"].tolist(), crs="EPSG:4326")
        areas = way_geom.to_crs(metric_crs).area.to_numpy()
        df.loc[way_mask, "parking_area_m2"] = areas

    # Size eligibility before point conversion.
    keep = (
        (pd.to_numeric(df["parking_area_m2"], errors="coerce").fillna(0) >= float(min_parking_m2))
        | (pd.to_numeric(df["parking_capacity"], errors="coerce").fillna(0) >= int(min_capacity))
    )
    df = df[keep].copy()
    if df.empty:
        return _empty_sites()

    gdf = gpd.GeoDataFrame(
        df.drop(columns=["_poly"]),
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326",
    )

    # THIS is the key v1 bug fix: bbox fetches are allowed to spill outside a
    # county, but candidates are only eligible if the exact county-union covers
    # their point.
    inside = gdf.geometry.apply(region_geom.covers)
    gdf = gdf[inside].copy()

    # Deduplicate the same lot appearing in overlapping Overpass tiles and
    # collapse node+way duplicates within ~20 m, keeping the richer/larger row.
    if not gdf.empty:
        gm = gdf.to_crs(metric_crs).copy()
        gm["_gx"] = (gm.geometry.x / 20.0).round().astype("int64")
        gm["_gy"] = (gm.geometry.y / 20.0).round().astype("int64")
        gm["_quality"] = (
            pd.to_numeric(gm["parking_capacity"], errors="coerce").fillna(0) * 1000
            + pd.to_numeric(gm["parking_area_m2"], errors="coerce").fillna(0)
        )
        gm = gm.sort_values("_quality", ascending=False).drop_duplicates(["_gx", "_gy"])
        gdf = gm.drop(columns=["_gx", "_gy", "_quality"]).to_crs("EPSG:4326")

    gdf = gdf.reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(path, driver="GeoJSON")
    return gdf


def _empty_sites():
    return gpd.GeoDataFrame(
        {
            "osm_type": pd.Series([], dtype="object"),
            "osm_id": pd.Series([], dtype="object"),
            "name": pd.Series([], dtype="object"),
            "amenity": pd.Series([], dtype="object"),
            "parking": pd.Series([], dtype="object"),
            "parking_capacity": pd.Series([], dtype="int64"),
            "parking_area_m2": pd.Series([], dtype="float64"),
        },
        geometry=gpd.points_from_xy([], []),
        crs="EPSG:4326",
    )


def add_site_capacity_columns(
    sites: gpd.GeoDataFrame,
    *,
    parking_m2_per_space: float = 30.0,
    parking_spaces_per_charger: float = 10.0,
    absolute_site_max: int = 6,
    min_site_max: int = 1,
) -> gpd.GeoDataFrame:
    """Estimate parking spaces and a conservative charger cap per site."""
    sites = sites.copy()
    if sites.empty:
        sites["estimated_spaces"] = pd.Series([], dtype="int64")
        sites["site_max"] = pd.Series([], dtype="int64")
        return sites

    explicit = pd.to_numeric(sites.get("parking_capacity", 0), errors="coerce").fillna(0)
    area = pd.to_numeric(sites.get("parking_area_m2", 0), errors="coerce").fillna(0)
    area_spaces = (area / max(float(parking_m2_per_space), 1.0)).round()
    est = explicit.where(explicit > 0, area_spaces).clip(lower=1)

    raw_cap = (est / max(float(parking_spaces_per_charger), 1.0)).astype(float)
    # At least one charger if the parking site itself passed eligibility.
    cap = raw_cap.apply(lambda x: max(int(min_site_max), int(x // 1)))
    cap = cap.clip(upper=int(absolute_site_max)).astype(int)

    sites["estimated_spaces"] = est.astype(int)
    sites["site_max"] = cap
    return sites
