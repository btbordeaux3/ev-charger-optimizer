"""Fetch OpenStreetMap data (road network + candidate public sites) via Overpass.

We use the Overpass API directly (with endpoint fallback) rather than osmnx
because osmnx's default endpoint handling has been unreliable. This gives us:

  1. road network -> for road-network coverage distances
  2. public facility POIs -> universal candidate sites (parking, libraries,
     community centres, town halls, parks) so the tool works in ANY region
     without a city-specific open-data portal.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import networkx as nx
import pandas as pd
import requests
from shapely.geometry import LineString, mapping, shape
import geopandas as gpd

_OVERPASS_ENDPOINTS = [
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass-api.de/api/interpreter",
]

# Number of concurrent tile fetches. Overpass instances tolerate modest
# parallelism; keep it low to avoid 429/timeouts.
_TILE_WORKERS = 4

# Candidate site tags: a site is only eligible if it has real parking, because
# the optimizer may stack several chargers (a "cluster") at one site and each
# charger needs parking spaces. We therefore restrict candidates to parking
# areas (surface lots, multi-storey decks, garages, park-and-rides).
# Each entry maps (osm-key, list-of-values). The final candidate set is filtered
# by *area* (ways) or *capacity* (nodes) so tiny roadside spots never qualify.
DEFAULT_CANDIDATE_TAGS = {
    "amenity": ["parking", "parking_multi_storey", "park_and_ride"],
    "parking": ["surface", "multi-storey", "underground", "garage",
                "park_and_ride"],
}

# A candidate must either be a parking way of at least this area (m^2) or a
# parking node declaring at least this many spaces.
MIN_PARKING_M2 = 1000.0
MIN_PARKING_CAPACITY = 20


class OverpassError(RuntimeError):
    pass


def geocode_place(place: str, timeout: int = 30) -> tuple[float, float, float, float]:
    """Geocode a place name to a bounding box (north, south, east, west)."""
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": place, "format": "json", "limit": 1, "polygon_geojson": 0}
    r = requests.get(url, params=params, timeout=timeout,
                     headers={"User-Agent": "ev-charger-opt/1.0 (research)"})
    r.raise_for_status()
    results = r.json()
    if not results:
        raise OverpassError(f"Could not geocode place: {place}")
    bbox = results[0]["boundingbox"]  # [south, north, west, east]
    south, north, west, east = map(float, bbox)
    return north, south, east, west


def bbox_str(north: float, south: float, east: float, west: float) -> str:
    return f"{south},{west},{north},{east}"


def _post(data: str, max_retries: int = 3) -> dict:
    last_err = None
    for attempt in range(max_retries):
        for ep in _OVERPASS_ENDPOINTS:
            try:
                r = requests.post(
                    ep, data={"data": data}, timeout=90,
                    headers={"User-Agent": "ev-charger-opt/1.0 (research)"},
                )
                if r.status_code == 200:
                    return r.json()
                last_err = OverpassError(f"{ep} -> {r.status_code}")
            except Exception as e:  # connection errors
                last_err = e
            time.sleep(1)
    raise last_err or OverpassError("Overpass query failed")


def _tiles(north, south, east, west, tile_deg):
    """Yield (south, west, north, east) tile dicts covering the bbox."""
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


def fetch_network_gdf(
    north: float, south: float, east: float, west: float,
    cache_dir: str = "data/raw",
    tile_deg: float = 0.05,
) -> gpd.GeoDataFrame:
    """Fetch road network edges in the bbox as a GeoDataFrame of LineStrings.

    The region is tiled so each Overpass query stays small and reliable.
    """
    cache_path = Path(cache_dir) / "osm_network.geojson"
    if cache_path.exists():
        return gpd.read_file(cache_path)

    tiles = list(_tiles(north, south, east, west, tile_deg))

    def fetch_tile(t):
        bb = bbox_str(t["north"], t["south"], t["east"], t["west"])
        query = f"""
[out:json][timeout:120];
(
  way["highway"]({bb});
);
out body geom;
"""
        try:
            data = _post(query)
        except Exception:
            return []
        lines = []
        for el in data.get("elements", []):
            if el["type"] == "way" and el.get("tags", {}).get("highway") \
                    and _valid_way(el):
                geom = el.get("geometry", [])
                # Overpass "geom" returns [{lat,lon},...] -> [(lon,lat),...]
                coords = [(g["lon"], g["lat"]) for g in geom]
                if len(coords) >= 2:
                    lines.append(LineString(coords))
        return lines

    all_gdfs = []
    with ThreadPoolExecutor(max_workers=_TILE_WORKERS) as ex:
        for lines in ex.map(fetch_tile, tiles):
            if lines:
                all_gdfs.append(
                    gpd.GeoDataFrame(
                        {"geometry": lines, "osmid": range(len(lines))},
                        crs="EPSG:4326",
                    )
                )

    if not all_gdfs:
        return gpd.GeoDataFrame({"geometry": [], "osmid": []}, crs="EPSG:4326")
    gdf = gpd.GeoDataFrame(pd.concat(all_gdfs, ignore_index=True), crs="EPSG:4326")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(cache_path, driver="GeoJSON")
    return gdf


def fetch_candidate_sites(
    north: float, south: float, east: float, west: float,
    tags: dict | None = None,
    cache_dir: str = "data/raw",
    tile_deg: float = 0.05,
    min_parking_m2: float = MIN_PARKING_M2,
    min_capacity: int = MIN_PARKING_CAPACITY,
    utm_epsg: int = 32617,
) -> gpd.GeoDataFrame:
    """Fetch candidate parking sites from OSM for the bbox.

    Only sites with real parking qualify (a site may host a *cluster* of
    chargers, so it must have parking area). Ways are kept when their projected
    area >= min_parking_m2; nodes are kept when they declare capacity >=
    min_capacity. Returns a point GeoDataFrame with ``parking_area_m2`` and
    ``parking_capacity`` attributes.
    """
    cache_path = Path(cache_dir) / "osm_candidate_sites.geojson"
    if cache_path.exists():
        return gpd.read_file(cache_path)

    tags = tags or DEFAULT_CANDIDATE_TAGS
    tiles = list(_tiles(north, south, east, west, tile_deg))

    def fetch_tile(t):
        bb = bbox_str(t["north"], t["south"], t["east"], t["west"])
        query = _build_tile_query(tags, bb)
        try:
            data = _post(query)
        except Exception:
            data = {"elements": []}
        rows = []
        for el in data.get("elements", []):
            eltype = el["type"]
            tags_el = el.get("tags", {})
            if eltype == "node":
                rows.append({
                    "lon": el["lon"],
                    "lat": el["lat"],
                    "name": tags_el.get("name", ""),
                    "amenity": tags_el.get("amenity", ""),
                    "parking": tags_el.get("parking", ""),
                    "capacity": _int_capacity(tags_el.get("capacity")),
                    "area_m2": 0.0,
                    "tags": str(tags_el),
                })
            else:  # way with geometry
                coords = [(g["lon"], g["lat"]) for g in el.get("geometry", [])]
                if len(coords) < 3:
                    continue
                poly = shape({"type": "Polygon", "coordinates": [coords]})
                center = el.get("center")
                lon_, lat_ = (center["lon"], center["lat"]) if center \
                    else (poly.centroid.x, poly.centroid.y)
                rows.append({
                    "lon": lon_,
                    "lat": lat_,
                    "name": tags_el.get("name", ""),
                    "amenity": tags_el.get("amenity", ""),
                    "parking": tags_el.get("parking", ""),
                    "capacity": _int_capacity(tags_el.get("capacity")),
                    "area_m2": _area_m2(poly, utm_epsg),
                    "tags": str(tags_el),
                })
        return rows

    all_rows = []
    with ThreadPoolExecutor(max_workers=_TILE_WORKERS) as ex:
        for rows in ex.map(fetch_tile, tiles):
            all_rows.extend(rows)

    if not all_rows:
        return gpd.GeoDataFrame(
            {"name": pd.Series([], dtype="object"),
             "tags": pd.Series([], dtype="object"),
             "parking_area_m2": pd.Series([], dtype="float64"),
             "parking_capacity": pd.Series([], dtype="int64")},
            geometry=gpd.points_from_xy([], []), crs="EPSG:4326",
        )

    df = pd.DataFrame(all_rows)
    df = df.drop_duplicates(subset=["lon", "lat", "name"])
    # Parking-size filter: big enough area, or explicit large capacity.
    keep = (df["area_m2"] >= min_parking_m2) | (df["capacity"] >= min_capacity)
    df = df[keep].copy()
    df["parking_area_m2"] = df["area_m2"].round(1)
    df["parking_capacity"] = df["capacity"]
    df = df.drop(columns=["area_m2", "capacity"])
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]),
                           crs="EPSG:4326")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(cache_path, driver="GeoJSON")
    return gdf


def _int_capacity(value) -> int:
    """Parse an OSM capacity tag (handles '200', '50-70', 'yes', ...)."""
    if not value:
        return 0
    m = re.search(r"\d+", str(value))
    return int(m.group()) if m else 0


def _area_m2(poly, utm_epsg: int) -> float:
    try:
        g = gpd.GeoSeries([poly], crs="EPSG:4326").to_crs(f"EPSG:{utm_epsg}")
        return float(g.area.iloc[0])
    except Exception:
        return 0.0


def _build_tile_query(tags: dict, bb: str) -> str:
    """Build one Overpass query string for a tile bbox.

    Parking candidates are fetched as *nodes* and *ways*. We need way geometry
    to compute parking area (capacity tags are rarely populated), so we request
    ``out body geom``. Ways dominate the payload but parking areas are bounded
    per tile and this is the only way to enforce the "large parking" rule.
    """
    subqueries = []
    for key, values in tags.items():
        if not values:
            continue
        val_pattern = "^(" + "|".join(re.escape(v) for v in values) + ")$"
        subqueries.append(f'  node["{key}"~"{val_pattern}"]({bb});\n')
        subqueries.append(f'  way["{key}"~"{val_pattern}"]({bb});\n')
    query = (
        "[out:json][timeout:90];\n(\n"
        + "".join(subqueries)
        + ");\nout body geom;\n"
    )
    return query


def _valid_way(el) -> bool:
    return "geometry" in el and len(el.get("geometry", [])) >= 2


def get_default_bbox():
    """Placeholder default region; users set their own via config."""
    return 35.98, 35.87, -78.75, -79.05