"""Routing graph construction for walking/driving isochrones.

V2 uses OSMnx to preserve one-way streets, access rules, and the network-type
filtering that the original raw-Overpass graph lost. Graphs are projected to a
local metric CRS and cached per region/network type.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import geopandas as gpd
import networkx as nx

log = logging.getLogger(__name__)


# Reasonable free-flow fallbacks (km/h) for edges without maxspeed data.
DEFAULT_HIGHWAY_SPEEDS = {
    "motorway": 105,
    "motorway_link": 65,
    "trunk": 85,
    "trunk_link": 55,
    "primary": 55,
    "primary_link": 40,
    "secondary": 45,
    "secondary_link": 35,
    "tertiary": 35,
    "tertiary_link": 30,
    "unclassified": 30,
    "residential": 25,
    "living_street": 15,
    "service": 15,
}


def estimate_metric_crs(region_geom):
    """Return a local projected CRS suitable for distance/area calculations."""
    gs = gpd.GeoSeries([region_geom], crs="EPSG:4326")
    crs = gs.estimate_utm_crs()
    if crs is None:
        raise ValueError("Could not estimate a local metric CRS for this region.")
    return crs


def region_cache_key(region_name: str, region_geom) -> str:
    """Stable short key so one region never silently reuses another's cache."""
    digest = hashlib.sha1(region_geom.wkb).hexdigest()[:10]
    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in region_name)
    slug = "-".join(filter(None, slug.split("-")))[:48] or "region"
    return f"{slug}-{digest}"


def _buffer_polygon(region_geom, metric_crs, buffer_m: float):
    gs = gpd.GeoSeries([region_geom], crs="EPSG:4326").to_crs(metric_crs)
    geom = gs.iloc[0].buffer(float(buffer_m))
    return gpd.GeoSeries([geom], crs=metric_crs).to_crs("EPSG:4326").iloc[0]


def _cache_path(cache_dir: str | Path, namespace: str, mode: str, buffer_m: float) -> Path:
    root = Path(cache_dir) / namespace
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{mode}_network_buffer-{int(round(buffer_m))}m.graphml"


def fetch_routing_graph(
    region_geom,
    *,
    mode: str,
    metric_crs,
    cache_dir: str | Path,
    namespace: str,
    buffer_m: float,
    walk_speed_kph: float = 4.8,
    drive_default_speed_kph: float = 35.0,
    force_refresh: bool = False,
):
    """Fetch/cache a projected OSMnx routing graph.

    ``mode`` is ``walk`` or ``drive``. The exact study boundary is buffered only
    for routing. Candidate sites and demand cells remain clipped to the region.
    """
    if mode not in {"walk", "drive"}:
        raise ValueError("mode must be 'walk' or 'drive'")

    try:
        import osmnx as ox
    except ImportError as e:  # pragma: no cover - user environment issue
        raise RuntimeError(
            "OSMnx is required for routing. Install requirements.txt first."
        ) from e

    path = _cache_path(cache_dir, namespace, mode, buffer_m)
    if path.exists() and not force_refresh:
        log.info("Loading cached %s graph: %s", mode, path)
        # OSMnx GraphML I/O is defined for MultiDiGraph objects. Keep the
        # cached representation in that native form, then collapse parallel
        # edges in memory for the SciPy routing matrix.
        G_cached = ox.io.load_graphml(path)
        return ox.convert.to_digraph(G_cached, weight="travel_time")

    polygon = _buffer_polygon(region_geom, metric_crs, buffer_m)
    log.info("Fetching OSM %s graph (buffer %.0f m)", mode, buffer_m)

    # OSMnx keeps its own request cache too, which reduces pressure on Overpass.
    ox.settings.use_cache = True
    ox.settings.cache_folder = str(Path(cache_dir) / "osmnx_http_cache")
    ox.settings.log_console = False

    G = ox.graph.graph_from_polygon(
        polygon,
        network_type=mode,
        simplify=True,
        retain_all=True,
        truncate_by_edge=True,
    )

    if mode == "drive":
        # Prefer posted OSM maxspeed values; fill missing values by road class.
        hwy = dict(DEFAULT_HIGHWAY_SPEEDS)
        fallback = float(drive_default_speed_kph)
        G = ox.routing.add_edge_speeds(G, hwy_speeds=hwy, fallback=fallback)
        G = ox.routing.add_edge_travel_times(G)
    else:
        walk_mps = float(walk_speed_kph) / 3.6
        for _u, _v, _k, data in G.edges(keys=True, data=True):
            length = float(data.get("length", 0.0) or 0.0)
            data["travel_time"] = length / walk_mps if walk_mps > 0 else float("inf")

    # Project after OSMnx has calculated length/speed/travel_time. The graph's
    # x/y are then in meters, which makes nearest-node snapping consistent with
    # candidate and demand geometries.
    G = ox.projection.project_graph(G, to_crs=metric_crs)

    # Persist the projected OSMnx graph in its native MultiDiGraph form.
    # ox.io.save_graphml expects a MultiDiGraph and will fail on a plain
    # nx.DiGraph (NetworkX's OutEdgeView has no ``keys`` argument).
    ox.io.save_graphml(G, filepath=path)

    # scipy.sparse on a MultiDiGraph would sum parallel-edge weights. Collapse
    # parallel edges by MINIMUM travel time only for the in-memory routing
    # representation. This preserves the fastest parallel edge between nodes.
    D = ox.convert.to_digraph(G, weight="travel_time")
    D.graph.update(G.graph)
    return D


def fetch_routing_graphs(
    region_geom,
    *,
    metric_crs,
    cache_dir: str | Path,
    namespace: str,
    walk_buffer_m: float,
    drive_buffer_m: float,
    walk_speed_kph: float,
    drive_default_speed_kph: float,
    force_refresh: bool = False,
) -> tuple[nx.DiGraph, nx.DiGraph]:
    walk = fetch_routing_graph(
        region_geom,
        mode="walk",
        metric_crs=metric_crs,
        cache_dir=cache_dir,
        namespace=namespace,
        buffer_m=walk_buffer_m,
        walk_speed_kph=walk_speed_kph,
        drive_default_speed_kph=drive_default_speed_kph,
        force_refresh=force_refresh,
    )
    drive = fetch_routing_graph(
        region_geom,
        mode="drive",
        metric_crs=metric_crs,
        cache_dir=cache_dir,
        namespace=namespace,
        buffer_m=drive_buffer_m,
        walk_speed_kph=walk_speed_kph,
        drive_default_speed_kph=drive_default_speed_kph,
        force_refresh=force_refresh,
    )
    return walk, drive


def graph_summary(G: nx.DiGraph) -> dict:
    return {
        "nodes": int(G.number_of_nodes()),
        "edges": int(G.number_of_edges()),
        "crs": str(G.graph.get("crs", "")),
    }
