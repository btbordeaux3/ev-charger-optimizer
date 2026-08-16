"""Build a routable road network graph from OSM edge features.

We assemble a directed networkx graph from raw Overpass "way" geometry so we can
compute road-network (drive) distances for coverage. This avoids depending on
osmnx's fragile endpoint handling.
"""
from __future__ import annotations

import geopandas as gpd
import networkx as nx
import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import Point

# UTM zones: NC spans 17S (32617) / 17N (32617). We approximate the region with
# a single local metric CRS passed by the caller for accurate driving lengths.
_UTM = 32617

# Assumed network speeds (km/h) by OSM highway class for travel-time coverage.
# Falls back to DEFAULT_DRIVE_SPEED_KPH for classes not listed or unknown.
SPEED_KPH_BY_CLASS = {
    "motorway": 110, "motorway_link": 70,
    "trunk": 90, "trunk_link": 60,
    "primary": 60, "primary_link": 45,
    "secondary": 50, "secondary_link": 40,
    "tertiary": 40, "tertiary_link": 35,
    "unclassified": 35, "residential": 30, "living_street": 12,
    "service": 20, "track": 15, "road": 35,
}
# Non-driving ways (walk/cycle only) get a low drive speed so Dijkstra
# deprioritises them but still allows walk-links in the network.
WALK_ONLY_CLASSES = {"footway", "path", "pedestrian", "cycleway", "steps",
                     "corridor", "sidewalk", "crossing"}
DEFAULT_DRIVE_SPEED_KPH = 35.0
DEFAULT_WALK_SPEED_KPH = 4.8


def _kph_to_mps(kph: float) -> float:
    return kph / 3.6


def speed_kph_for(highway: str | None,
                  speed_map: dict | None = None,
                  default: float = DEFAULT_DRIVE_SPEED_KPH) -> float:
    """Drive speed (km/h) for an OSM highway class."""
    if speed_map is not None and highway and highway in speed_map:
        return float(speed_map[highway])
    if highway in SPEED_KPH_BY_CLASS:
        return float(SPEED_KPH_BY_CLASS[highway])
    if highway in WALK_ONLY_CLASSES:
        return DEFAULT_WALK_SPEED_KPH
    return float(default)


def build_graph(edges: gpd.GeoDataFrame,
                utm_epsg: int = _UTM,
                walk_speed_kph: float = DEFAULT_WALK_SPEED_KPH) -> nx.DiGraph:
    """Build a directed graph from a GeoDataFrame of LineString edges.

    Vertices are de-duplicated by ~0.0001 m grid. Both directions are added.
    Edge 'length' is metric distance; nodes store x/y in projected CRS.
    If the input has a 'highway' column the class is kept per edge and per-edge
    'walk_time' / 'drive_time' (seconds) are added for travel-time coverage.
    """
    G = nx.DiGraph()
    if edges.empty:
        return G

    proj = edges.to_crs(utm_epsg) if edges.crs and edges.crs != utm_epsg else edges
    has_class = "highway" in proj.columns
    walk_mps = _kph_to_mps(walk_speed_kph)
    node_ids: dict[tuple, int] = {}
    next_id = 0

    def _gid(x, y):
        nonlocal next_id
        key = (round(x, 1), round(y, 1))
        if key in node_ids:
            return node_ids[key]
        nid = next_id
        next_id += 1
        node_ids[key] = nid
        G.add_node(nid, x=x, y=y)
        return nid

    for _, row in proj.iterrows():
        geom = row.geometry
        if geom is None or geom.geom_type != "LineString":
            continue
        hw = str(row.get("highway", "") or "") if has_class else None
        drive_speed_mps = _kph_to_mps(speed_kph_for(hw))
        coords = list(geom.coords)
        for i in range(len(coords) - 1):
            (x1, y1), (x2, y2) = coords[i], coords[i + 1]
            u = _gid(x1, y1)
            v = _gid(x2, y2)
            length = float(np.hypot(x2 - x1, y2 - y1))
            if length <= 0:
                continue
            attrs = {"length": length, "walk_time": length / walk_mps,
                     "drive_time": length / drive_speed_mps}
            if hw:
                attrs["highway"] = hw
            G.add_edge(u, v, **attrs)
            G.add_edge(v, u, **attrs)
    return G


def add_travel_times(G: nx.DiGraph,
                     walk_speed_kph: float = DEFAULT_WALK_SPEED_KPH,
                     default_speed_kph: float = DEFAULT_DRIVE_SPEED_KPH,
                     speed_map: dict | None = None) -> None:
    """Ensure every edge carries 'walk_time' and 'drive_time' (seconds).

    For graphs built without travel times (e.g. cached old graphs with only
    'length'), fills them in from the edge length and class-based speeds.
    """
    walk_mps = _kph_to_mps(walk_speed_kph)
    for u, v, d in G.edges(data=True):
        if "walk_time" not in d:
            d["walk_time"] = d["length"] / walk_mps
        if "drive_time" not in d:
            drive_mps = _kph_to_mps(
                speed_kph_for(d.get("highway"), speed_map, default_speed_kph))
            d["drive_time"] = d["length"] / drive_mps


def max_drive_speed_mps(G: nx.DiGraph,
                        default_kph: float = DEFAULT_DRIVE_SPEED_KPH) -> float:
    """Fastest drive speed present on any edge (m/s); safe euclid bound."""
    best = _kph_to_mps(default_kph)
    for _u, _v, d in G.edges(data=True):
        s = d.get("drive_speed_mps")
        if s is not None:
            best = max(best, float(s))
        elif "highway" in d:
            best = max(best, _kph_to_mps(speed_kph_for(d["highway"])))
    return best


def snap_index(G: nx.DiGraph):
    """Return a cKDTree over node coordinates plus the node list."""
    if not G.nodes:
        return None, None
    coords = np.array([[G.nodes[n]["x"], G.nodes[n]["y"]] for n in G.nodes])
    return cKDTree(coords), list(G.nodes)


def snap_points(tree, node_list, points, crs=_UTM) -> list[int | None]:
    """Snap (epsg:4326 or projected) points to nearest graph node ids."""
    if tree is None or not node_list:
        return [None] * len(points)
    ids = []
    for p in points:
        if p is None:
            ids.append(None)
            continue
        # assume points already in graph CRS
        d, idx = tree.query([p.x, p.y])
        ids.append(node_list[int(idx)])
    return ids


def project_points_to(gdf: gpd.GeoDataFrame, utm_epsg: int = _UTM):
    """Return representative (centroid) coordinates in projected CRS.

    Works for Point or Polygon geometries (demand cells are polygons).
    """
    if gdf is None or gdf.empty:
        return []
    proj = gdf.to_crs(utm_epsg) if gdf.crs != utm_epsg else gdf
    pts = []
    for g in proj.geometry:
        if g is None:
            pts.append(None)
        elif g.geom_type == "Point":
            pts.append(Point(g.x, g.y))
        else:
            c = g.centroid
            pts.append(Point(c.x, c.y))
    return pts


def road_distance_matrix(
    G: nx.DiGraph,
    source_points: list[tuple],
    target_points: list[tuple],
    tree,
    node_list,
    max_dist_m: float,
    weight: str = "length",
) -> np.ndarray:
    """Compute a boolean coverage matrix between sources and targets.

    For each source (a candidate site), find which targets are within
    max_dist_m by road. Returns a (len(sources) x len(targets)) bool array.

    This is an optimization: run single-source Dijkstra per source, but only on
    the pre-filtered set of targets within max_dist_m (set by tree + euclid).
    """
    n_src, n_tgt = len(source_points), len(target_points)
    cov = np.zeros((n_src, n_tgt), dtype=bool)
    if tree is None or node_list is None:
        return cov

    for si, src in enumerate(source_points):
        src_node = node_list[int(tree.query([src[0], src[1]])[1])]
        # Candidate target indices within euclidean max_dist (upper bound)
        # Query returns (dists, idx) of points within distance r
        eucl_dists, eucl_idx = tree.query([src[0], src[1]],
                                          k=len(node_list),
                                          distance_upper_bound=max_dist_m)
        reachable_nodes = set()
        for d, ni in zip(eucl_dists, eucl_idx):
            if ni >= len(node_list):
                continue
            if d <= max_dist_m:
                reachable_nodes.add(node_list[int(ni)])
        if not reachable_nodes:
            continue
        # Single-source Dijkstra from src_node, bounded by max_dist
        lengths = nx.single_source_dijkstra_path_length(
            G, src_node, cutoff=max_dist_m, weight=weight
        )
        for ti, tgt in enumerate(target_points):
            tgt_node = node_list[int(tree.query([tgt[0], tgt[1]])[1])]
            if lengths.get(tgt_node, float("inf")) <= max_dist_m:
                cov[si, ti] = True
    return cov