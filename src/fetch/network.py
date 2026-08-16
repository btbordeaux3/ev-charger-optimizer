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


def build_graph(edges: gpd.GeoDataFrame, utm_epsg: int = _UTM) -> nx.DiGraph:
    """Build a directed graph from a GeoDataFrame of LineString edges.

    Vertices are de-duplicated by ~0.0001 m grid. Both directions are added.
    Edge 'length' is metric distance; nodes store x/y in projected CRS.
    """
    G = nx.DiGraph()
    if edges.empty:
        return G

    proj = edges.to_crs(utm_epsg) if edges.crs and edges.crs != utm_epsg else edges
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

    for geom in proj.geometry:
        if geom is None or geom.geom_type != "LineString":
            continue
        coords = list(geom.coords)
        for i in range(len(coords) - 1):
            (x1, y1), (x2, y2) = coords[i], coords[i + 1]
            u = _gid(x1, y1)
            v = _gid(x2, y2)
            length = float(np.hypot(x2 - x1, y2 - y1))
            if length <= 0:
                continue
            G.add_edge(u, v, length=length)
            G.add_edge(v, u, length=length)
    return G


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