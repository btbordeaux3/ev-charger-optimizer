"""Build the coverage matrix between candidate sites and demand cells.

Coverage is decided by *travel time*, not distance: a site covers a demand
cell when the OSM network travel time between them is within the charger
type's isochrone cap (default 20 minutes):

  - Level 2 (slow) chargers serve local residents, so we use WALKING time:
    per-edge walk_time = length / walking speed (4.8 km/h default). This is a
    true pedestrian isochrone over the road network, no detour fudge factor.

  - DC fast chargers serve corridor/commuter traffic, so we use DRIVING time:
    per-edge drive_time = length / class-based speed (from OSM `highway` tags,
    e.g. motorway 110 km/h ... residential 30 km/h). This is a driving
    isochrone in the "20-minute city" sense. Walk-only ways (footways, paths,
    cycleways) are excluded from the drive graph.

Performance: with a 20-minute drive cap the isochrone spans most of the region,
so scipy's C-level Dijkstra runs per site (chunked to bound memory) across a
process pool. `limit` stops label growth past the time cap — exact, same result
as a cutoff Dijkstra.
"""
from __future__ import annotations

import multiprocessing
import numpy as np
import geopandas as gpd
import networkx as nx
from scipy.sparse.csgraph import dijkstra

from src.fetch.network import (
    add_travel_times,
    snap_index,
    project_points_to,
    WALK_ONLY_CLASSES,
)

# Per-worker globals set by _worker_init (inherited copy-on-write via fork).
_WORKER = {}


def _worker_init(csr, demand_node_pos, limit_s, n_demand):
    _WORKER["csr"] = csr
    _WORKER["dpos"] = demand_node_pos
    _WORKER["limit"] = limit_s
    _WORKER["n_demand"] = n_demand


def _chunk_dijkstra(idx: np.ndarray) -> np.ndarray:
    """Travel times from a chunk of site nodes to all demand cells (seconds)."""
    dist = dijkstra(_WORKER["csr"], directed=True, indices=idx,
                    limit=_WORKER["limit"], return_predecessors=False)
    return np.asarray(dist[:, _WORKER["dpos"]], dtype=np.float64)


def _drive_weight(u, v, d):
    """Drive-time edge weight; None drops walk-only ways from the graph."""
    if d.get("highway") in WALK_ONLY_CLASSES:
        return None
    return d["drive_time"]


class CoverageMatrix:
    """Isochrone coverage between sites and demand cells per charger type.

    L2 coverage is a walking-time isochrone (`l2_walk_time_min`).
    DCFC coverage is a driving-time isochrone (`dcfc_drive_time_min`).
    """

    def __init__(
        self,
        G: nx.DiGraph,
        sites: gpd.GeoDataFrame,
        demand: gpd.GeoDataFrame,
        l2_walk_time_min: float = 20.0,
        dcfc_drive_time_min: float = 20.0,
        walk_speed_kph: float = 4.8,
        drive_default_speed_kph: float = 35.0,
        dcfc_min_aadt: float = 0.0,
        utm_epsg: int = 32617,
        n_jobs: int = -1,
    ):
        self.G = G
        self.sites = sites
        self.demand = demand
        self.l2_walk_time_min = l2_walk_time_min
        self.dcfc_drive_time_min = dcfc_drive_time_min
        self.dcfc_min_aadt = dcfc_min_aadt
        self.utm_epsg = utm_epsg
        self.n_jobs = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs

        self._walk_cutoff_s = l2_walk_time_min * 60.0
        self._drive_cutoff_s = dcfc_drive_time_min * 60.0

        self.site_pts = project_points_to(sites, utm_epsg)
        self.demand_pts = project_points_to(demand, utm_epsg)
        self._tree, self._node_list = snap_index(G)
        self._pos = {n: i for i, n in enumerate(self._node_list)} if self._node_list else {}

        self._csr = {}
        if G is not None and G.number_of_edges():
            add_travel_times(G, walk_speed_kph, drive_default_speed_kph)
            self._csr["walk_time"] = nx.to_scipy_sparse_array(
                G, nodelist=self._node_list, weight="walk_time",
                format="csr", dtype=float)
            self._csr["drive_time"] = nx.to_scipy_sparse_array(
                G, nodelist=self._node_list, weight=_drive_weight,
                format="csr", dtype=float)

    def _snap_all(self, pts_array):
        """Snap an array of (x,y) coords to node ids via the KD tree."""
        if self._tree is None or len(pts_array) == 0:
            return [None] * len(pts_array)
        dist, idx = self._tree.query(pts_array)
        return [self._node_list[int(i)] for i in idx]

    def _dist_to_cells(self, site_idx, weight: str, cutoff_s: float,
                       demand_node_pos, chunk: int = 24) -> np.ndarray:
        """(n_sites, n_demand) min travel time site->cell within cutoff."""
        n = len(site_idx)
        if n == 0 or self._csr.get(weight) is None:
            return np.full((n, len(self.demand)), np.inf)
        csr = self._csr[weight]
        idx = np.asarray(site_idx, dtype=int)
        chunks = [idx[s:s + chunk] for s in range(0, n, chunk)]

        if self.n_jobs <= 1 or len(chunks) <= 1:
            _worker_init(csr, demand_node_pos, cutoff_s, len(self.demand))
            parts = [_chunk_dijkstra(c) for c in chunks]
        else:
            ctx = multiprocessing.get_context("fork")
            with ctx.Pool(processes=self.n_jobs, initializer=_worker_init,
                          initargs=(csr, demand_node_pos, cutoff_s,
                                    len(self.demand))) as pool:
                parts = pool.map(_chunk_dijkstra, chunks, chunksize=1)
        return np.vstack(parts) if parts else np.full((n, len(self.demand)), np.inf)

    def build(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (A_l2, A_dcfc) boolean coverage matrices.

        A_l2[s, i]   == True if site s is within ``l2_walk_time_min`` of cell
                       i by WALKING the OSM network.
        A_dcfc[s, i] == True if site s is within ``dcfc_drive_time_min`` of
                       cell i by DRIVING the OSM network AND the cell's traffic
                       is above ``dcfc_min_aadt`` (corridor-only coverage).
        """
        n_sites = len(self.sites)
        n_demand = len(self.demand)
        A_l2 = np.zeros((n_sites, n_demand), dtype=bool)
        A_dcfc = np.zeros((n_sites, n_demand), dtype=bool)
        if n_sites == 0 or n_demand == 0 or self._tree is None:
            return A_l2, A_dcfc

        demand_nodes = self._snap_all(self._demand_pt_array())
        site_nodes = self._snap_all(self._site_pt_array())
        valid_cells = np.array([nd is not None for nd in demand_nodes])
        demand_node_pos = np.array(
            [self._pos[nd] for nd in demand_nodes if nd is not None], dtype=int
        )
        site_node_pos = [self._pos.get(nd) for nd in site_nodes]

        # dedupe sites sharing a snap node: identical coverage, compute once
        site_node_uid = {}
        for si, p in enumerate(site_node_pos):
            if p is not None:
                site_node_uid.setdefault(p, []).append(si)
        unique = list(site_node_uid)

        traffic = (
            self.demand["traffic_count"].values
            if "traffic_count" in self.demand.columns
            else np.zeros(n_demand)
        )
        dcfc_eligible = traffic >= self.dcfc_min_aadt

        if demand_node_pos.size and unique:
            D_walk = self._dist_to_cells(unique, "walk_time", self._walk_cutoff_s,
                                         demand_node_pos)
            D_drive = self._dist_to_cells(unique, "drive_time", self._drive_cutoff_s,
                                          demand_node_pos)
            for k, p in enumerate(unique):
                for si in site_node_uid[p]:
                    A_l2[si, valid_cells] = D_walk[k] <= self._walk_cutoff_s
                    A_dcfc[si, valid_cells] = (
                        (D_drive[k] <= self._drive_cutoff_s) & dcfc_eligible[valid_cells]
                    )
        return A_l2, A_dcfc

    def _site_pt_array(self):
        return np.array(
            [tuple(p.coords[0]) for p in self.site_pts]
        ) if self.site_pts else np.zeros((0, 2))

    def _demand_pt_array(self):
        return np.array(
            [tuple(p.coords[0]) for p in self.demand_pts]
        ) if self.demand_pts else np.zeros((0, 2))
