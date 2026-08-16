"""Build the coverage matrix between candidate sites and demand cells.

For each (site, demand cell, charger type) we determine whether the road-network
distance is within the type's radius. To keep this fast:

  1. We snap every site and demand centroid to graph nodes once.
  2. For each site we run a single-source Dijkstra with a distance cutoff equal
     to the largest radius (DCFC). This gives us all reachable nodes in one pass.
  3. A site covers a demand cell if that cell's nearest node is in the reachable
     set AND the road distance is within the type radius.

Only demand cells whose *euclidean* distance is within the max radius are even
candidates (spatial pre-filter), which is the big speedup.
"""
from __future__ import annotations

import numpy as np
import geopandas as gpd
import networkx as nx
from scipy.spatial import cKDTree

from src.fetch.network import build_graph, snap_index, project_points_to


class CoverageMatrix:
    """Road-network coverage between sites and demand cells per charger type."""

    def __init__(
        self,
        G: nx.DiGraph,
        sites: gpd.GeoDataFrame,
        demand: gpd.GeoDataFrame,
        radius_l2_m: float,
        radius_dcfc_m: float,
        dcfc_min_aadt: float = 0.0,
        utm_epsg: int = 32617,
    ):
        self.G = G
        self.sites = sites
        self.demand = demand
        self.radius_l2_m = radius_l2_m
        self.radius_dcfc_m = radius_dcfc_m
        self.dcfc_min_aadt = dcfc_min_aadt
        self.utm_epsg = utm_epsg
        self._max_radius = max(radius_l2_m, radius_dcfc_m)

        # Project site/demand points once to the graph CRS
        self.site_pts = project_points_to(sites, utm_epsg)
        self.demand_pts = project_points_to(demand, utm_epsg)
        self._tree, self._node_list = snap_index(G)

        # Node index for demand cells (for quick lookups)
        self._demand_pt_array = np.array(
            [tuple(p.coords[0]) for p in self.demand_pts]
        ) if self.demand_pts else np.zeros((0, 2))
        self._site_pt_array = np.array(
            [tuple(p.coords[0]) for p in self.site_pts]
        ) if self.site_pts else np.zeros((0, 2))

    def _snap_all(self, pts_array):
        """Snap an array of (x,y) coords to node ids via the KD tree."""
        if self._tree is None or len(pts_array) == 0:
            return [None] * len(pts_array)
        dist, idx = self._tree.query(pts_array)
        return [self._node_list[int(i)] for i in idx]

    def build(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (A_l2, A_dcfc) boolean coverage matrices.

        A_l2[s, i] == True if site s covers demand cell i within radius_l2_m
        A_dcfc[s, i] == True if site s covers cell i within radius_dcfc_m AND
        the cell's traffic is above dcfc_min_aadt (corridor-only coverage).
        """
        n_sites = len(self.sites)
        n_demand = len(self.demand)
        A_l2 = np.zeros((n_sites, n_demand), dtype=bool)
        A_dcfc = np.zeros((n_sites, n_demand), dtype=bool)
        if not self.G or n_sites == 0 or n_demand == 0:
            return A_l2, A_dcfc

        demand_nodes = self._snap_all(self._demand_pt_array)
        site_nodes = self._snap_all(self._site_pt_array)
        # demand node -> list of cell indices (a demand node can have multiple)
        node_to_cells: dict = {}
        for ci, nd in enumerate(demand_nodes):
            node_to_cells.setdefault(nd, []).append(ci)

        # Pre-compute euclidean distances site->demand for pre-filter
        eucl = np.linalg.norm(
            self._site_pt_array[:, None, :] - self._demand_pt_array[None, :, :],
            axis=-1,
        )
        # traffic threshold for DCFC
        traffic = (
            self.demand["traffic_count"].values
            if "traffic_count" in self.demand.columns
            else np.zeros(n_demand)
        )
        dcfc_eligible = traffic >= self.dcfc_min_aadt

        for si in range(n_sites):
            src = site_nodes[si]
            if src is None:
                continue
            # single-source dijkstra with cutoff = max radius
            try:
                lengths = nx.single_source_dijkstra_path_length(
                    self.G, src, cutoff=self._max_radius, weight="length"
                )
            except nx.NetworkXError:
                continue
            for ci in np.where(eucl[si] <= self._max_radius)[0]:
                nd = demand_nodes[ci]
                if nd is None:
                    continue
                d = lengths.get(nd, float("inf"))
                if d <= self.radius_l2_m:
                    A_l2[si, ci] = True
                if d <= self.radius_dcfc_m and dcfc_eligible[ci]:
                    A_dcfc[si, ci] = True
        return A_l2, A_dcfc