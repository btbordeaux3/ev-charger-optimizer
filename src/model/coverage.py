"""Travel-time coverage matrices and isochrone summaries.

The original project built one hand-made road graph for both walking and
Driving, treated every road as two-way, and used ``fork`` multiprocessing.
V2 instead accepts separate OSMnx walk/drive graphs, respects directionality,
and uses batched SciPy Dijkstra that works identically on Windows and macOS.
An optional cuGraph backend is available on Linux/WSL2 with RAPIDS installed.
"""
from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import geopandas as gpd
import networkx as nx
import numpy as np
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

log = logging.getLogger(__name__)


@dataclass
class _PreparedGraph:
    graph: nx.DiGraph
    node_list: list
    pos: dict
    tree: cKDTree
    csr: object
    node_xy: np.ndarray


def _prepare_graph(G: nx.DiGraph, direction: str) -> _PreparedGraph:
    if direction not in {"to_site", "from_site"}:
        raise ValueError("coverage.direction must be 'to_site' or 'from_site'")
    R = G.reverse(copy=False) if direction == "to_site" else G
    nodes = list(R.nodes)
    if not nodes:
        raise ValueError("Routing graph has no nodes.")
    xy = np.array([[float(R.nodes[n]["x"]), float(R.nodes[n]["y"])] for n in nodes])
    tree = cKDTree(xy)
    pos = {n: i for i, n in enumerate(nodes)}
    csr = nx.to_scipy_sparse_array(
        R,
        nodelist=nodes,
        weight="travel_time",
        format="csr",
        dtype=np.float64,
    )
    return _PreparedGraph(R, nodes, pos, tree, csr, xy)


def _points_xy(gdf: gpd.GeoDataFrame, crs) -> np.ndarray:
    if gdf is None or gdf.empty:
        return np.empty((0, 2), dtype=float)
    p = gdf.to_crs(crs)
    geom = p.geometry
    reps = geom if geom.geom_type.eq("Point").all() else geom.representative_point()
    return np.column_stack([reps.x.to_numpy(), reps.y.to_numpy()]).astype(float)


def _snap_positions(prep: _PreparedGraph, xy: np.ndarray) -> np.ndarray:
    if len(xy) == 0:
        return np.empty(0, dtype=np.int64)
    _dist, idx = prep.tree.query(xy)
    return np.asarray(idx, dtype=np.int64)


def _auto_workers(n: int) -> int:
    if n and n > 0:
        return int(n)
    # Dijkstra chunks can be memory-heavy. Four concurrent C-level calls are a
    # safer default than spawning every logical core.
    return max(1, min(4, (os.cpu_count() or 1)))


def _can_use_cugraph() -> bool:
    try:
        import cudf  # noqa: F401
        import cugraph  # noqa: F401
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


class CoverageMatrix:
    """Compute L2-walk and DCFC-drive travel-time coverage."""

    def __init__(
        self,
        *,
        walk_graph: nx.DiGraph,
        drive_graph: nx.DiGraph,
        sites: gpd.GeoDataFrame,
        demand: gpd.GeoDataFrame,
        metric_crs,
        l2_walk_time_min: float = 10.0,
        dcfc_drive_time_min: float = 10.0,
        dcfc_min_aadt: float = 0.0,
        direction: str = "to_site",
        backend: str = "auto",
        n_workers: int = 0,
        chunk_size: int = 24,
    ):
        self.sites = sites.reset_index(drop=True)
        self.demand = demand.reset_index(drop=True)
        self.metric_crs = metric_crs
        self.l2_walk_time_min = float(l2_walk_time_min)
        self.dcfc_drive_time_min = float(dcfc_drive_time_min)
        self.dcfc_min_aadt = float(dcfc_min_aadt)
        self.direction = direction
        self.n_workers = _auto_workers(n_workers)
        self.chunk_size = max(1, int(chunk_size))

        self.walk = _prepare_graph(walk_graph, direction)
        self.drive = _prepare_graph(drive_graph, direction)

        requested = backend.lower()
        if requested not in {"auto", "scipy", "cugraph"}:
            raise ValueError("routing_backend must be auto, scipy, or cugraph")
        if requested == "cugraph" and not _can_use_cugraph():
            log.warning("cuGraph requested but unavailable; falling back to SciPy.")
            requested = "scipy"
        if requested == "auto":
            requested = "cugraph" if _can_use_cugraph() else "scipy"
        self.backend_used = requested

        site_xy = _points_xy(self.sites, metric_crs)
        demand_xy = _points_xy(self.demand, metric_crs)
        self._site_pos = {
            "l2": _snap_positions(self.walk, site_xy),
            "dcfc": _snap_positions(self.drive, site_xy),
        }
        self._demand_pos = {
            "l2": _snap_positions(self.walk, demand_xy),
            "dcfc": _snap_positions(self.drive, demand_xy),
        }

    def _scipy_times(self, mode: str, site_indices, cutoff_s: float) -> np.ndarray:
        prep = self.walk if mode == "l2" else self.drive
        spos = self._site_pos[mode][np.asarray(site_indices, dtype=int)]
        dpos = self._demand_pos[mode]
        if len(spos) == 0 or len(dpos) == 0:
            return np.empty((len(spos), len(dpos)), dtype=np.float32)

        # Several candidate points can snap to the same graph node. Compute each
        # unique source once, then expand back to candidate order.
        uniq, inverse = np.unique(spos, return_inverse=True)
        chunks = [uniq[i:i + self.chunk_size] for i in range(0, len(uniq), self.chunk_size)]

        def solve(chunk):
            dist = dijkstra(
                prep.csr,
                directed=True,
                indices=np.asarray(chunk, dtype=int),
                limit=float(cutoff_s),
                return_predecessors=False,
            )
            dist = np.atleast_2d(np.asarray(dist, dtype=np.float32))
            return dist[:, dpos]

        if self.n_workers > 1 and len(chunks) > 1:
            with ThreadPoolExecutor(max_workers=self.n_workers) as ex:
                parts = list(ex.map(solve, chunks))
        else:
            parts = [solve(c) for c in chunks]
        uniq_times = np.vstack(parts) if parts else np.empty((0, len(dpos)), dtype=np.float32)
        return uniq_times[inverse]

    def _cugraph_times(self, mode: str, site_indices, cutoff_s: float) -> np.ndarray:
        """Optional NVIDIA path. Requires RAPIDS/cuGraph (Linux or WSL2)."""
        try:
            import cudf
            import cugraph
        except Exception:
            return self._scipy_times(mode, site_indices, cutoff_s)

        prep = self.walk if mode == "l2" else self.drive
        spos = self._site_pos[mode][np.asarray(site_indices, dtype=int)]
        dpos = self._demand_pos[mode]
        coo = prep.csr.tocoo()
        edges = cudf.DataFrame({
            "src": coo.row.astype(np.int32),
            "dst": coo.col.astype(np.int32),
            "weight": coo.data.astype(np.float32),
        })
        cg = cugraph.Graph(directed=True)
        cg.from_cudf_edgelist(
            edges,
            source="src",
            destination="dst",
            edge_attr="weight",
            renumber=False,
        )

        out = np.full((len(spos), len(dpos)), np.inf, dtype=np.float32)
        dpos_arr = np.asarray(dpos, dtype=np.int64)
        for row, source in enumerate(spos):
            result = cugraph.sssp(cg, source=int(source))
            # Current cuGraph returns vertex/distance columns. Convert only the
            # rows we need; thresholding happens after transfer.
            pdf = result[["vertex", "distance"]].to_pandas().set_index("vertex")
            vals = pdf["distance"].reindex(dpos_arr).to_numpy(dtype=np.float32)
            vals[vals > cutoff_s] = np.inf
            out[row] = vals
        return out

    def travel_times(
        self,
        mode: str,
        site_indices=None,
        *,
        cutoff_min: float | None = None,
    ) -> np.ndarray:
        """Return site -> demand (or demand -> site) time matrix in seconds."""
        if mode not in {"l2", "dcfc"}:
            raise ValueError("mode must be l2 or dcfc")
        if site_indices is None:
            site_indices = np.arange(len(self.sites), dtype=int)
        else:
            site_indices = np.asarray(site_indices, dtype=int)
        default = self.l2_walk_time_min if mode == "l2" else self.dcfc_drive_time_min
        cutoff_s = float(cutoff_min if cutoff_min is not None else default) * 60.0
        if self.backend_used == "cugraph":
            return self._cugraph_times(mode, site_indices, cutoff_s)
        return self._scipy_times(mode, site_indices, cutoff_s)

    def build(self) -> tuple[np.ndarray, np.ndarray]:
        n_sites, n_demand = len(self.sites), len(self.demand)
        if n_sites == 0 or n_demand == 0:
            return (
                np.zeros((n_sites, n_demand), dtype=bool),
                np.zeros((n_sites, n_demand), dtype=bool),
            )

        t_l2 = self.travel_times("l2", cutoff_min=self.l2_walk_time_min)
        t_dc = self.travel_times("dcfc", cutoff_min=self.dcfc_drive_time_min)
        A_l2 = np.isfinite(t_l2) & (t_l2 <= self.l2_walk_time_min * 60.0)
        A_dc = np.isfinite(t_dc) & (t_dc <= self.dcfc_drive_time_min * 60.0)

        if self.dcfc_min_aadt > 0 and "traffic_count" in self.demand.columns:
            eligible = (
                self.demand["traffic_count"].to_numpy(dtype=float)
                >= self.dcfc_min_aadt
            )
            A_dc &= eligible[None, :]
        return A_l2, A_dc

    def min_times_for_selected(self, mode: str, site_indices, max_threshold_min: float) -> np.ndarray:
        """Minimum time from each demand cell to any selected site of a mode."""
        site_indices = list(site_indices)
        if not site_indices:
            return np.full(len(self.demand), np.inf, dtype=np.float32)
        times = self.travel_times(mode, site_indices, cutoff_min=max_threshold_min)
        return np.min(times, axis=0) if len(times) else np.full(len(self.demand), np.inf)
