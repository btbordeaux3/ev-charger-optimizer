"""Fetch distribution-grid hosting-capacity layers for the grid-feasibility check.

Utilities publish "where can I plug this in" hosting-capacity layers (e.g.
California CPUC's Integration Capacity Analysis maps, or utility transformer /
feeder capacity feature services). Each site's aggregate charger load is later
compared against the capacity taken from the polygon/point that contains (or is
nearest to) the site. See ``src.model.grid_check``.
"""
from __future__ import annotations

import geopandas as gpd
import pandas as pd

from src.fetch.arcgis import query_layer


def fetch_grid_capacity(
    layers: list[str],
    capacity_column: str,
    bbox: tuple[float, float, float, float] | None = None,
) -> gpd.GeoDataFrame | None:
    """Fetch hosting-capacity features; returns None when not configured.

    bbox: (west, south, east, north) spatial filter for the query.

    Returns a GeoDataFrame whose ``capacity_column`` holds numeric available
    capacity (expected in kVA). The configured column must exist on every layer.
    """
    if not layers or not capacity_column:
        return None

    geometry = {
        "xmin": bbox[0], "ymin": bbox[1], "xmax": bbox[2], "ymax": bbox[3],
        "spatialReference": {"wkid": 4326},
    } if bbox else None

    frames = []
    for url in layers:
        try:
            gdf = query_layer(url, geometry=geometry)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                "Grid layer %s failed: %s", url, e
            )
            continue
        if gdf.empty:
            continue
        if capacity_column not in gdf.columns:
            import logging
            logging.getLogger(__name__).warning(
                "Grid layer %s has no column %r; skipping",
                url, capacity_column,
            )
            continue
        gdf[capacity_column] = pd.to_numeric(
            gdf[capacity_column], errors="coerce"
        )
        frames.append(gdf[["geometry", capacity_column]])

    if not frames:
        return None
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs="EPSG:4326")
