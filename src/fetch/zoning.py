"""Fetch municipal zoning polygons (multifamily-designated zones).

Zoning is a *permission* layer, not an occupancy layer: a zone may allow
apartments without any apartments existing. We therefore never use zoning on
its own for demand — it only refines the census multifamily counts by
upweighting cells that sit inside multifamily-designated zones (see
``src.model.demand.assign_zoning``).

Layers are ArcGIS feature services (same code path as the municipal-sites
fetcher). Each layer is tagged with an attribute (``multifamily_column``) whose
values (``multifamily_values``) mark multifamily designations. If a layer does
not expose the configured column it is skipped with a warning.
"""
from __future__ import annotations

import geopandas as gpd
import pandas as pd

from src.fetch.arcgis import query_layer


def _bbox_geometry(west, south, east, north) -> dict:
    return {
        "xmin": west, "ymin": south, "xmax": east, "ymax": north,
        "spatialReference": {"wkid": 4326},
    }


def fetch_zoning(
    layers: list[str],
    multifamily_column: str,
    multifamily_values: list[str],
    bbox: tuple[float, float, float, float] | None = None,
) -> gpd.GeoDataFrame:
    """Fetch zoning polygons and flag multifamily-designated zones.

    bbox: (west, south, east, north) used as a spatial filter on the query.

    Returns a polygon GeoDataFrame with a boolean ``is_multifamily`` column.
    Returns an empty GeoDataFrame when nothing is configured or nothing matches.
    """
    empty = gpd.GeoDataFrame(
        {"geometry": gpd.GeoSeries([], crs="EPSG:4326")}, crs="EPSG:4326"
    )
    if not layers or not multifamily_column or not multifamily_values:
        return empty

    geometry = _bbox_geometry(*bbox) if bbox else None
    frames = []
    for url in layers:
        try:
            gdf = query_layer(url, geometry=geometry)
        except Exception as e:  # one bad layer shouldn't kill the pipeline
            import logging
            logging.getLogger(__name__).warning(
                "Zoning layer %s failed: %s", url, e
            )
            continue
        if gdf.empty:
            continue
        if multifamily_column not in gdf.columns:
            import logging
            logging.getLogger(__name__).warning(
                "Zoning layer %s has no column %r; skipping",
                url, multifamily_column,
            )
            continue
        gdf["is_multifamily"] = gdf[multifamily_column].astype(str).isin(
            multifamily_values
        )
        gdf = gdf[gdf["is_multifamily"]].copy()
        frames.append(gdf[["geometry", "is_multifamily"]])

    if not frames:
        return empty
    out = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs="EPSG:4326")
    return out[out["is_multifamily"]].copy()
