"""Helpers for querying ArcGIS REST Feature Services.

ArcGIS Feature Servers expose a standard REST + /query endpoint. We wrap the
paged query logic (maxRecordCount up to 2000 per request) so fetch modules for
NCDOT AADT and municipal portals share one code path.
"""
from __future__ import annotations

import geopandas as gpd
import pandas as pd
import requests
from tqdm import tqdm


class ArcGisError(RuntimeError):
    pass


def query_layer(
    layer_url: str,
    where: str = "1=1",
    out_fields: str = "*",
    geometry: dict | None = None,
    max_records: int = 2000,
    timeout: int = 90,
) -> gpd.GeoDataFrame:
    """Query an ArcGIS feature layer and return features as a GeoDataFrame.

    geometry (optional): an ArcGIS geometry JSON dict, e.g. for a spatial
    filter restricted to a polygon/bbox ({"xmin":..,"ymin":..,"xmax":..,
    "ymax":..,"spatialReference":{"wkid":4326}}).
    """
    meta = requests.get(layer_url, params={"f": "json"}, timeout=timeout)
    if meta.status_code != 200:
        raise ArcGisError(f"ArcGIS layer metadata failed with {meta.status_code}")
    meta_json = meta.json()
    if "error" in meta_json:
        raise ArcGisError(f"ArcGIS layer error: {meta_json['error']}")
    cap = meta_json.get("maxRecordCount", max_records)
    page_size = min(cap, max_records)
    geom_type = meta_json.get("geometryType")

    all_features = []
    offset = 0
    with tqdm(desc="querying ArcGIS layer", unit="rec") as pbar:
        while True:
            params = {
                "f": "geojson",
                "where": where,
                "outFields": out_fields,
                "resultOffset": offset,
                "resultRecordCount": page_size,
                "outSR": "4326",
            }
            if geometry is not None:
                params["geometryType"] = "esriGeometryEnvelope"
                params["spatialRel"] = "esriSpatialRelIntersects"
                params["geometry"] = _json_dumps(geometry)
            resp = requests.get(f"{layer_url}/query", params=params, timeout=timeout)
            if resp.status_code != 200:
                raise ArcGisError(
                    f"ArcGIS query failed with {resp.status_code}: {resp.text[:200]}"
                )
            data = resp.json()
            feats = data.get("features", [])
            all_features.extend(feats)
            pbar.update(len(feats))
            if not feats:
                break
            offset += len(feats)
            if len(feats) < page_size:
                break

    return _to_gdf(all_features, geom_type)


def _to_gdf(features: list[dict], geom_type: str | None) -> gpd.GeoDataFrame:
    if not features:
        return gpd.GeoDataFrame({"geometry": gpd.GeoSeries([], crs="EPSG:4326")},
                                crs="EPSG:4326")
    # Use geopandas read_file-style construction from GeoJSON dict
    geojson = {"type": "FeatureCollection", "features": features}
    gdf = gpd.GeoDataFrame.from_features(geojson, crs="EPSG:4326")
    if "OBJECTID" in gdf.columns:
        gdf = gdf.drop(columns=["OBJECTID"])
    if "FID" in gdf.columns:
        gdf = gdf.drop(columns=["FID"])
    return gdf


def _json_dumps(obj: dict) -> str:
    import json

    return json.dumps(obj)