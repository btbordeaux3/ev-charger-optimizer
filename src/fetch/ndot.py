"""Fetch NCDOT Annual Average Daily Traffic (AADT) stations.

The NCDOT AADT layer lives on Esri's public ArcGIS Online org. We query it for
the counties in our region and extract the most recent AADT value per station,
plus road classification. AADT is our spatial *traffic demand* weight.
"""
from __future__ import annotations

import geopandas as gpd
import pandas as pd

from .arcgis import query_layer

# Public NCDOT AADT Stations feature layer (Esri-hosted, no key required)
_AADT_LAYER = (
    "https://services.arcgis.com/NuWFvHYDMVmmxMeM/ArcGIS/rest/"
    "services/NCDOT_AADTT_Traffic_Segmentation/FeatureServer/0"
)



def fetch_aadt(
    counties: list[str] | None = None,
    layer_url: str = _AADT_LAYER,
) -> gpd.GeoDataFrame:
    """Fetch NCDOT AADT stations, optionally restricted to county names/FIPS.

    counties: list of county names (e.g. "Durham") or 5-digit FIPS codes.
    The layer's COUNTY field stores uppercase names, so we match either.
    """
    where = "1=1"
    gdf = query_layer(layer_url, where=where, out_fields="*")

    if gdf.empty:
        return gdf

    # Extract latest AADT + road class; build a clean frame.
    df = gdf.copy()
    aadt, aadt_year = _latest_aadt_series(df)
    df["aadt"] = aadt.values
    df["aadt_year"] = aadt_year.values

    # Normalise standard columns (avoid duplicates from renames).
    renames = {
        "LOCATION": "location",
        "COUNTY": "county",
        "ROUTE": "route",
        "RTE_CLS": "route_class",
        "LocationID": "location_id",
    }
    df = df.rename(columns=renames)
    out = gpd.GeoDataFrame(df[["geometry"]], geometry="geometry", crs="EPSG:4326")
    for col in ["aadt", "aadt_year", "route_class", "route", "county",
                "location", "location_id", "name"]:
        if col in df.columns:
            out[col] = df[col]

    if counties is not None and "county" in out.columns:
        wanted = {str(c).upper() for c in counties}
        mask = out["county"].astype(str).str.upper().isin(wanted)
        out = out[mask].copy()

    # Drop stations with no usable traffic value (AADT == 0 means no data).
    out = out[out["aadt"] > 0].copy()
    out = out.dropna(subset=["geometry"]).copy()
    return out


def _latest_aadt_series(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Return newest positive AADT and its year for every station.

    V1 accidentally excluded ``AADT_2022`` while still labelling every station
    as year 2022. This detects every ``AADT_YYYY`` field dynamically.
    """
    import re

    cols = []
    for c in df.columns:
        m = re.fullmatch(r"AADT_(\d{4})", str(c))
        if m:
            cols.append((int(m.group(1)), c))
    cols.sort(reverse=True)

    out = pd.Series(0.0, index=df.index, dtype="float64")
    yr = pd.Series(pd.NA, index=df.index, dtype="Int64")
    for year, col in cols:
        val = pd.to_numeric(df[col], errors="coerce")
        mask = (out <= 0) & val.notna() & (val > 0)
        out.loc[mask] = val.loc[mask].astype(float)
        yr.loc[mask] = year
    return out, yr
