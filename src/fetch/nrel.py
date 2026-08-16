"""Fetch existing EV charging stations from the NREL/NLR Alt-Fuel Stations API.

The NREL (rebranded NLR) API requires a free API key, obtained at
https://developer.nlr.gov/signup/. The key must be present either in the region
config or in the environment as NREL_API_KEY.

This returns a GeoDataFrame of stations with:
  - lat/lon geometry
  - Level 2 and DC fast port counts
  - network operator
  - access type (public / private)
  - status
We use this as the "avoid over-clustering near existing stations" layer.
"""
from __future__ import annotations

import os
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import Point
from tqdm import tqdm

_API_URL = "https://developer.nlr.gov/api/alt-fuel-stations/v1.json"
_CSV_URL = "https://developer.nlr.gov/api/alt-fuel-stations/v1.csv"


class NrelFetchError(RuntimeError):
    pass


def get_api_key(api_key: str | None = None) -> str:
    """Resolve the NREL API key from arg, env, or raise."""
    key = api_key or os.environ.get("NREL_API_KEY", "")
    if not key:
        raise NrelFetchError(
            "No NREL API key found. Add it to your region config or set "
            "NREL_API_KEY in your environment (get one free at "
            "https://developer.nlr.gov/signup/)."
        )
    return key


def fetch_stations(
    state: str = "NC",
    fuel_type: str = "ELEC",
    status: str = "all",
    api_key: str | None = None,
) -> gpd.GeoDataFrame:
    """Fetch EV charging stations for a state and return as GeoDataFrame.

    We pull all stations for the whole state (to avoid the unreliable radius
    filtering upstream) and leave county filtering to the caller, which has the
    actual county geometry to intersect against.
    """
    key = get_api_key(api_key)
    params = {
        "api_key": key,
        "fuel_type": fuel_type,
        "state": state,
        "status": status,
        "limit": "all",
    }
    resp = requests.get(_API_URL, params=params, timeout=60)
    if resp.status_code != 200:
        raise NrelFetchError(
            f"NREL API returned {resp.status_code}: {resp.text[:200]}"
        )
    data = resp.json()
    stations = data.get("fuel_stations", [])

    rows = []
    for s in stations:
        rows.append(
            {
                "station_id": s.get("id"),
                "station_name": s.get("station_name"),
                "latitude": s.get("latitude"),
                "longitude": s.get("longitude"),
                "status_code": s.get("status_code"),
                "access_code": s.get("groups_with_access_code") or s.get("access_code"),
                "owner_type_code": s.get("owner_type_code"),
                "ev_network": s.get("ev_network"),
                "ev_level2_evse_num": _port_count(s, "ev_level2_evse_num"),
                "ev_dc_fast_num": _port_count(s, "ev_dc_fast_num"),
                "ev_level1_evse_num": _port_count(s, "ev_level1_evse_num"),
                "city": s.get("city"),
                "street_address": s.get("street_address"),
                "zip": s.get("zip"),
            }
        )

    if not rows:
        return _empty_gdf()

    df = pd.DataFrame(rows)
    geometry = gpd.points_from_xy(df["longitude"], df["latitude"])
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    return gdf


def save_stations(
    gdf: gpd.GeoDataFrame,
    path: str | Path,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(path, driver="GeoJSON")


def _port_count(station: dict, field: str) -> int:
    v = station.get(field)
    try:
        return int(v) if v is not None else 0
    except (TypeError, ValueError):
        return 0


def _empty_gdf() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "station_name": pd.Series([], dtype="object"),
            "latitude": pd.Series([], dtype="float64"),
            "longitude": pd.Series([], dtype="float64"),
            "ev_level2_evse_num": pd.Series([], dtype="int64"),
            "ev_dc_fast_num": pd.Series([], dtype="int64"),
            "access_code": pd.Series([], dtype="object"),
        },
        geometry=gpd.points_from_xy([], []),
        crs="EPSG:4326",
    )