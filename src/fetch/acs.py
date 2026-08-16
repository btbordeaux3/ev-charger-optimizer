"""Fetch American Community Survey (ACS) 5-year tract data for the region.

We join:
  - tract *geometries* from pygris (TIGER shapefiles - no API key required)
  - tract *attributes* from the Census Data API (key required since May 2026)

Attributes used to build the equity/need demand:
  - B01003_001E  total population
  - B19013_001E  median household income
  - B25008_001E  total occupied housing units
  - B25008_003E  renter-occupied housing units
  - B25024_001E  total units in structure
  - B25024_007E  5 to 9 units in structure
  - B25024_008E  10 to 19 units
  - B25024_009E  20 to 49 units
  - B25024_010E  50 or more units
  - B25024_011E  mobile homes

Multifamily count = units in structures with 5+ units. This approximates
households without a garage/home charger, i.e. the equity demand.
"""
from __future__ import annotations

import os
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import Point

import pygris

_YEAR = 2022

# Table B25024 uses columns for 1,2,3-4,5-9,10-19,20-49,50+,mobile,boat,other
_MF_SUFFIXES = ("007", "008", "009", "010")  # 5-9, 10-19, 20-49, 50+

_ATTRIBUTE_COLS = [
    "B01003_001E",  # total population
    "B19013_001E",  # median household income
    "B25008_001E",  # total occupied housing units
    "B25008_003E",  # renter-occupied
    "B25024_001E",  # total units in structure
    "B25024_007E",  # 5-9
    "B25024_008E",  # 10-19
    "B25024_009E",  # 20-49
    "B25024_010E",  # 50+
]


class CensusKeyError(RuntimeError):
    pass


def get_census_key(api_key: str | None = None) -> str:
    key = api_key or os.environ.get("CENSUS_API_KEY", "")
    if not key:
        raise CensusKeyError(
            "No Census API key found. Set CENSUS_API_KEY in your environment "
            "(free key at https://api.census.gov/data/key_signup.html)."
        )
    return key


def fetch_tract_attributes(
    state_fips: str,
    county_fips_list: list[str],
    api_key: str | None = None,
    year: int = _YEAR,
) -> pd.DataFrame:
    """Fetch ACS tract attributes for the given counties from the Census API."""
    key = get_census_key(api_key)
    # county_fips_list holds full 5-digit FIPS (state+county); strip state part
    # and join the 3-digit county codes with commas: "state:37 county:063,135"
    cfips = [f[-3:] for f in county_fips_list]
    counties = ",".join(cfips)
    url = f"https://api.census.gov/data/{year}/acs/acs5"
    params = {
        "get": "NAME," + ",".join(_ATTRIBUTE_COLS),
        "for": "tract:*",
        "in": f"state:{state_fips} county:{counties}",
        "key": key,
    }
    resp = requests.get(url, params=params, timeout=60)
    if resp.status_code != 200:
        raise CensusKeyError(
            f"Census API returned {resp.status_code}: {resp.text[:200]}"
        )
    rows = resp.json()
    header = rows[0]
    data = rows[1:]
    df = pd.DataFrame(data, columns=header)

    # Build derived equity fields
    df["geoid"] = (
        df["state"] + df["county"] + df["tract"]
    ).astype(str)
    df["total_population"] = _to_num(df["B01003_001E"])
    df["median_income"] = _to_num(df["B19013_001E"]).clip(lower=0.0)
    df["renter_units"] = _to_num(df["B25008_003E"])
    df["total_housing_units"] = _to_num(df["B25008_001E"])
    df["multifamily_units"] = df[
        ["B25024_" + s + "E" for s in _MF_SUFFIXES]
    ].apply(
        lambda r: sum(_to_num(v) for v in r), axis=1
    )
    df["no_garage_households"] = df["multifamily_units"]
    return df


def fetch_tract_geometries(
    state_fips: str,
    county_fips_list: list[str],
    year: int = _YEAR,
    cache_dir: str | None = None,
) -> gpd.GeoDataFrame:
    """Fetch tract polygons for the counties via pygris (no key needed).

    county_fips_list may hold 5-digit (state+county) or 3-digit county codes.
    """
    all_gdf = []
    for cfips in county_fips_list:
        cf3 = str(cfips)[-3:]  # pygris wants 3-digit county FIPS
        try:
            gdf = pygris.tracts(
                state=state_fips,
                county=cf3,
                year=year,
                cache=cache_dir is not None,
            )
        except TypeError:
            gdf = pygris.tracts(
                state=state_fips, county=cf3, year=year, cache=True
            )
        all_gdf.append(gdf.to_crs("EPSG:4326"))
    return gpd.GeoDataFrame(
        pd.concat(all_gdf, ignore_index=True), crs="EPSG:4326"
    )


def fetch_tracts(
    state_fips: str,
    county_fips_list: list[str],
    api_key: str | None = None,
    year: int = _YEAR,
) -> gpd.GeoDataFrame:
    """Return a GeoDataFrame of tracts with geometry + equity attributes."""
    attrs = fetch_tract_attributes(state_fips, county_fips_list, api_key, year)
    geom = fetch_tract_geometries(state_fips, county_fips_list, year)
    geom["geoid"] = geom["GEOID"].astype(str)
    merged = geom.merge(attrs, on="geoid", how="left")
    keep = [
        "geometry", "geoid",
        "total_population", "median_income",
        "renter_units", "total_housing_units",
        "multifamily_units", "no_garage_households",
    ]
    merged = merged[[c for c in keep if c in merged.columns]]
    return gpd.GeoDataFrame(merged, geometry="geometry", crs="EPSG:4326")


def _to_num(s):
    """Coerce a scalar or pandas Series to float, NaN -> 0."""
    try:
        if hasattr(s, "astype"):
            v = pd.to_numeric(s, errors="coerce").astype("float64")
            return v.fillna(0.0)
        v = float(s)
        return v if v == v else 0.0  # NaN -> 0
    except (TypeError, ValueError):
        return 0.0