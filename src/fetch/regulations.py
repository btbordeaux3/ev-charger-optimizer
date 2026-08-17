"""Jurisdiction-specific zoning / parking / EV regulatory screening.

The optimizer is a planning tool, not a permit engine. This module therefore
separates three things explicitly:

1. GIS facts (parcel, zoning, land class), fetched from an authoritative layer;
2. codified rules, represented in config so they are reviewable/versioned; and
3. planning guardrails, such as limiting how much of an existing lot is
   converted to charging at once.

Only an explicit ``prohibit_ev_charging`` rule or ``unknown_context_policy:
exclude`` can make a candidate ineligible by default. Parking-rate and EV-space
requirements are written to the audit table and remain advisory unless a local
profile deliberately marks them otherwise.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests

log = logging.getLogger(__name__)


def _clean(value, default=""):
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    return value


def _number(value, default=0.0) -> float:
    try:
        v = float(value)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def _candidate_signature(sites: gpd.GeoDataFrame, layer_url: str) -> str:
    g = sites.to_crs("EPSG:4326")
    payload = [layer_url, str(len(g))]
    for p in g.geometry:
        payload.append(f"{p.x:.6f},{p.y:.6f}")
    return hashlib.sha1("|".join(payload).encode()).hexdigest()[:12]


def _cache_path(cache_dir: str, namespace: str, sites, layer_url: str) -> Path:
    root = Path(cache_dir) / namespace
    root.mkdir(parents=True, exist_ok=True)
    sig = _candidate_signature(sites, layer_url)
    return root / f"regulatory_site_context_{sig}.geojson"


def _query_point(layer_url: str, lon: float, lat: float, out_fields: list[str], timeout: int = 45) -> dict:
    """Return the first ArcGIS polygon feature containing a WGS84 point."""
    params = {
        "f": "json",
        "where": "1=1",
        "outFields": ",".join(dict.fromkeys([f for f in out_fields if f])),
        "returnGeometry": "false",
        "geometry": f"{lon:.8f},{lat:.8f}",
        "geometryType": "esriGeometryPoint",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
    }
    r = requests.get(f"{layer_url.rstrip('/')}/query", params=params, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    if "error" in data:
        raise RuntimeError(str(data["error"]))
    feats = data.get("features", [])
    if not feats:
        return {}
    return feats[0].get("attributes", {}) or {}



def _query_multipoint(layer_url: str, points: list[tuple[float, float]], out_fields: list[str], timeout: int = 90) -> gpd.GeoDataFrame:
    """Fetch parcel polygons intersecting a batch of WGS84 candidate points."""
    if not points:
        return gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs="EPSG:4326"), crs="EPSG:4326")
    params = {
        "f": "geojson",
        "where": "1=1",
        "outFields": ",".join(dict.fromkeys([f for f in out_fields if f])),
        "returnGeometry": "true",
        "geometry": json.dumps({"points": [[x, y] for x, y in points], "spatialReference": {"wkid": 4326}}),
        "geometryType": "esriGeometryMultipoint",
        "inSR": "4326",
        "outSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "resultRecordCount": 2000,
    }
    r = requests.get(f"{layer_url.rstrip('/')}/query", params=params, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    if "error" in data:
        raise RuntimeError(str(data["error"]))
    feats = data.get("features", [])
    if not feats:
        return gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs="EPSG:4326"), crs="EPSG:4326")
    return gpd.GeoDataFrame.from_features({"type": "FeatureCollection", "features": feats}, crs="EPSG:4326")

def attach_parcel_context(
    sites: gpd.GeoDataFrame,
    regulations_cfg,
    *,
    cache_dir: str,
    namespace: str,
    force_refresh: bool = False,
) -> gpd.GeoDataFrame:
    """Attach parcel/zoning attributes to each OSM candidate point.

    The function uses one lightweight point-in-polygon query per candidate and
    caches the result. That avoids downloading an entire county parcel fabric.
    Failure to match a parcel is preserved as an explicit review state instead
    of silently inventing a zoning classification.
    """
    sites = sites.copy()
    if sites.empty or not regulations_cfg.enabled or not regulations_cfg.parcel_layer_url:
        return _ensure_context_columns(sites)

    path = _cache_path(cache_dir, namespace, sites, regulations_cfg.parcel_layer_url)
    if path.exists() and not force_refresh:
        try:
            cached = gpd.read_file(path)
            if len(cached) == len(sites):
                return cached
        except Exception as e:
            log.warning("Could not read regulatory context cache %s: %s", path, e)

    fields = {
        "parcel_id": regulations_cfg.parcel_id_field,
        "parcel_address": regulations_cfg.address_field,
        "zoning_code": regulations_cfg.zoning_field,
        "land_class": regulations_cfg.land_class_field,
        "parcel_units": regulations_cfg.units_field,
        "parcel_floor_area_sf": regulations_cfg.floor_area_field,
        "parcel_gla_sf": regulations_cfg.gla_field,
    }
    out_fields = list(fields.values())
    pts = sites.to_crs("EPSG:4326")
    attrs: list[dict] = [{} for _ in range(len(pts))]

    # First try batched ArcGIS multipoint queries. A batch typically replaces
    # hundreds of individual requests, which is much faster and friendlier to a
    # municipal GIS server. Only unmatched/failed candidates fall back to
    # concurrent point queries.
    batch_size = 250
    batch_failed = False
    try:
        for start in range(0, len(pts), batch_size):
            stop = min(start + batch_size, len(pts))
            idxs = list(range(start, stop))
            xy = [(float(pts.geometry.iloc[i].x), float(pts.geometry.iloc[i].y)) for i in idxs]
            parcels = _query_multipoint(regulations_cfg.parcel_layer_url, xy, out_fields)
            if parcels.empty:
                continue
            chunk_points = pts.iloc[idxs][["geometry"]].copy()
            chunk_points["_candidate_pos"] = idxs
            joined = gpd.sjoin(chunk_points, parcels, how="left", predicate="intersects")
            for _, jr in joined.iterrows():
                pos = int(jr["_candidate_pos"])
                if attrs[pos]:
                    continue
                candidate = {}
                for source in out_fields:
                    if source in jr and pd.notna(jr[source]):
                        candidate[source] = jr[source]
                if candidate:
                    attrs[pos] = candidate
    except Exception as e:
        batch_failed = True
        log.warning("Batch parcel lookup failed; falling back to point queries: %s", e)

    missing = [i for i, a in enumerate(attrs) if not a]
    if missing:
        def one(i: int):
            p = pts.geometry.iloc[i]
            return i, _query_point(
                regulations_cfg.parcel_layer_url,
                float(p.x), float(p.y), out_fields,
            )

        workers = max(1, int(regulations_cfg.request_workers or 1))
        failures = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(one, i) for i in missing]
            for fut in as_completed(futures):
                try:
                    i, row = fut.result()
                    attrs[i] = row
                except Exception as e:
                    failures += 1
                    if failures <= 5:
                        log.warning("Parcel lookup failed: %s", e)
        if failures:
            log.warning("Parcel lookup failures: %d of %d fallback candidate sites", failures, len(missing))
    if batch_failed and not any(attrs):
        log.warning("No parcel contexts could be matched; candidates will be marked for review.")

    for standard, source in fields.items():
        sites[standard] = [_clean(a.get(source, "")) for a in attrs]
    sites["parcel_context_matched"] = [bool(a) for a in attrs]

    # Normalize numeric fields so downstream YAML rules are predictable.
    for c in ["parcel_units", "parcel_floor_area_sf", "parcel_gla_sf"]:
        sites[c] = pd.to_numeric(sites[c], errors="coerce").fillna(0.0)

    try:
        sites.to_file(path, driver="GeoJSON")
    except Exception as e:
        log.warning("Could not write regulatory context cache %s: %s", path, e)
    return sites


def _ensure_context_columns(sites: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    sites = sites.copy()
    defaults = {
        "parcel_id": "", "parcel_address": "", "zoning_code": "",
        "land_class": "", "parcel_units": 0.0, "parcel_floor_area_sf": 0.0,
        "parcel_gla_sf": 0.0, "parcel_context_matched": False,
    }
    for c, default in defaults.items():
        if c not in sites.columns:
            sites[c] = default
    return sites


def _matches(rule: dict, row: pd.Series) -> bool:
    zone = str(_clean(row.get("zoning_code", ""), ""))
    land = str(_clean(row.get("land_class", ""), ""))
    zr = str(rule.get("zone_regex", ".*") or ".*")
    lr = str(rule.get("land_class_regex", ".*") or ".*")
    try:
        return bool(re.search(zr, zone, flags=re.I)) and bool(re.search(lr, land, flags=re.I))
    except re.error as e:
        raise ValueError(f"Bad regulatory regex in rule {rule.get('name', '<unnamed>')}: {e}") from e


def _rule_parking_min(rule: dict, row: pd.Series) -> float | None:
    """Calculate one rule's parking minimum, if it specifies one.

    If multiple bases are provided within a single rule, use the largest. This
    supports common ordinance forms such as "4 spaces or X per unit, whichever
    is greater" without assuming they are additive.
    """
    vals = []
    if rule.get("parking_min_fixed") is not None:
        vals.append(_number(rule.get("parking_min_fixed")))
    if rule.get("parking_min_per_unit") is not None:
        vals.append(_number(rule.get("parking_min_per_unit")) * _number(row.get("parcel_units", 0)))
    if rule.get("parking_min_per_1000_sf") is not None:
        sf = max(_number(row.get("parcel_gla_sf", 0)), _number(row.get("parcel_floor_area_sf", 0)))
        vals.append(_number(rule.get("parking_min_per_1000_sf")) * sf / 1000.0)
    return max(vals) if vals else None


def apply_regulatory_rules(sites: gpd.GeoDataFrame, regulations_cfg) -> gpd.GeoDataFrame:
    """Apply config rules and derive an optimizer-safe candidate capacity.

    ``site_max`` remains the physical/parking-size cap from the core model.
    V3 creates ``site_max_pre_regulation`` and then may lower ``site_max`` for:
    - an explicit prohibition;
    - unknown context when the profile says to exclude; or
    - the configurable *planning* maximum EV share of an existing parking lot.

    Ordinary motor-vehicle parking minima are audited but do not automatically
    consume EV stalls: an EV charging stall is still a parking stall. This avoids
    the common modeling mistake of treating every charger as deletion of a
    parking space.
    """
    sites = _ensure_context_columns(sites)
    if sites.empty:
        return sites

    sites = sites.copy()
    base = pd.to_numeric(sites.get("site_max", 0), errors="coerce").fillna(0).astype(int)
    est = pd.to_numeric(sites.get("estimated_spaces", 0), errors="coerce").fillna(0).astype(int)
    sites["site_max_pre_regulation"] = base

    # Planning guardrail: at most X% of existing stalls get chargers in this
    # phase. This is explicitly not labeled as a zoning law.
    share = max(0.0, float(regulations_cfg.max_ev_share_existing_spaces or 0.0))
    if share > 0:
        share_cap = np.floor(est.to_numpy(float) * share).astype(int)
        share_cap = np.where((est.to_numpy() > 0) & (share_cap < 1), 1, share_cap)
        share_cap = np.where(est.to_numpy() > 0, share_cap, base.to_numpy())
    else:
        share_cap = base.to_numpy()
    sites["planning_ev_share_cap"] = share_cap

    out_site_max = np.minimum(base.to_numpy(), share_cap).astype(int)
    parking_min = np.full(len(sites), np.nan)
    parking_max_factor = np.full(len(sites), np.nan)
    ev_min = np.zeros(len(sites), dtype=int)
    hard_prohibited = np.zeros(len(sites), dtype=bool)
    review = np.zeros(len(sites), dtype=bool)
    rule_names: list[str] = []
    notes: list[str] = []

    for pos, (_, row) in enumerate(sites.iterrows()):
        matched_names = []
        row_notes = []
        row_min_values = []
        row_max_values = []
        row_ev_values = []
        prohibited = False
        needs_review = not bool(row.get("parcel_context_matched", False))

        for rule in regulations_cfg.rules:
            if not _matches(rule, row):
                continue
            name = str(rule.get("name", "unnamed rule"))
            matched_names.append(name)
            pmin = _rule_parking_min(rule, row)
            if pmin is not None:
                row_min_values.append(pmin)
            if rule.get("parking_max_factor") is not None:
                row_max_values.append(_number(rule.get("parking_max_factor")))
            if rule.get("ev_min_installed") is not None:
                row_ev_values.append(int(max(0, round(_number(rule.get("ev_min_installed"))))))
            if bool(rule.get("prohibit_ev_charging", False)):
                prohibited = True
            if bool(rule.get("manual_review", False)) or str(rule.get("applicability", "general")) != "general":
                needs_review = True
            note = str(rule.get("note", "") or "").strip()
            if note:
                row_notes.append(note)

        if row_min_values:
            parking_min[pos] = max(row_min_values)
        if row_max_values:
            # If multiple location rules accidentally overlap, retain the most
            # restrictive maximum factor and flag for review.
            parking_max_factor[pos] = min(row_max_values)
            if len(set(row_max_values)) > 1:
                needs_review = True
        if row_ev_values:
            ev_min[pos] = max(row_ev_values)

        if prohibited:
            out_site_max[pos] = 0
        if not bool(row.get("parcel_context_matched", False)) and regulations_cfg.unknown_context_policy == "exclude":
            out_site_max[pos] = 0
            prohibited = True
            row_notes.append("Excluded because parcel/zoning context could not be confirmed.")

        hard_prohibited[pos] = prohibited
        review[pos] = needs_review
        rule_names.append("; ".join(matched_names))
        notes.append("; ".join(dict.fromkeys(row_notes)))

    sites["reg_parking_min_spaces"] = parking_min
    sites["reg_parking_max_factor"] = parking_max_factor
    sites["reg_ev_min_installed"] = ev_min
    sites["reg_rule_names"] = rule_names
    sites["reg_manual_review"] = review
    sites["reg_hard_prohibited"] = hard_prohibited
    sites["reg_notes"] = notes
    sites["site_max"] = out_site_max
    sites["regulatory_status"] = np.select(
        [hard_prohibited, review, sites["parcel_context_matched"].to_numpy(bool)],
        ["excluded", "manual_review", "context_matched"],
        default="no_context",
    )
    return sites


def regulatory_audit_frame(sites: gpd.GeoDataFrame) -> pd.DataFrame:
    """Compact, human-readable audit table for all candidate sites."""
    cols = [
        "name", "osm_type", "osm_id", "estimated_spaces",
        "site_max_pre_regulation", "planning_ev_share_cap", "site_max",
        "parcel_id", "parcel_address", "zoning_code", "land_class",
        "parcel_units", "parcel_floor_area_sf", "parcel_gla_sf",
        "reg_parking_min_spaces", "reg_parking_max_factor",
        "reg_ev_min_installed", "regulatory_status", "reg_manual_review",
        "reg_hard_prohibited", "reg_rule_names", "reg_notes",
    ]
    out = sites[[c for c in cols if c in sites.columns]].copy()
    pts = sites.to_crs("EPSG:4326").geometry
    out.insert(0, "site_index", np.arange(len(sites), dtype=int))
    out["lon"] = pts.x.to_numpy()
    out["lat"] = pts.y.to_numpy()
    return out
