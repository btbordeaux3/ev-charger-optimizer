import math

import geopandas as gpd
import networkx as nx
import numpy as np
from shapely.geometry import Point

from src.config import load_config
from src.fetch.osm import add_site_capacity_columns
from src.model.coverage import CoverageMatrix
from src.viz.map import _site_points


def _tiny_graph():
    g = nx.DiGraph()
    g.graph["crs"] = "EPSG:3857"
    g.add_node(0, x=0.0, y=0.0)
    g.add_node(1, x=100.0, y=0.0)
    # Demand at 0 can travel TO a charger at 1; reverse trip is impossible.
    g.add_edge(0, 1, travel_time=10.0)
    return g


def test_directed_to_site_respects_one_way():
    g = _tiny_graph()
    sites = gpd.GeoDataFrame({"name": ["site"]}, geometry=[Point(100, 0)], crs="EPSG:3857")
    demand = gpd.GeoDataFrame({"demand": [1.0]}, geometry=[Point(0, 0)], crs="EPSG:3857")

    cm_to = CoverageMatrix(
        walk_graph=g,
        drive_graph=g,
        sites=sites,
        demand=demand,
        metric_crs="EPSG:3857",
        l2_walk_time_min=1,
        dcfc_drive_time_min=1,
        direction="to_site",
        backend="scipy",
        n_workers=1,
    )
    assert cm_to.travel_times("dcfc", cutoff_min=1)[0, 0] == 10.0

    cm_from = CoverageMatrix(
        walk_graph=g,
        drive_graph=g,
        sites=sites,
        demand=demand,
        metric_crs="EPSG:3857",
        l2_walk_time_min=1,
        dcfc_drive_time_min=1,
        direction="from_site",
        backend="scipy",
        n_workers=1,
    )
    assert math.isinf(float(cm_from.travel_times("dcfc", cutoff_min=1)[0, 0]))


def test_parking_capacity_becomes_site_specific_cap():
    sites = gpd.GeoDataFrame(
        {
            "parking_capacity": [20, 100, 0],
            "parking_area_m2": [0.0, 0.0, 1800.0],
        },
        geometry=[Point(0, 0), Point(1, 0), Point(2, 0)],
        crs="EPSG:4326",
    )
    out = add_site_capacity_columns(
        sites,
        parking_m2_per_space=30,
        parking_spaces_per_charger=10,
        absolute_site_max=6,
        min_site_max=1,
    )
    assert out["estimated_spaces"].tolist() == [20, 100, 60]
    assert out["site_max"].tolist() == [2, 6, 6]


def test_recommended_points_keep_source_metadata():
    sites = gpd.GeoDataFrame(
        {"name": ["A"], "estimated_spaces": [42], "site_max": [4], "osm_id": ["123"]},
        geometry=[Point(-78.9, 36.0)], crs="EPSG:4326",
    )
    rec = _site_points(sites, [(0, "l2", 2), (0, "dcfc", 1)])
    assert rec["site_index"].tolist() == [0, 0]
    assert rec["estimated_spaces"].tolist() == [42, 42]
    assert rec["charger_count"].tolist() == [2, 1]


def test_durham_config_enforces_spread_and_time_layers():
    cfg = load_config("configs/durham_only.yaml")
    assert cfg.optimization.min_sites_used == 5
    assert cfg.optimization.min_l2 == 2
    assert cfg.optimization.min_dcfc == 3
    assert cfg.budget.site_max == 6
    assert cfg.coverage.map_thresholds_min == [5.0, 10.0, 15.0]


def test_regulatory_rules_are_auditable_without_inventing_durham_minimums():
    from types import SimpleNamespace
    from src.fetch.regulations import apply_regulatory_rules

    sites = gpd.GeoDataFrame(
        {
            "estimated_spaces": [40],
            "site_max": [6],
            "parcel_context_matched": [True],
            "zoning_code": ["DD-C"],
            "land_class": ["COMMERCIAL"],
            "parcel_units": [0],
            "parcel_floor_area_sf": [10000],
            "parcel_gla_sf": [9000],
        },
        geometry=[Point(-78.9, 36.0)], crs="EPSG:4326",
    )
    cfg = SimpleNamespace(
        max_ev_share_existing_spaces=0.15,
        unknown_context_policy="allow_with_review",
        rules=[
            {"name": "no parking min", "zone_regex": ".*", "parking_min_fixed": 0},
            {"name": "DD max", "zone_regex": "^DD", "parking_max_factor": 1.0},
            {
                "name": "conditional EV",
                "zone_regex": "^DD",
                "ev_min_installed": 3,
                "applicability": "conditional_height_provision_public_parking",
                "manual_review": True,
            },
        ],
    )
    out = apply_regulatory_rules(sites, cfg)
    assert out.loc[0, "reg_parking_min_spaces"] == 0
    assert out.loc[0, "reg_parking_max_factor"] == 1.0
    assert out.loc[0, "reg_ev_min_installed"] == 3
    assert bool(out.loc[0, "reg_manual_review"])
    # 15% of 40 = 6, so the planning guardrail does not reduce the physical cap.
    assert out.loc[0, "site_max"] == 6


def test_regulatory_planning_share_can_reduce_site_cluster_size():
    from types import SimpleNamespace
    from src.fetch.regulations import apply_regulatory_rules

    sites = gpd.GeoDataFrame(
        {
            "estimated_spaces": [20], "site_max": [6],
            "parcel_context_matched": [True], "zoning_code": ["IL"], "land_class": ["INDUSTRIAL"],
        },
        geometry=[Point(-78.9, 36.0)], crs="EPSG:4326",
    )
    cfg = SimpleNamespace(max_ev_share_existing_spaces=0.15, unknown_context_policy="allow_with_review", rules=[])
    out = apply_regulatory_rules(sites, cfg)
    assert out.loc[0, "planning_ev_share_cap"] == 3
    assert out.loc[0, "site_max"] == 3


def test_unknown_regulatory_context_can_be_excluded_by_profile():
    from types import SimpleNamespace
    from src.fetch.regulations import apply_regulatory_rules

    sites = gpd.GeoDataFrame(
        {"estimated_spaces": [50], "site_max": [5], "parcel_context_matched": [False]},
        geometry=[Point(-78.9, 36.0)], crs="EPSG:4326",
    )
    cfg = SimpleNamespace(max_ev_share_existing_spaces=0.0, unknown_context_policy="exclude", rules=[])
    out = apply_regulatory_rules(sites, cfg)
    assert out.loc[0, "site_max"] == 0
    assert out.loc[0, "regulatory_status"] == "excluded"


def test_durham_config_has_regulatory_profile_and_audit_output():
    cfg = load_config("configs/durham_only.yaml")
    assert cfg.regulations.enabled
    assert "DurhamGISReferenceLayers/MapServer/116" in cfg.regulations.parcel_layer_url
    assert any(r.get("ev_min_installed") == 3 for r in cfg.regulations.rules)
    assert cfg.output.regulatory_full.endswith("regulatory_audit.csv")
