"""Configuration for the plug-and-play EV charger optimizer.

Version 3 keeps the original top-level YAML layout, but fixes several hidden
assumptions in v1:
- metric CRS is chosen per region instead of hard-coding North Carolina's UTM
- routing and candidate-site cache paths are region-specific
- budget-model spread/type constraints are configurable and actually enforced
- isochrone map thresholds and routing backend are configurable
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import yaml


_STATE_ABBR = {
    "37": "NC",
    "01": "AL", "04": "AZ", "05": "AR", "06": "CA", "08": "CO",
    "09": "CT", "10": "DE", "11": "DC", "12": "FL", "13": "GA",
    "15": "HI", "16": "ID", "17": "IL", "18": "IN", "19": "IA",
    "20": "KS", "21": "KY", "22": "LA", "23": "ME", "24": "MD",
    "25": "MA", "26": "MI", "27": "MN", "28": "MS", "29": "MO",
    "30": "MT", "31": "NE", "32": "NV", "33": "NH", "34": "NJ",
    "35": "NM", "36": "NY", "38": "ND", "39": "OH", "40": "OK",
    "41": "OR", "42": "PA", "44": "RI", "45": "SC", "46": "SD",
    "47": "TN", "48": "TX", "49": "UT", "50": "VT", "51": "VA",
    "53": "WA", "54": "WV", "55": "WI", "56": "WY",
}


@dataclass
class County:
    name: str
    state_fips: str
    county_fips: str
    state_abbr: Optional[str] = None

    @property
    def fips(self) -> str:
        return self.state_fips.zfill(2) + self.county_fips.zfill(3)

    def abbr(self) -> str:
        return self.state_abbr or _STATE_ABBR.get(self.state_fips.zfill(2), "US")


@dataclass
class FetcherConfig:
    nrel_api_key: Optional[str] = None
    census_api_key: Optional[str] = None
    municipal_layers: list[dict] = field(default_factory=list)
    road_network_source: str = "osmnx"
    cache_dir: str = "data/raw"

    # Candidate parking filters.
    min_parking_m2: float = 1000.0
    min_parking_capacity: int = 20
    candidate_tile_deg: float = 0.1

    # Routing graph buffers. Demand/candidates are clipped to the exact county
    # polygon; only the network gets a buffer so legitimate paths may leave and
    # re-enter the study boundary.
    walk_network_buffer_m: float = 2500.0
    drive_network_buffer_m: float = 8000.0

    @property
    def has_nrel_key(self) -> bool:
        return bool(self.nrel_api_key)


@dataclass
class DemandConfig:
    grid_resolution_m: int = 400
    alpha_traffic: float = 1.0
    beta_equity: float = 1.0
    income_weighted_equity: bool = False
    # Metric smoothing radius around each grid cell center for AADT stations.
    traffic_influence_m: float = 600.0


@dataclass
class CoverageConfig:
    """Network-time accessibility assumptions.

    Level 2 is modeled as walking access to a charger. DC fast is modeled as
    driving access to a charger. ``direction='to_site'`` means the route is
    demand -> charger, which respects one-way streets correctly.
    """

    l2_walk_time_min: float = 10.0
    dcfc_drive_time_min: float = 10.0
    walk_speed_kph: float = 4.8
    drive_default_speed_kph: float = 35.0
    dcfc_min_aadt: float = 0.0
    direction: str = "to_site"  # to_site | from_site
    map_thresholds_min: list[float] = field(default_factory=lambda: [5.0, 10.0, 15.0])

    # Cross-platform default: SciPy. ``auto`` will use cuGraph only when it is
    # installed and usable (normally Linux/WSL2 + NVIDIA); otherwise SciPy.
    routing_backend: str = "auto"  # auto | scipy | cugraph
    routing_workers: int = 0        # 0=auto; thread workers for SciPy chunks
    routing_chunk_size: int = 24


@dataclass
class OptimizationConfig:
    # Legacy MCLP knobs retained for compatibility.
    k_sites: int = 5
    min_l2: int = 2
    min_dcfc: int = 3
    gamma: float = 0.5

    solver: str = "auto"  # auto | gurobi | cbc
    time_limit_s: int = 120
    mip_gap: float = 0.01

    # V2/V3 budget-model spread controls. V1 ignored min_l2/min_dcfc and gamma in
    # the primary budget model, which made one giant DCFC cluster a common
    # optimum. These controls apply to the primary model.
    min_sites_used: int = 5
    max_sites_used: int = 0         # 0 = no explicit maximum
    min_site_spacing_m: float = 0.0 # optional hard spacing between active sites
    coverage_bonus: float = 0.25    # rewards geographic first-access as well as throughput


@dataclass
class BudgetConfig:
    budget: float = 1_500_000.0
    cost_l2: float = 50_000.0
    cost_dcfc: float = 150_000.0
    cap_l2: float = 15.0
    cap_dcfc: float = 60.0

    # Parking-aware site sizing. ``site_max`` remains a hard absolute fallback.
    site_max: int = 6
    parking_spaces_per_charger: float = 10.0
    parking_m2_per_space: float = 30.0
    min_site_max: int = 1


@dataclass
class ExistingConfig:
    """How current public chargers affect new-site demand.

    A value of 1.0 means existing coverage does not change demand. A value of
    0.35 means cells already accessible to a current public charger retain 35%
    of their original optimization weight, so the model prioritizes gaps while
    still allowing reinforcement where demand is strong.
    """

    covered_demand_multiplier: float = 0.35
    public_only: bool = True


@dataclass
class ZoningConfig:
    layers: list[str] = field(default_factory=list)
    multifamily_column: str = ""
    multifamily_values: list[str] = field(default_factory=list)
    cell_multiplier: float = 1.5


@dataclass
class RegulationsConfig:
    """Config-driven zoning/parking/EV regulatory screening layer.

    This is deliberately a screening layer, not a legal determination. Rules
    can be hard constraints only when a jurisdiction-specific profile marks
    them as such; conditional or ambiguous requirements stay advisory.
    """

    enabled: bool = False
    jurisdiction: str = ""
    as_of_date: str = ""
    mode: str = "advisory"  # reserved for stricter jurisdiction profiles

    # Optional parcel layer used to attach zoning/use context to OSM parking
    # candidates. The default field names match Durham's public parcel layer.
    parcel_layer_url: str = ""
    parcel_id_field: str = "REID"
    address_field: str = "LOCATION_ADDR"
    zoning_field: str = "ZONING"
    land_class_field: str = "LAND_CLASS"
    units_field: str = "TOTAL_UNITS"
    floor_area_field: str = "HEATED_AREA"
    gla_field: str = "GROSS_LEASABLE_AREA"
    request_workers: int = 6

    # A planning/operations guardrail, not a law: never recommend converting
    # more than this share of estimated existing parking stalls unless disabled
    # with 0. The existing parking-derived site_max still also applies.
    max_ev_share_existing_spaces: float = 0.15

    # Unknown/missing GIS context is never silently treated as legal approval.
    unknown_context_policy: str = "allow_with_review"  # allow_with_review | exclude

    # Generic rule schema (dicts loaded from YAML). Supported selectors:
    # zone_regex, land_class_regex. Supported calculations: parking_min_fixed,
    # parking_min_per_unit, parking_min_per_1000_sf, parking_max_factor,
    # ev_min_installed, prohibit_ev_charging, applicability, and manual_review.
    # Parking rates are audited separately from charger-stall counts because an
    # EV charging stall remains a parking stall; explicit prohibitions are the
    # mechanism for true regulatory exclusion.
    rules: list[dict] = field(default_factory=list)


@dataclass
class GridConfig:
    enabled: bool = False
    layers: list[str] = field(default_factory=list)
    capacity_column: str = ""
    charger_power_kw: dict = field(default_factory=lambda: {"l2": 7.2, "dcfc": 150.0})
    simultaneity: float = 1.0
    margin: float = 0.85
    max_iterations: int = 5


@dataclass
class OutputConfig:
    out_dir: str = "data/output"
    map_path: str = "ev_charger_map.html"
    results_path: str = "recommended_sites.csv"
    coverage_path: str = "coverage_summary.csv"
    regulatory_path: str = "regulatory_audit.csv"
    charts_dir: str = "data/output/charts"

    @property
    def map_full(self) -> str:
        return f"{self.out_dir}/{self.map_path}"

    @property
    def results_full(self) -> str:
        return f"{self.out_dir}/{self.results_path}"

    @property
    def coverage_full(self) -> str:
        return f"{self.out_dir}/{self.coverage_path}"

    @property
    def regulatory_full(self) -> str:
        return f"{self.out_dir}/{self.regulatory_path}"


@dataclass
class RegionConfig:
    name: str
    counties: list[County]
    fetcher: FetcherConfig = field(default_factory=FetcherConfig)
    demand: DemandConfig = field(default_factory=DemandConfig)
    coverage: CoverageConfig = field(default_factory=CoverageConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    existing: ExistingConfig = field(default_factory=ExistingConfig)
    zoning: ZoningConfig = field(default_factory=ZoningConfig)
    regulations: RegulationsConfig = field(default_factory=RegulationsConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @property
    def state_fips(self) -> str:
        states = {c.state_fips.zfill(2) for c in self.counties}
        if len(states) != 1:
            raise ValueError("All counties in one run must be in the same state.")
        return next(iter(states))

    @property
    def state_abbr(self) -> str:
        abbrs = {c.abbr() for c in self.counties}
        if len(abbrs) != 1:
            raise ValueError("All counties in one run must be in the same state.")
        return next(iter(abbrs))

    @property
    def county_fips_list(self) -> list[str]:
        return [c.fips for c in self.counties]


def _parse_county(data: dict) -> County:
    state_fips = str(data["state_fips"]).zfill(2)
    return County(
        name=data["name"],
        state_fips=state_fips,
        county_fips=str(data["county_fips"]).zfill(3),
        state_abbr=data.get("state_abbr") or _STATE_ABBR.get(state_fips),
    )


def load_config(path: str) -> RegionConfig:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    fetcher = FetcherConfig(
        nrel_api_key=raw.get("nrel_api_key"),
        census_api_key=raw.get("census_api_key"),
        municipal_layers=raw.get("municipal_layers", []),
        road_network_source=raw.get("road_network_source", "osmnx"),
        cache_dir=raw.get("cache_dir", "data/raw"),
        min_parking_m2=raw.get("min_parking_m2", 1000.0),
        min_parking_capacity=raw.get("min_parking_capacity", 20),
        candidate_tile_deg=raw.get("candidate_tile_deg", 0.1),
        walk_network_buffer_m=raw.get("walk_network_buffer_m", 2500.0),
        drive_network_buffer_m=raw.get("drive_network_buffer_m", 8000.0),
    )

    d = raw.get("demand", {})
    demand = DemandConfig(
        grid_resolution_m=d.get("grid_resolution_m", 400),
        alpha_traffic=d.get("alpha_traffic", 1.0),
        beta_equity=d.get("beta_equity", 1.0),
        income_weighted_equity=d.get("income_weighted_equity", False),
        traffic_influence_m=d.get("traffic_influence_m", 600.0),
    )

    c = raw.get("coverage", {})
    coverage = CoverageConfig(
        l2_walk_time_min=c.get("l2_walk_time_min", 10.0),
        dcfc_drive_time_min=c.get("dcfc_drive_time_min", 10.0),
        walk_speed_kph=c.get("walk_speed_kph", 4.8),
        drive_default_speed_kph=c.get("drive_default_speed_kph", 35.0),
        dcfc_min_aadt=c.get("dcfc_min_aadt", 0.0),
        direction=c.get("direction", "to_site"),
        map_thresholds_min=[float(x) for x in c.get("map_thresholds_min", [5, 10, 15])],
        routing_backend=c.get("routing_backend", "auto"),
        routing_workers=int(c.get("routing_workers", 0)),
        routing_chunk_size=int(c.get("routing_chunk_size", 24)),
    )

    o = raw.get("optimization", {})
    optimization = OptimizationConfig(
        k_sites=int(o.get("k_sites", 5)),
        min_l2=int(o.get("min_l2", 2)),
        min_dcfc=int(o.get("min_dcfc", 3)),
        gamma=float(o.get("gamma", 0.5)),
        solver=o.get("solver", "auto"),
        time_limit_s=int(o.get("time_limit_s", 120)),
        mip_gap=float(o.get("mip_gap", 0.01)),
        min_sites_used=int(o.get("min_sites_used", o.get("k_sites", 5))),
        max_sites_used=int(o.get("max_sites_used", 0)),
        min_site_spacing_m=float(o.get("min_site_spacing_m", 0.0)),
        coverage_bonus=float(o.get("coverage_bonus", 0.25)),
    )

    b = raw.get("budget", {})
    budget = BudgetConfig(
        budget=float(b.get("budget", 1_500_000.0)),
        cost_l2=float(b.get("cost_l2", 50_000.0)),
        cost_dcfc=float(b.get("cost_dcfc", 150_000.0)),
        cap_l2=float(b.get("cap_l2", 15.0)),
        cap_dcfc=float(b.get("cap_dcfc", 60.0)),
        site_max=int(b.get("site_max", 6)),
        parking_spaces_per_charger=float(b.get("parking_spaces_per_charger", 10.0)),
        parking_m2_per_space=float(b.get("parking_m2_per_space", 30.0)),
        min_site_max=int(b.get("min_site_max", 1)),
    )

    e = raw.get("existing", {})
    existing = ExistingConfig(
        covered_demand_multiplier=float(e.get("covered_demand_multiplier", 0.35)),
        public_only=bool(e.get("public_only", True)),
    )

    z = raw.get("zoning", {})
    zoning = ZoningConfig(
        layers=z.get("layers", []),
        multifamily_column=z.get("multifamily_column", ""),
        multifamily_values=z.get("multifamily_values", []),
        cell_multiplier=float(z.get("cell_multiplier", 1.5)),
    )

    r = raw.get("regulations", {})
    regulations = RegulationsConfig(
        enabled=bool(r.get("enabled", False)),
        jurisdiction=str(r.get("jurisdiction", "")),
        as_of_date=str(r.get("as_of_date", "")),
        mode=str(r.get("mode", "advisory")),
        parcel_layer_url=str(r.get("parcel_layer_url", "")),
        parcel_id_field=str(r.get("parcel_id_field", "REID")),
        address_field=str(r.get("address_field", "LOCATION_ADDR")),
        zoning_field=str(r.get("zoning_field", "ZONING")),
        land_class_field=str(r.get("land_class_field", "LAND_CLASS")),
        units_field=str(r.get("units_field", "TOTAL_UNITS")),
        floor_area_field=str(r.get("floor_area_field", "HEATED_AREA")),
        gla_field=str(r.get("gla_field", "GROSS_LEASABLE_AREA")),
        request_workers=int(r.get("request_workers", 6)),
        max_ev_share_existing_spaces=float(r.get("max_ev_share_existing_spaces", 0.15)),
        unknown_context_policy=str(r.get("unknown_context_policy", "allow_with_review")),
        rules=list(r.get("rules", [])),
    )

    g = raw.get("grid", {})
    grid = GridConfig(
        enabled=bool(g.get("enabled", False)),
        layers=g.get("layers", []),
        capacity_column=g.get("capacity_column", ""),
        charger_power_kw=g.get("charger_power_kw", {"l2": 7.2, "dcfc": 150.0}),
        simultaneity=float(g.get("simultaneity", 1.0)),
        margin=float(g.get("margin", 0.85)),
        max_iterations=int(g.get("max_iterations", 5)),
    )

    out = raw.get("output", {})
    out_dir = out.get("out_dir", "data/output")
    output = OutputConfig(
        out_dir=out_dir,
        map_path=out.get("map_path", "ev_charger_map.html"),
        results_path=out.get("results_path", "recommended_sites.csv"),
        coverage_path=out.get("coverage_path", "coverage_summary.csv"),
        regulatory_path=out.get("regulatory_path", "regulatory_audit.csv"),
        charts_dir=out.get("charts_dir", f"{out_dir}/charts"),
    )

    return RegionConfig(
        name=raw["name"],
        counties=[_parse_county(c) for c in raw["counties"]],
        fetcher=fetcher,
        demand=demand,
        coverage=coverage,
        optimization=optimization,
        budget=budget,
        existing=existing,
        zoning=zoning,
        regulations=regulations,
        grid=grid,
        output=output,
    )
