"""Region configuration - the plug-and-play entry point.

Users define a region (one or more counties / states) in a config file and the
entire pipeline adapts. A region is identified by county FIPS codes, which drive
every data source (NREL filtering, NCDOT spatial query, Census tracts, municipal
sites, and the OSM network we use for road-network distances).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import yaml


@dataclass
class County:
    """A single county that makes up part of the study region."""
    name: str
    state_fips: str
    county_fips: str
    state_abbr: Optional[str] = None

    @property
    def fips(self) -> str:
        """Full 5-digit FIPS code (state + county)."""
        return self.state_fips + self.county_fips

    def abbr(self) -> str:
        return self.state_abbr or _STATE_ABBR.get(self.state_fips, "US")


# FIPS state_code -> USPS abbreviation (for OSM/NREL queries)
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
class FetcherConfig:
    nrel_api_key: Optional[str] = None
    census_api_key: Optional[str] = None
    # Municipal open-data portals (ArcGIS Hub / Socrata / OpenDataSoft feature
    # service URLs). Each entry is a layer describing candidate sites that a
    # government body could host a charger on (parking decks, park-and-rides,
    # libraries, community centers, owned parcels, ...).
    municipal_layers: list[dict] = field(default_factory=list)
    road_network_source: str = "osmnx"  # only "osmnx" supported for now
    cache_dir: str = "data/raw"
    # Parking-focused candidate sites: only keep OSM parking ways at least this
    # large (m^2) or parking nodes declaring at least this many spaces.
    min_parking_m2: float = 1000.0
    min_parking_capacity: int = 20
    # Tile size (degrees) for the parking-site Overpass queries. Smaller tiles
    # = smaller/safer queries but more of them; 0.1 is a good balance.
    candidate_tile_deg: float = 0.1

    @property
    def has_nrel_key(self) -> bool:
        return bool(self.nrel_api_key)


@dataclass
class DemandConfig:
    # Demand grid resolution in meters (edge length of each cell)
    grid_resolution_m: int = 400
    # Weights: alpha scales traffic (AADT) demand, beta scales equity demand.
    alpha_traffic: float = 1.0
    beta_equity: float = 1.0
    # Equity is measured as count of multifamily/renter households. If
    # income_weighted is True, multiply by a low-income factor so poorer
    # tracts get priority.
    income_weighted_equity: bool = False


@dataclass
class CoverageConfig:
    """Coverage isochrones by charger type (travel time, not distance).

    Level 2 (slow) chargers serve local residents, so coverage is a WALKING-
    TIME isochrone: OSM network travel time at ``walk_speed_kph``.
    DC fast chargers serve corridor/commuter traffic, so coverage is a
    DRIVING-TIME isochrone: OSM network travel time using per-road-class
    speeds derived from the OSM ``highway`` tag.
    """
    l2_walk_time_min: float = 20.0       # Level 2 walking isochrone cap (min)
    dcfc_drive_time_min: float = 20.0    # DC fast driving isochrone cap (min)
    walk_speed_kph: float = 4.8          # assumed walking speed (km/h)
    drive_default_speed_kph: float = 35.0  # fallback drive speed (km/h)
    dcfc_min_aadt: float = 0.0           # only allow DCFC coverage of cells above this AADT


@dataclass
class OptimizationConfig:
    k_sites: int = 5
    # Minimum number of each type forced into the solution
    min_l2: int = 0
    min_dcfc: int = 0
    # Soft penalty price p for duplicating coverage (over-clustering control).
    # gamma >= 0. Larger gamma spreads chargers out.
    gamma: float = 0.5
    # Solver preference: "gurobi" (licensed) or "cbc" (open-source fallback)
    # or "auto" (use gurobi if available, else cbc).
    solver: str = "auto"
    time_limit_s: int = 120
    mip_gap: float = 0.01


@dataclass
class BudgetConfig:
    """Capacitated, budgeted charger sizing (the primary model).

    budget: total spend ($) on installed chargers
    cost_l2 / cost_dcfc: installed cost per charger of each class
    cap_l2 / cap_dcfc: demand units one charger of that class can serve
    site_max: max total chargers allowed at any single site (parking cap)
    """
    budget: float = 1_500_000.0
    cost_l2: float = 50_000.0
    cost_dcfc: float = 150_000.0
    cap_l2: float = 15.0
    cap_dcfc: float = 60.0
    site_max: int = 12


@dataclass
class ZoningConfig:
    """Optional zoning layer used to refine where apartment demand really is.

    Census ACS gives multifamily household *counts* per tract/block-group but
    not exact locations. A municipal zoning layer (ArcGIS feature service) tells
    us which parcels are actually designated multifamily. Cells whose centroid
    falls inside a multifamily-designated zone get their census multifamily
    weight multiplied by `cell_multiplier`; cells elsewhere keep the census
    value. Empty `layers` (the default) keeps the pure census behavior.
    """
    layers: list[str] = field(default_factory=list)      # ArcGIS feature-layer URLs
    multifamily_column: str = ""                         # attribute holding the zone class
    multifamily_values: list[str] = field(default_factory=list)  # values = multifamily
    cell_multiplier: float = 1.5                         # boost for cells in MF zones


@dataclass
class GridConfig:
    """Grid-feasibility check that runs AFTER optimization and re-solves.

    The budget MILP optimizes siting ignoring the distribution grid. When
    enabled, each recommended site's aggregate charger load (count * kW per
    charger * simultaneity) is compared against the available hosting capacity
    from a utility feature layer. Sites that exceed `margin` of their capacity
    are cut back (proportionally to the overload) and the MILP is re-solved so
    the freed budget is spent at other sites. Repeats until feasible or
    `max_iterations`.
    """
    enabled: bool = False
    layers: list[str] = field(default_factory=list)      # hosting-capacity feature layers
    capacity_column: str = ""                            # available capacity attribute (kVA)
    charger_power_kw: dict = field(
        default_factory=lambda: {"l2": 7.2, "dcfc": 150.0}
    )
    simultaneity: float = 1.0        # fraction of chargers assumed loaded at peak
    margin: float = 0.85             # max fraction of capacity a site may use
    max_iterations: int = 5


@dataclass
class OutputConfig:
    out_dir: str = "data/output"
    map_path: str = "ev_charger_map.html"
    results_path: str = "recommended_sites.csv"
    charts_dir: str = "charts"

    @property
    def map_full(self) -> str:
        return f"{self.out_dir}/{self.map_path}"

    @property
    def results_full(self) -> str:
        return f"{self.out_dir}/{self.results_path}"


@dataclass
class RegionConfig:
    name: str
    counties: list[County]
    fetcher: FetcherConfig = field(default_factory=FetcherConfig)
    demand: DemandConfig = field(default_factory=DemandConfig)
    coverage: CoverageConfig = field(default_factory=CoverageConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    zoning: ZoningConfig = field(default_factory=ZoningConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @property
    def state_fips(self) -> str:
        states = {c.state_fips for c in self.counties}
        if len(states) != 1:
            raise ValueError("All counties must be in the same state.")
        return next(iter(states))

    @property
    def state_abbr(self) -> str:
        abbrs = {c.state_abbr for c in self.counties}
        if len(abbrs) != 1:
            raise ValueError("All counties must be in the same state.")
        return next(iter(abbrs))

    @property
    def county_fips_list(self) -> list[str]:
        return [c.fips for c in self.counties]


def _parse_county(data: dict) -> County:
    return County(
        name=data["name"],
        state_fips=str(data["state_fips"]),
        county_fips=str(data["county_fips"]).zfill(3),
        state_abbr=data.get("state_abbr"),
    )


def load_config(path: str) -> RegionConfig:
    """Load a region config from a YAML file."""
    with open(path) as f:
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
    )

    demand_raw = raw.get("demand", {})
    demand = DemandConfig(
        grid_resolution_m=demand_raw.get("grid_resolution_m", 400),
        alpha_traffic=demand_raw.get("alpha_traffic", 1.0),
        beta_equity=demand_raw.get("beta_equity", 1.0),
        income_weighted_equity=demand_raw.get("income_weighted_equity", False),
    )

    cov_raw = raw.get("coverage", {})
    coverage = CoverageConfig(
        l2_walk_time_min=cov_raw.get("l2_walk_time_min", 20.0),
        dcfc_drive_time_min=cov_raw.get("dcfc_drive_time_min", 20.0),
        walk_speed_kph=cov_raw.get("walk_speed_kph", 4.8),
        drive_default_speed_kph=cov_raw.get("drive_default_speed_kph", 35.0),
        dcfc_min_aadt=cov_raw.get("dcfc_min_aadt", 0.0),
    )

    opt_raw = raw.get("optimization", {})
    optimization = OptimizationConfig(
        k_sites=opt_raw.get("k_sites", 5),
        min_l2=opt_raw.get("min_l2", 0),
        min_dcfc=opt_raw.get("min_dcfc", 0),
        gamma=opt_raw.get("gamma", 0.5),
        solver=opt_raw.get("solver", "auto"),
        time_limit_s=opt_raw.get("time_limit_s", 120),
        mip_gap=opt_raw.get("mip_gap", 0.01),
    )

    budget_raw = raw.get("budget", {})
    budget = BudgetConfig(
        budget=budget_raw.get("budget", 1_500_000.0),
        cost_l2=budget_raw.get("cost_l2", 50_000.0),
        cost_dcfc=budget_raw.get("cost_dcfc", 150_000.0),
        cap_l2=budget_raw.get("cap_l2", 15.0),
        cap_dcfc=budget_raw.get("cap_dcfc", 60.0),
        site_max=budget_raw.get("site_max", 12),
    )

    zone_raw = raw.get("zoning", {})
    zoning = ZoningConfig(
        layers=zone_raw.get("layers", []),
        multifamily_column=zone_raw.get("multifamily_column", ""),
        multifamily_values=zone_raw.get("multifamily_values", []),
        cell_multiplier=zone_raw.get("cell_multiplier", 1.5),
    )

    grid_raw = raw.get("grid", {})
    grid = GridConfig(
        enabled=grid_raw.get("enabled", False),
        layers=grid_raw.get("layers", []),
        capacity_column=grid_raw.get("capacity_column", ""),
        charger_power_kw=grid_raw.get("charger_power_kw",
                                      {"l2": 7.2, "dcfc": 150.0}),
        simultaneity=grid_raw.get("simultaneity", 1.0),
        margin=grid_raw.get("margin", 0.85),
        max_iterations=grid_raw.get("max_iterations", 5),
    )

    out_raw = raw.get("output", {})
    output = OutputConfig(
        out_dir=out_raw.get("out_dir", "data/output"),
        map_path=out_raw.get("map_path", "ev_charger_map.html"),
        results_path=out_raw.get("results_path", "recommended_sites.csv"),
        charts_dir=out_raw.get("charts_dir", "charts"),
    )

    counties = [_parse_county(c) for c in raw["counties"]]
    return RegionConfig(
        name=raw["name"],
        counties=counties,
        fetcher=fetcher,
        demand=demand,
        coverage=coverage,
        optimization=optimization,
        budget=budget,
        zoning=zoning,
        grid=grid,
        output=output,
    )