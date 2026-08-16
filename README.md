# EV Charger Placement Optimization

Two-model optimizer for siting public EV chargers with equity-aware demand
weighting, built for Durham/Orange County, NC but designed to be
**plug-and-play**: point it at any region's county FIPS codes in a config file
and it fetches the data, builds the road-network coverage matrix, and solves
the MILP — no manual data prep.

- **Budget model (primary)** — spend a fixed budget on any mix of chargers.
  Sites can host *clusters* sized to local demand; charger throughput
  (capacity) caps how much nearby demand each charger actually serves; only
  parking-rich sites are eligible.
- **MCLP (legacy)** — classic exactly-*k*-site Maximal Coverage model with a
  `gamma` spread penalty, kept for backward compatibility.

## What it does

1. **Fetches live data** for the region (all cached under `data/raw/`):
   - NCDOT Annual Average Daily Traffic (AADT) stations → traffic demand
   - Census ACS tract data → equity demand (multifamily/renter households)
   - OpenStreetMap road network → road-network (driving) coverage distances
   - OpenStreetMap **parking areas** (surface lots, multi-storey decks,
     garages, park-and-rides) → candidate host sites, filtered by size so
     chargers only land where there is real parking
   - NREL existing chargers → context / over-clustering layer
2. **Builds a weighted demand grid** over the region:
   `demand = alpha·norm(AADT) + beta·norm(multifamily renter units)`
3. **Computes road-network coverage matrices** (single-source Dijkstra with a
   Euclidean pre-filter) for two charger types:
   - Level 2 (radius 1 mi) — serves local residents
   - DC fast (radius 4 mi) — serves corridor/commuter traffic
4. **Solves the capacitated budget model** with Gurobi (falls back to CBC/PuLP):
   - maximize demand units served, subject to a total budget
   - charger throughput caps (L2 = 15 units, DCFC = 60 units)
   - per-site parking cap (`site_max` chargers) so clusters stay feasible
   - warm-started from the greedy baseline so the optimizer is never worse
5. **Emits deliverables**:
   - interactive `folium` map (`ev_charger_map.html`) with per-site charger
     cluster counts
   - ranked site CSV (`recommended_sites.csv`) with `charger_count`
   - before/after chart: optimized plan vs. a greedy "build where traffic is"
     baseline
   - charger-mix/spend chart + equity income-profile chart

## Quickstart

```bash
# 1. Create venv + install deps
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Add API keys (optional but recommended)
cp .env.example .env
#    NREL_API_KEY=<get one free at https://developer.nlr.gov/signup/>
#    CENSUS_API_KEY=<get one free at https://api.census.gov/data/key_signup.html>
#    (the Census email includes an activation link you MUST click)

# 3. Run the pipeline for a region
python -m src.pipeline configs/durham_orange.yaml
#    or override any optimization setting:
python -m src.pipeline configs/durham_orange.yaml --budget 2500000 --solver gurobi
```

Outputs land in `data/output/` (map HTML, CSV) and `data/output/charts/` (PNGs).

## Local dashboard website

Each run also builds a self-contained `data/output/index.html` dashboard that
embeds the interactive map, the comparison/sensitivity/equity charts, a stat
panel, and the recommended-sites table. Host it locally with the bundled
stdlib-only server:

```bash
python -m src.web.serve 8000 data/output
# open  http://localhost:8000/index.html
```

That's it — no Flask, no extra dependencies. Every `python -m src.pipeline ...`
run regenerates the dashboard automatically.

## Plug-and-play: a new region

Copy a config and change only the county list + keys:

```yaml
name: "Your County"
counties:
  - name: "Wake"
    state_fips: 37
    county_fips: 183
    state_abbr: NC
```

That's it — county FIPS drive every data source (NCDOT filter, Census tracts,
OSM network bbox). Municipal open-data portals can be added under
`municipal_layers` to supply government-owned candidate sites; when empty, the
pipeline falls back to OSM parking areas (works for any region).

## Configuration reference

See `configs/durham_orange.yaml` for the full set of knobs:

| Section | Key | Default | Meaning |
|---|---|---|---|
| `demand` | `grid_resolution_m` | 400 | demand cell edge length (m) |
| `demand` | `alpha_traffic` / `beta_equity` | 1.0 / 1.0 | demand weighting |
| `demand` | `income_weighted_equity` | false | upweight low-income tracts |
| `coverage` | `radius_l2_m` | 1609.34 | Level-2 radius (~1 mi) |
| `coverage` | `radius_dcfc_m` | 6437.38 | DC-fast radius (~4 mi) |
| `budget` | `budget` | 1500000 | total spend ($) |
| `budget` | `cost_l2` / `cost_dcfc` | 50000 / 150000 | installed cost per charger |
| `budget` | `cap_l2` / `cap_dcfc` | 15 / 60 | demand units one charger serves |
| `budget` | `site_max` | 12 | max chargers at one site (parking cap) |
| `optimization` | `solver` | auto | `auto` \| `gurobi` \| `cbc` |
| (fetch) | `min_parking_m2` | 1000 | min candidate parking area (m²) |
| (fetch) | `min_parking_capacity` | 20 | min declared parking spaces (node) |

## Project layout

```
configs/            region configs (plug-and-play entry)
src/
  config.py         dataclasses + YAML loader
  fetch/            nrel, ndot (AADT), acs (Census), osm (Overpass), arcgis, network
  model/            demand grid, coverage matrix, budget/MCLP optimizers
  analysis/         greedy baselines (MCLP + capacitated budget)
  viz/              folium map + matplotlib charts
  web/              static dashboard builder + stdlib web server
  pipeline.py       end-to-end runner / CLI
data/
  raw/              cached fetched data
  output/           map HTML, site CSV, charts
```

## Dependencies

Gurobi (licensed, used by default) with an automatic CBC/PuLP fallback; the
CBC path needs only the `pulp` package. Geopandas, pygris, osmnx/Overpass,
shapely, folium, matplotlib, scipy.

## Notes / limitations

- Census attributes need an **activated** key (the signup email contains a link
  that must be clicked; until then equity demand is zero and results are
  traffic-only).
- DCFC is the more cost-efficient buy (4× throughput at 3× the cost), so an
  unconstrained budget model stacks DC fast chargers; force a mix with
  `min_l2`/`min_dcfc` (MCLP) if you need a balanced L2/DCFC plan.
- Free Overpass instances rate-limit under concurrent load; fetches are cached,
  so re-runs are fast. Network/candidate fetches are tiled and parallelized.
- The first full-region fetch is slow (~5 min for Durham+Orange) but only
  happens once.
- Candidates are OSM parking ways >= `min_parking_m2` (or nodes declaring
  >= `min_parking_capacity` spaces); delete `data/raw/osm_candidate_sites.geojson`
  to force a refetch after changing those thresholds.
