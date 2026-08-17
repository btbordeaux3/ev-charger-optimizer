# EV Charger Placement Optimization — V3

A cross-platform, equity-aware public EV-charger siting optimizer with **network travel-time accessibility**. The project fetches the study region from county FIPS codes, builds walk/drive networks, finds parking-rich candidate sites, solves a capacitated budget MILP, and produces a local interactive dashboard.

V2 fixed the original geography/optimization failure; **V3 adds an auditable zoning, parcel, parking, and EV-regulation screening layer**. The redesign began after a Durham run put essentially the whole optimized build at one parking site outside the intended study area.

## What V2/V3 fixes

The bad one-point result was not just a plotting problem. Several V1 assumptions pushed the optimizer toward that outcome:

1. **Candidate parking was fetched by county bounding box but never clipped to the exact county polygon.** A site could therefore be outside Durham and still be eligible.
2. **OSM caches were global filenames.** A run for one region could silently reuse another region's candidate/network cache.
3. **The primary budget model ignored `min_l2`, `min_dcfc`, and the legacy spread control.** Because DC fast had more modeled throughput per dollar, the MILP could rationally stack DCFC at one location.
4. **Every site had the same large charger cap.** V2/V3 estimates parking capacity and derives a site-specific charger cap.
5. **One hand-built graph was used for both driving and walking and road direction was lost.** V2/V3 uses separate OSMnx `walk` and `drive` graphs and retains directed travel.
6. **A hard-coded North Carolina UTM CRS made the supposedly plug-and-play project geographically fragile.** V2/V3 estimates the local metric CRS from the study region.
7. **The old Dijkstra multiprocessing path depended on `fork`, which is not portable to native Windows.** V2/V3 uses batched SciPy routing with cross-platform threads, plus an optional cuGraph backend.
8. **Census tract totals were effectively repeated in every grid cell.** V2/V3 converts tract values to spatial density and allocates them by cell area.
9. **Existing chargers were only a display layer.** V2/V3 can reduce the optimization weight of demand already accessible to public chargers so the solution prioritizes infrastructure gaps.
10. **Output chart/dashboard names were Durham-specific.** V2/V3 uses generic filenames and adds a real accessibility summary.

## V3 regulatory / zoning screening

V3 can attach every OSM parking candidate to an authoritative parcel layer and then apply a versioned, jurisdiction-specific rule profile. For Durham, `configs/durham_only.yaml` uses Durham's public parcel GIS layer to add:

- parcel ID and address;
- zoning code and land class;
- unit count, heated floor area, and gross leasable area when available;
- applicable parking-minimum / parking-maximum screening fields;
- potentially applicable EV-space requirements;
- a regulatory status and manual-review flag; and
- the final site charger cap after physical + planning screening.

The resulting `regulatory_audit.csv` is deliberately separate from the optimization output so every assumption can be inspected.

### Important Durham-specific modeling choice

Do **not** encode a made-up "minimum parking by zoning district" for Durham. The current electronic Durham UDO parking-location table lists the motor-vehicle parking **minimum as `None`** for Downtown Design, Suburban/Rural, Urban, Compact Neighborhood, CSD, and CI locations. The use-based parking schedule is a *baseline calculator* that is then subject to location-specific **maximums** (for example 50% of baseline in CSD-C, 100% in several compact/design locations, and 175% in Suburban/Rural and much of the Urban tier).

That means V3 records Durham's location parking minimum as zero for screening rather than pretending that converting an existing stall to an EV stall deletes a required parking space. An EV charging stall is still a parking stall. The model separately keeps a conservative planning guardrail on the share of an existing lot converted to charging in one phase.

Durham also has a **conditional** rule in the DD/CD building-height provisions: when a development supplies qualifying public parking to obtain additional height, at least three of those public parking spaces must be EV charging spaces. V3 flags DD/CD candidates for manual review and reports `reg_ev_min_installed = 3`; it does **not** incorrectly force three chargers onto every existing DD/CD lot.

The Durham profile is based on:

- Durham UDO Sec. 10.3, Parking Rates: `https://udo.durhamnc.gov/udo/10_03_Required%20Parking.htm`
- Durham UDO Sec. 16.3, Building Design: `https://udo.durhamnc.gov/udo/16_03_Building%20Design.htm`
- Durham public Parcel Boundary layer (including `REID`, `ZONING`, `LAND_CLASS`, `TOTAL_UNITS`, `HEATED_AREA`, and `GROSS_LEASABLE_AREA`): `https://webgis.durhamnc.gov/server/rest/services/PublicWorksServices/DurhamGISReferenceLayers/MapServer/116`

**Regulatory disclaimer:** the electronic UDO itself states that it is not the official version and may not include recently adopted amendments. Treat this layer as planning screening, not a permit/legal determination. Before a real installation, confirm the current official UDO, building/accessibility requirements, utility requirements, ownership/easements, and site-plan approvals with the applicable agencies.

### Generic rule schema for another city

The same code supports local YAML rules using selectors such as `zone_regex` and `land_class_regex`, with fields including:

```yaml
regulations:
  enabled: true
  parcel_layer_url: "https://.../FeatureServer/0"
  zoning_field: "ZONE"
  units_field: "UNITS"
  floor_area_field: "SQFT"
  max_ev_share_existing_spaces: 0.15
  unknown_context_policy: "allow_with_review"
  rules:
    - name: "Apartment parking minimum"
      zone_regex: "^RM"
      parking_min_per_unit: 1.25

    - name: "Retail parking screen"
      land_class_regex: "RETAIL"
      parking_min_per_1000_sf: 2.0

    - name: "EV requirement"
      zone_regex: "^MX"
      ev_min_installed: 4
      applicability: "new_development_only"
      manual_review: true

    - name: "No public charging in protected zone"
      zone_regex: "^PROTECTED$"
      prohibit_ev_charging: true
```

A rule can be made a true exclusion with `prohibit_ev_charging: true`. Unknown GIS context defaults to **allow with manual review**, not silent approval; set `unknown_context_policy: "exclude"` for a stricter screening profile.

## Main model

For every eligible parking facility, the budget model decides how many Level 2 and DC-fast chargers to build.

It maximizes two things together:

- demand units actually served, subject to charger throughput limits; and
- **first access**, so covering a new part of the region is valuable instead of repeatedly adding capacity to one already-covered place.

It also enforces:

- total budget;
- minimum Level 2 and DCFC counts;
- minimum number of distinct physical sites;
- parking-derived per-site charger caps;
- optional minimum spacing between active sites; and
- optional post-solve electric-grid hosting-capacity constraints.

The greedy baseline remains for comparison.

## Accessibility / isochrone-style outputs

The optimization uses network travel time rather than a Euclidean circle:

- **Level 2:** walking time to the charger;
- **DC fast:** driving time to the charger.

The dashboard additionally computes 5-, 10-, and 15-minute service layers. The displayed service areas are unions of the demand-grid cells reachable within each threshold, which makes them stable for citywide coverage accounting.

`coverage_summary.csv` reports, for each threshold:

- weighted demand covered;
- optimization-weighted demand covered;
- estimated population covered; and
- land area covered.

The Folium map lets you toggle:

- combined service areas;
- L2 walking service;
- DCFC driving service;
- weighted demand cells;
- every eligible parking candidate;
- recommended sites; and
- existing public chargers when an NREL/NLR key is supplied.

## Quick start — Windows

Run these commands from the project folder in PowerShell or Command Prompt:

```bat
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Optional Gurobi support:

```bat
pip install -r requirements-gurobi.txt
```

Copy `.env.example` to `.env`, add your keys, then run Durham:

```bat
copy .env.example .env
python -m src.pipeline configs/durham_only.yaml --refresh
```

After the first clean V3 run, omit `--refresh` so the region-specific caches make reruns much faster:

```bat
python -m src.pipeline configs/durham_only.yaml
```

Serve the dashboard:

```bat
python -m src.web.serve 8000 data/output_durham
```

Open `http://localhost:8000/index.html`.

## Quick start — macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
python -m src.pipeline configs/durham_only.yaml --refresh
python -m src.web.serve 8000 data/output_durham
```

Optional Gurobi:

```bash
pip install -r requirements-gurobi.txt
```

The same YAML and source files are used on Mac and Windows.

## NVIDIA acceleration

The default configuration is:

```yaml
coverage:
  routing_backend: "auto"
```

That means:

- macOS and ordinary/native Windows installations use the cross-platform SciPy routing implementation;
- if RAPIDS/cuGraph and a usable NVIDIA CUDA device are present, V3 can use the `cugraph` routing backend automatically;
- you can force either path with `--routing-backend scipy` or `--routing-backend cugraph`.

For an NVIDIA Windows laptop, the practical GPU route is to run the repository inside a CUDA-enabled **WSL2 Linux** environment and install RAPIDS/cuGraph there. Use the RAPIDS installation selector for versions matching your CUDA/driver rather than hard-pinning GPU packages in this cross-platform repository.

Important: the MILP solver itself is not converted into a CUDA job here. Gurobi/CBC remain CPU optimization solvers. The optional NVIDIA path accelerates the repeated shortest-path/accessibility work, which is the part of this pipeline that maps naturally to cuGraph.

If you want the least setup friction, stay native Windows and let `routing_backend: auto` fall back to SciPy. The code still uses multiple CPU cores where appropriate.

## API keys

Create `.env` from `.env.example`:

```text
NREL_API_KEY=...
CENSUS_API_KEY=...
```

- Without NREL/NLR, the optimizer still runs; it simply cannot downweight areas already served by existing public chargers.
- Without Census ACS, the optimizer can still run on the traffic component, but the equity component is zero.

Do not commit `.env`.

## Durham configurations

### Durham County only

```bash
python -m src.pipeline configs/durham_only.yaml --refresh
```

Outputs:

```text
data/output_durham/
  index.html
  ev_charger_map.html
  map.html
  recommended_sites.csv
  coverage_summary.csv
  regulatory_audit.csv
  charts/
    before_after.png
    budget_mix.png
    equity.png        # only when tract income data are available
```

### Durham + Orange

```bash
python -m src.pipeline configs/durham_orange.yaml --refresh
```

Outputs go to `data/output/`.

## Important V3 configuration knobs

```yaml
coverage:
  l2_walk_time_min: 10
  dcfc_drive_time_min: 10
  map_thresholds_min: [5, 10, 15]
  direction: "to_site"
  routing_backend: "auto"

optimization:
  min_l2: 2
  min_dcfc: 3
  min_sites_used: 5
  min_site_spacing_m: 0
  coverage_bonus: 0.25

budget:
  site_max: 6
  parking_spaces_per_charger: 10
  parking_m2_per_space: 30

existing:
  covered_demand_multiplier: 0.35

regulations:
  enabled: true
  max_ev_share_existing_spaces: 0.15
  unknown_context_policy: "allow_with_review"
```

### If recommendations still cluster too much

Do not immediately shrink the routing threshold. First try one of these:

```yaml
optimization:
  min_sites_used: 7
  coverage_bonus: 0.50
```

or add a modest hard spacing rule:

```yaml
optimization:
  min_site_spacing_m: 750
```

A hard spacing rule changes the feasible set, so use it only when you actually want to prohibit nearby sites. `coverage_bonus` is the softer and usually preferable way to reward geographic reach.

## New region

Copy a config and change the county list:

```yaml
name: "Wake County NC"
counties:
  - name: "Wake"
    state_fips: "37"
    county_fips: "183"
    state_abbr: NC
```

The county union controls:

- exact candidate eligibility;
- demand-grid clipping;
- region-specific cache names; and
- local projected CRS selection.

NCDOT AADT is currently the built-in traffic source for North Carolina. A non-NC region still runs, but the traffic component is zero until an additional state/local traffic adapter is configured.

## Refreshing data safely

V3 caches OSM and regulatory-context files under a unique region namespace, so Durham cannot accidentally load Wake/Orange network files.

Use:

```bash
python -m src.pipeline configs/durham_only.yaml --refresh
```

when you change:

- study geography;
- parking eligibility thresholds;
- routing-network buffers; or
- anything that should force fresh OSM candidate/network retrieval.

Normal reruns should omit it.

## Debugging a suspicious solution

The map contains a **Parking candidates (post-screening)** layer. Turn it on before trusting the optimization. If any candidate appears outside the study boundary, that is a data-pipeline bug, not an optimization preference.

Also inspect:

```text
recommended_sites.csv
coverage_summary.csv
regulatory_audit.csv
```

Every recommendation now carries its original `site_index`, OSM identifiers, parking estimate, parcel/zoning context when matched, regulatory flags, and final per-site charger cap. The regulatory audit also includes candidates that were not selected.

## Project layout

```text
configs/
  durham_only.yaml
  durham_orange.yaml
src/
  config.py
  fetch/
    acs.py
    ndot.py
    nrel.py
    network.py
    osm.py
    regulations.py
  model/
    demand.py
    coverage.py
    optimize.py
    grid_check.py
  analysis/
    baseline.py
  viz/
    map.py
  web/
    dashboard.py
    serve.py
  pipeline.py
tests/
data/
  raw/
  output/
```

## Recommended first verification run

For the first Durham V3 test:

```bash
python -m src.pipeline configs/durham_only.yaml --refresh --verbose
```

Then check five things before tuning the model:

1. the study-boundary layer actually outlines Durham County;
2. candidate points are all inside it;
3. `regulatory_audit.csv` shows sensible Durham parcel/zoning matches and does not silently classify unmatched candidates as approved;
4. `recommended_sites.csv` contains at least the configured number of distinct site indices; and
5. the 5/10/15-minute coverage layers expand in a geographically sensible way along the network.

If those five pass, then it makes sense to start tuning budget, equity weight, charger capacities, or the geographic-spread preference.
