"""MCLP optimizer for EV charger placement.

Two formulations:

1. ``solve_mclp`` — classic two-type Maximal Coverage Location Problem with
   exactly ``k`` sites, one charger per site, and an optional gamma penalty.

   Determines "which sites, which type" (a siting tool).

2. ``solve_budget`` — capacitated, budgeted flow-based model. We spend a fixed
   budget on chargers (any cost, any class, integer counts). A site can host a
   *cluster* of chargers; charger throughput (capacity) limits how much nearby
   demand it can actually serve, so high-demand areas justify more plugs.

   Determine "how many chargers of each type at each site" to maximize demand
   served (a sizing + siting tool).

Capacitated (budget) model
--------------------------
    maximize   sum_{i,j,t} f_ijt
    s.t.       sum_{j,t} cost_t * y_jt  <= B          (budget)
               sum_t y_jt              <= site_max_j  (per-site parking limit)
               sum_{j,t} f_ijt         <= w_i         (cell demand served)
               sum_i f_ijt             <= cap_t * y_jt (charger throughput)
               f_ijt                   = 0  if a_ijt = 0
               y_jt integer, f_ijt >= 0 continuous

w_i     demand units of cell i
a_ijt   coverage: type-t site j can serve cell i
cost_t  installed cost per charger of type t
cap_t   demand units one charger of type t can serve
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SOLVERS = ("auto", "gurobi", "cbc")


def _available_solvers() -> list[str]:
    avail = []
    try:
        import gurobipy  # noqa: F401
        avail.append("gurobi")
    except ImportError:
        pass
    try:
        import pulp  # noqa: F401
        avail.append("cbc")
    except ImportError:
        pass
    return avail


def solve_mclp(
    demand_w: np.ndarray,
    A_l2: np.ndarray,
    A_dcfc: np.ndarray,
    k: int,
    gamma: float = 0.5,
    min_l2: int = 0,
    min_dcfc: int = 0,
    solver: str = "auto",
    time_limit_s: int = 120,
    mip_gap: float = 0.01,
) -> dict:
    """Solve the two-type MCLP. Returns a result dict.

    demand_w: (n_demand,) weights
    A_l2:     (n_sites, n_demand) bool coverage for Level 2
    A_dcfc:   (n_sites, n_demand) bool coverage for DCFC

    Returns:
      dict with 'status', 'objective', 'site_types' (list of (site_idx, type)),
      'covered', 'coverage_dups', 'solution_metrics'.
    """
    n_sites = A_l2.shape[0]
    n_demand = len(demand_w)
    demand_w = np.asarray(demand_w, dtype=float)

    if solver == "auto":
        solvers = _available_solvers()
        solver = "gurobi" if "gurobi" in solvers else "cbc"

    if solver == "gurobi":
        return _solve_gurobi(
            demand_w, A_l2, A_dcfc, k, gamma, min_l2, min_dcfc,
            time_limit_s, mip_gap,
        )
    elif solver == "cbc":
        return _solve_cbc(
            demand_w, A_l2, A_dcfc, k, gamma, min_l2, min_dcfc, time_limit_s
        )
    else:
        raise ValueError(f"Unknown solver: {solver}. Choose from {SOLVERS}")


def _solve_gurobi(demand_w, A_l2, A_dcfc, k, gamma, min_l2, min_dcfc,
                  time_limit_s, mip_gap):
    import gurobipy as gp
    from gurobipy import GRB

    n_sites = A_l2.shape[0]
    n_demand = len(demand_w)

    m = gp.Model("ev_mclp")
    m.Params.TimeLimit = time_limit_s
    m.Params.MIPGap = mip_gap
    m.Params.LogToConsole = 0

    y = m.addVars(n_sites, 2, vtype=GRB.BINARY, name="open")
    x = m.addVars(n_demand, vtype=GRB.BINARY, name="covered")
    u = m.addVars(n_demand, lb=0.0, vtype=GRB.CONTINUOUS, name="excess")

    # Objective: maximize covered demand, penalize duplicated coverage
    m.setObjective(
        gp.quicksum(demand_w[i] * x[i] for i in range(n_demand))
        - gamma * gp.quicksum(demand_w[i] * u[i] for i in range(n_demand)),
        GRB.MAXIMIZE,
    )

    # Total sites = k
    m.addConstr(gp.quicksum(y[j, t] for j in range(n_sites) for t in range(2)) == k,
                "total_sites")
    # At most one type per site
    m.addConstrs(
        (y[j, 0] + y[j, 1] <= 1 for j in range(n_sites)), "one_type"
    )
    # Type minimums
    if min_l2 > 0:
        m.addConstr(gp.quicksum(y[j, 0] for j in range(n_sites)) >= min_l2, "min_l2")
    if min_dcfc > 0:
        m.addConstr(gp.quicksum(y[j, 1] for j in range(n_sites)) >= min_dcfc,
                    "min_dcfc")

    # Coverage validity
    for i in range(n_demand):
        m.addConstr(
            x[i] <= gp.quicksum(
                A_l2[j, i] * y[j, 0] + A_dcfc[j, i] * y[j, 1]
                for j in range(n_sites)
            ),
            f"cov_{i}",
        )

    # Excess coverage (spreading mechanism)
    for i in range(n_demand):
        m.addConstr(
            u[i] >= gp.quicksum(
                A_l2[j, i] * y[j, 0] + A_dcfc[j, i] * y[j, 1]
                for j in range(n_sites)
            )
            - 1,
            f"excess_{i}",
        )

    m.optimize()

    if m.status in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT) and m.SolCount > 0:
        site_types = []
        for j in range(n_sites):
            if y[j, 0].X > 0.5:
                site_types.append((j, "l2"))
            elif y[j, 1].X > 0.5:
                site_types.append((j, "dcfc"))
        return _package_result(m, site_types, demand_w, A_l2, A_dcfc)
    else:
        return {"status": str(m.status), "objective": None, "site_types": [],
                "covered": [], "coverage_dups": None, "solution_metrics": None}


def _solve_cbc(demand_w, A_l2, A_dcfc, k, gamma, min_l2, min_dcfc, time_limit_s):
    import pulp

    n_sites = A_l2.shape[0]
    n_demand = len(demand_w)

    prob = pulp.LpProblem("ev_mclp", pulp.LpMaximize)
    y = pulp.LpVariable.dicts("y", ((j, t) for j in range(n_sites) for t in range(2)),
                              cat="Binary")
    x = pulp.LpVariable.dicts("x", range(n_demand), cat="Binary")
    u = pulp.LpVariable.dicts("u", range(n_demand), lowBound=0, cat="Continuous")

    prob += (
        pulp.lpSum(demand_w[i] * x[i] for i in range(n_demand))
        - gamma * pulp.lpSum(demand_w[i] * u[i] for i in range(n_demand))
    )

    prob += pulp.lpSum(y[j, t] for j in range(n_sites) for t in range(2)) == k
    for j in range(n_sites):
        prob += y[j, 0] + y[j, 1] <= 1
    if min_l2 > 0:
        prob += pulp.lpSum(y[j, 0] for j in range(n_sites)) >= min_l2
    if min_dcfc > 0:
        prob += pulp.lpSum(y[j, 1] for j in range(n_sites)) >= min_dcfc

    for i in range(n_demand):
        prob += x[i] <= pulp.lpSum(
            A_l2[j, i] * y[j, 0] + A_dcfc[j, i] * y[j, 1]
            for j in range(n_sites)
        )
        prob += u[i] >= pulp.lpSum(
            A_l2[j, i] * y[j, 0] + A_dcfc[j, i] * y[j, 1]
            for j in range(n_sites)
        ) - 1

    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit_s)
    prob.solve(solver)

    if pulp.LpStatus[prob.status] in ("Optimal", "Not Solved"):
        site_types = []
        for j in range(n_sites):
            if y[j, 0].value() and y[j, 0].value() > 0.5:
                site_types.append((j, "l2"))
            elif y[j, 1].value() and y[j, 1].value() > 0.5:
                site_types.append((j, "dcfc"))
        return _package_result(prob, site_types, demand_w, A_l2, A_dcfc,
                               is_pulp=True)
    else:
        return {"status": pulp.LpStatus[prob.status], "objective": None,
                "site_types": [], "covered": [], "coverage_dups": None,
                "solution_metrics": None}


def _package_result(model, site_types, demand_w, A_l2, A_dcfc, is_pulp=False):
    n_demand = len(demand_w)
    # Compute which cells are covered and their multiplicity
    covered_cover = np.zeros(n_demand)
    for j, t in site_types:
        cov = A_dcfc[j] if t == "dcfc" else A_l2[j]
        covered_cover += cov.astype(float)
    covered = covered_cover > 0
    weight_total = demand_w.sum() if demand_w.sum() else 1.0
    weight_covered = demand_w[covered].sum()

    if is_pulp:
        import pulp
        obj = model.objective.value() if model.objective else None
        status = pulp.LpStatus[model.status]
    else:
        obj = model.ObjVal if model.SolCount > 0 else None
        status = str(model.status)

    metrics = {
        "n_sites": len(site_types),
        "n_demand": n_demand,
        "n_covered_cells": int(covered.sum()),
        "n_uncovered_cells": int((~covered).sum()),
        "frac_cells_covered": float(covered.mean()),
        "weight_covered": float(weight_covered),
        "frac_weight_covered": float(weight_covered / weight_total),
        "duplicated_coverage": float((covered_cover[covered] - 1).sum()),
        "mean_coverage_per_covered": float(covered_cover[covered].mean())
        if covered.any() else 0.0,
    }
    return {
        "status": status,
        "objective": obj,
        "site_types": site_types,
        "covered": covered,
        "coverage_multiplicity": covered_cover,
        "solution_metrics": metrics,
    }


# ---------------------------------------------------------------------------
# Capacitated, budgeted flow-based model
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Capacitated, budgeted flow-based model (V2)
# ---------------------------------------------------------------------------

L2_IDX, DCFC_IDX = 0, 1
_TYPE_NAMES = ("l2", "dcfc")


def solve_budget(
    demand_w: np.ndarray,
    A_l2: np.ndarray,
    A_dcfc: np.ndarray,
    budget: float,
    cost: dict,
    capacity: dict,
    site_max=6,
    solver: str = "auto",
    time_limit_s: int = 120,
    mip_gap: float = 0.01,
    initial_solution: list[tuple[int, str, int]] | None = None,
    *,
    min_l2: int = 0,
    min_dcfc: int = 0,
    min_sites_used: int = 0,
    max_sites_used: int = 0,
    spacing_pairs: list[tuple[int, int]] | None = None,
    coverage_bonus: float = 0.0,
) -> dict:
    """Solve the V2 budgeted siting/sizing model.

    Relative to V1, this formulation adds:
    - type minimums that are ACTUALLY enforced in the primary budget model;
    - binary site-activation variables so a minimum number of distinct places
      can be required;
    - optional minimum-spacing conflicts between active sites;
    - a first-access bonus that rewards geographic coverage in addition to raw
      charger throughput.
    """
    demand_w = np.asarray(demand_w, dtype=float)
    A_l2 = np.asarray(A_l2, dtype=bool)
    A_dcfc = np.asarray(A_dcfc, dtype=bool)
    if A_l2.shape != A_dcfc.shape:
        raise ValueError("L2 and DCFC coverage matrices must have the same shape")
    n_sites, n_demand = A_l2.shape
    if n_demand != len(demand_w):
        raise ValueError("Coverage matrix demand dimension does not match demand_w")

    cost_arr = np.array([cost["l2"], cost["dcfc"]], dtype=float)
    cap_arr = np.array([capacity["l2"], capacity["dcfc"]], dtype=float)
    A = np.stack([A_l2, A_dcfc]).astype(np.int8)
    site_max_arr = np.broadcast_to(np.asarray(site_max, dtype=float), (n_sites,)).copy()
    site_max_arr = np.maximum(0, np.floor(site_max_arr)).astype(int)
    spacing_pairs = spacing_pairs or []

    if min_sites_used > int((site_max_arr > 0).sum()):
        raise ValueError("min_sites_used exceeds the number of eligible candidate sites")
    if min_l2 * cost_arr[0] + min_dcfc * cost_arr[1] > budget + 1e-9:
        raise ValueError("Type minimums cost more than the total budget")

    chosen = solver
    if chosen == "auto":
        available = _available_solvers()
        chosen = "gurobi" if "gurobi" in available else "cbc"

    if chosen == "gurobi":
        try:
            return _solve_budget_gurobi_v2(
                demand_w, A, cost_arr, cap_arr, budget, site_max_arr,
                time_limit_s, mip_gap, initial_solution,
                min_l2, min_dcfc, min_sites_used, max_sites_used,
                spacing_pairs, float(coverage_bonus),
            )
        except Exception as e:
            # Auto mode should not die merely because gurobipy imports but the
            # local machine lacks a usable license. Fall back to CBC/PuLP.
            if solver == "auto":
                import warnings
                warnings.warn(f"Gurobi unavailable at solve time ({e}); falling back to CBC.")
                chosen = "cbc"
            else:
                raise

    if chosen == "cbc":
        return _solve_budget_cbc_v2(
            demand_w, A, cost_arr, cap_arr, budget, site_max_arr,
            time_limit_s, min_l2, min_dcfc, min_sites_used, max_sites_used,
            spacing_pairs, float(coverage_bonus),
        )
    raise ValueError(f"Unknown solver: {solver}. Choose from {SOLVERS}")


def _package_budget_v2(y_vals, z_vals, f_vals, x_vals, demand_w, cost_arr,
                       status=None, objective=None, solver_used=None):
    n_sites = len(z_vals)
    y = np.zeros((n_sites, 2), dtype=int)
    for (j, t), value in y_vals.items():
        if value is not None:
            y[j, t] = max(0, int(round(float(value))))

    chargers = [
        (j, _TYPE_NAMES[t], int(y[j, t]))
        for j in range(n_sites) for t in range(2) if y[j, t] > 0
    ]
    active_sites = [j for j, v in z_vals.items() if v is not None and float(v) > 0.5]

    served = np.zeros(len(demand_w), dtype=float)
    for (_j, _t, i), value in f_vals.items():
        if value is not None:
            served[i] += max(0.0, float(value))
    covered = np.array([
        bool(x_vals.get(i, 0) is not None and float(x_vals.get(i, 0)) > 0.5)
        for i in range(len(demand_w))
    ])

    budget_spent = float(np.dot(y.sum(axis=0), cost_arr))
    total_w = float(demand_w.sum()) if demand_w.sum() else 1.0
    metrics = {
        "n_demand": int(len(demand_w)),
        "n_chargers_total": int(y.sum()),
        "budget_spent": budget_spent,
        "n_sites_used": int(len(active_sites)),
        "demand_units_served": float(served.sum()),
        "frac_demand_served": float(served.sum() / total_w),
        "first_access_weight": float(demand_w[covered].sum()),
        "frac_first_access_weight": float(demand_w[covered].sum() / total_w),
        "chargers_by_type": {
            "l2": int(y[:, 0].sum()),
            "dcfc": int(y[:, 1].sum()),
        },
        "solver_used": solver_used,
    }
    return {
        "status": status,
        "objective": objective,
        "objective_value": objective,
        "site_chargers": chargers,
        "active_sites": active_sites,
        "budget_spent": budget_spent,
        "chargers": chargers,
        "served": served,
        "covered": covered,
        "metrics": metrics,
        "solution_metrics": metrics,
    }


def _solve_budget_gurobi_v2(
    demand_w, A, cost, cap, budget, site_max, time_limit_s, mip_gap,
    initial_solution, min_l2, min_dcfc, min_sites_used, max_sites_used,
    spacing_pairs, coverage_bonus,
):
    import gurobipy as gp
    from gurobipy import GRB

    n_sites = A.shape[1]
    n_demand = A.shape[2]
    m = gp.Model("ev_budget_v2")
    m.Params.TimeLimit = float(time_limit_s)
    m.Params.MIPGap = float(mip_gap)
    m.Params.LogToConsole = 0
    m.Params.Threads = 0  # all logical CPU cores; Gurobi is CPU, not GPU
    m.Params.MIPFocus = 2

    y = m.addVars(n_sites, 2, vtype=GRB.INTEGER, lb=0, name="chargers")
    z = m.addVars(n_sites, vtype=GRB.BINARY, name="site_active")
    x = m.addVars(n_demand, vtype=GRB.BINARY, name="first_access")

    f = {}
    for j in range(n_sites):
        for t in range(2):
            for i in np.where(A[t, j])[0]:
                f[j, t, int(i)] = m.addVar(
                    lb=0.0, ub=float(demand_w[i]), name=f"flow_{j}_{t}_{i}"
                )

    m.setObjective(
        gp.quicksum(f.values())
        + coverage_bonus * gp.quicksum(demand_w[i] * x[i] for i in range(n_demand)),
        GRB.MAXIMIZE,
    )

    m.addConstr(
        gp.quicksum(cost[t] * y[j, t] for j in range(n_sites) for t in range(2)) <= budget,
        "budget",
    )
    for j in range(n_sites):
        m.addConstr(y[j, 0] + y[j, 1] <= int(site_max[j]) * z[j], f"sitecap_{j}")
        m.addConstr(y[j, 0] + y[j, 1] >= z[j], f"activate_{j}")

    if min_l2 > 0:
        m.addConstr(gp.quicksum(y[j, 0] for j in range(n_sites)) >= min_l2, "min_l2")
    if min_dcfc > 0:
        m.addConstr(gp.quicksum(y[j, 1] for j in range(n_sites)) >= min_dcfc, "min_dcfc")
    if min_sites_used > 0:
        m.addConstr(gp.quicksum(z[j] for j in range(n_sites)) >= min_sites_used, "min_sites")
    if max_sites_used and max_sites_used > 0:
        m.addConstr(gp.quicksum(z[j] for j in range(n_sites)) <= max_sites_used, "max_sites")
    for a, b in spacing_pairs:
        m.addConstr(z[int(a)] + z[int(b)] <= 1, f"spacing_{a}_{b}")

    for i in range(n_demand):
        flow_vars = [f[j, t, i] for j in range(n_sites) for t in range(2) if (j, t, i) in f]
        if flow_vars:
            m.addConstr(gp.quicksum(flow_vars) <= demand_w[i], f"demand_{i}")
        else:
            m.addConstr(x[i] == 0, f"unreachable_{i}")
        # x rewards existence of at least one charger covering the cell, not
        # number of chargers at the same site.
        cover_terms = [
            y[j, t]
            for j in range(n_sites) for t in range(2) if A[t, j, i]
        ]
        if cover_terms:
            m.addConstr(x[i] <= gp.quicksum(cover_terms), f"access_{i}")
        else:
            m.addConstr(x[i] == 0, f"no_access_{i}")

    for j in range(n_sites):
        for t in range(2):
            vars_jt = [f[j, t, i] for i in range(n_demand) if (j, t, i) in f]
            if vars_jt:
                m.addConstr(gp.quicksum(vars_jt) <= cap[t] * y[j, t], f"throughput_{j}_{t}")

    if initial_solution:
        for j, t_name, count in initial_solution:
            t = 1 if t_name == "dcfc" else 0
            if 0 <= j < n_sites:
                y[j, t].Start = int(count)
                z[j].Start = 1 if count > 0 else z[j].Start

    m.optimize()
    if m.status in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT) and m.SolCount > 0:
        return _package_budget_v2(
            {(j, t): y[j, t].X for j in range(n_sites) for t in range(2)},
            {j: z[j].X for j in range(n_sites)},
            {(j, t, i): var.X for (j, t, i), var in f.items()},
            {i: x[i].X for i in range(n_demand)},
            demand_w, cost,
            status=m.status, objective=float(m.ObjVal), solver_used="gurobi",
        )
    return {"status": str(m.status), "objective": None, "site_chargers": [],
            "chargers": [], "metrics": None, "solution_metrics": None}


def _solve_budget_cbc_v2(
    demand_w, A, cost, cap, budget, site_max, time_limit_s,
    min_l2, min_dcfc, min_sites_used, max_sites_used, spacing_pairs,
    coverage_bonus,
):
    try:
        import pulp
    except ImportError as e:
        raise RuntimeError(
            "No optimization solver is available. Install 'pulp' for the CBC fallback "
            "or install/configure Gurobi."
        ) from e

    n_sites = A.shape[1]
    n_demand = A.shape[2]
    prob = pulp.LpProblem("ev_budget_v2", pulp.LpMaximize)
    y = pulp.LpVariable.dicts(
        "y", ((j, t) for j in range(n_sites) for t in range(2)),
        lowBound=0, cat="Integer"
    )
    z = pulp.LpVariable.dicts("z", range(n_sites), cat="Binary")
    x = pulp.LpVariable.dicts("x", range(n_demand), cat="Binary")
    f = pulp.LpVariable.dicts(
        "f",
        ((j, t, int(i)) for j in range(n_sites) for t in range(2)
         for i in np.where(A[t, j])[0]),
        lowBound=0, cat="Continuous",
    )

    prob += (
        pulp.lpSum(f.values())
        + coverage_bonus * pulp.lpSum(demand_w[i] * x[i] for i in range(n_demand))
    )
    prob += pulp.lpSum(cost[t] * y[j, t] for j in range(n_sites) for t in range(2)) <= budget

    for j in range(n_sites):
        prob += y[j, 0] + y[j, 1] <= int(site_max[j]) * z[j]
        prob += y[j, 0] + y[j, 1] >= z[j]
    if min_l2 > 0:
        prob += pulp.lpSum(y[j, 0] for j in range(n_sites)) >= min_l2
    if min_dcfc > 0:
        prob += pulp.lpSum(y[j, 1] for j in range(n_sites)) >= min_dcfc
    if min_sites_used > 0:
        prob += pulp.lpSum(z[j] for j in range(n_sites)) >= min_sites_used
    if max_sites_used and max_sites_used > 0:
        prob += pulp.lpSum(z[j] for j in range(n_sites)) <= max_sites_used
    for a, b in spacing_pairs:
        prob += z[int(a)] + z[int(b)] <= 1

    for i in range(n_demand):
        flow_vars = [f[j, t, i] for j in range(n_sites) for t in range(2) if (j, t, i) in f]
        if flow_vars:
            prob += pulp.lpSum(flow_vars) <= demand_w[i]
        else:
            prob += x[i] == 0
        cover_terms = [A[t, j, i] * y[j, t] for j in range(n_sites) for t in range(2) if A[t, j, i]]
        if cover_terms:
            prob += x[i] <= pulp.lpSum(cover_terms)
        else:
            prob += x[i] == 0

    for j in range(n_sites):
        for t in range(2):
            vars_jt = [f[j, t, i] for i in range(n_demand) if (j, t, i) in f]
            if vars_jt:
                prob += pulp.lpSum(vars_jt) <= cap[t] * y[j, t]

    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=float(time_limit_s), threads=0)
    prob.solve(solver)
    status = pulp.LpStatus[prob.status]
    if status in ("Optimal", "Not Solved"):
        return _package_budget_v2(
            {(j, t): y[j, t].value() for j in range(n_sites) for t in range(2)},
            {j: z[j].value() for j in range(n_sites)},
            {(j, t, i): var.value() for (j, t, i), var in f.items()},
            {i: x[i].value() for i in range(n_demand)},
            demand_w, cost,
            status=status, objective=pulp.value(prob.objective), solver_used="cbc",
        )
    return {"status": status, "objective": None, "site_chargers": [],
            "chargers": [], "metrics": None, "solution_metrics": None}
