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

# Charger class indices (must match order of coverage matrices passed in).
L2_IDX, DCFC_IDX = 0, 1
_TYPE_NAMES = ("l2", "dcfc")


def solve_budget(
    demand_w: np.ndarray,
    A_l2: np.ndarray,
    A_dcfc: np.ndarray,
    budget: float,
    cost: dict,
    capacity: dict,
    site_max: int = 12,
    solver: str = "auto",
    time_limit_s: int = 120,
    mip_gap: float = 0.01,
    initial_solution: list[tuple[int, str, int]] | None = None,
) -> dict:
    """Solve the capacitated budgeted placement problem.

    demand_w : (n_demand,) demand units per cell
    A_l2     : (n_sites, n_demand) bool coverage for Level 2
    A_dcfc   : (n_sites, n_demand) bool coverage for DC fast
    budget   : total spend ($)
    cost     : {'l2': float, 'dcfc': float} installed $ per charger
    capacity : {'l2': float, 'dcfc': float} demand units one charger can serve
    site_max : max total chargers at any single site (parking/cluster cap)
    initial_solution : [(site_idx, 'l2'/'dcfc', count), ...] warm start (e.g. the
        greedy baseline). Seeding the MIP with a good feasible plan guarantees
        the returned objective is never worse than that plan.

    Returns dict: status, objective, objective_value (demand units served),
      budget_spent, chargers (list of (site_idx, 'l2'/'dcfc', count)), and
      metrics.
    """
    n_sites = A_l2.shape[0]
    n_demand = len(demand_w)
    demand_w = np.asarray(demand_w, dtype=float)

    cost_arr = np.array([cost["l2"], cost["dcfc"]], dtype=float)
    cap_arr = np.array([capacity["l2"], capacity["dcfc"]], dtype=float)
    A = np.stack([A_l2, A_dcfc]).astype(int)  # (2, n_sites, n_demand)

    if solver == "auto":
        solvers = _available_solvers()
        solver = "gurobi" if "gurobi" in solvers else "cbc"

    if solver == "gurobi":
        return _solve_budget_gurobi(
            demand_w, A, cost_arr, cap_arr, budget, site_max,
            time_limit_s, mip_gap, initial_solution,
        )
    elif solver == "cbc":
        return _solve_budget_cbc(
            demand_w, A, cost_arr, cap_arr, budget, site_max, time_limit_s
        )
    else:
        raise ValueError(f"Unknown solver: {solver}. Choose from {SOLVERS}")


def _package_budget(y_vars, f_vars, A, demand_w, cost_arr,
                    status=None, objective=None):
    """Package budget-model results from explicit variable containers.

    y_vars : dict {(j,t): value}  derived on call via `.X` or `.value()`
    f_vars : dict {(j,t,i): value}
    """
    n_sites = A.shape[1]

    y = np.zeros((n_sites, 2), dtype=int)
    for (j, t), v in y_vars.items():
        y[j, t] = int(round(v))

    chargers = []
    for j in range(n_sites):
        for t in range(2):
            if y[j, t] > 0:
                chargers.append((j, _TYPE_NAMES[t], int(y[j, t])))

    budget_spent = float(np.dot(y.sum(axis=0), cost_arr))
    n_chargers = int(y.sum())

    n_demand = demand_w.shape[0]
    served = np.zeros(n_demand)
    for (j, t, i), v in f_vars.items():
        served[i] += v

    total_w = demand_w.sum() if demand_w.sum() else 1.0
    served_w = served.sum()
    metrics = {
        "n_demand": n_demand,
        "n_chargers_total": n_chargers,
        "budget_spent": budget_spent,
        "n_sites_used": len(chargers),
        "demand_units_served": float(served_w),
        "frac_demand_served": float(served_w / total_w),
        "chargers_by_type": {
            "l2": int(y[:, 0].sum()),
            "dcfc": int(y[:, 1].sum()),
        },
    }
    return {
        "status": status,
        "objective": objective,
        "objective_value": objective,
        "site_chargers": chargers,
        "budget_spent": budget_spent,
        "chargers": chargers,
        "served": served,
        "metrics": metrics,
        "solution_metrics": metrics,
    }


def _solve_budget_gurobi(demand_w, A, cost, cap, budget, site_max,
                         time_limit_s, mip_gap, initial_solution=None):
    import gurobipy as gp
    from gurobipy import GRB

    n_sites = A.shape[1]
    n_demand = A.shape[2]

    m = gp.Model("ev_budget")
    m.Params.TimeLimit = time_limit_s
    m.Params.MIPGap = mip_gap
    m.Params.LogToConsole = 0
    m.Params.MIPFocus = 2  # prioritize improving the incumbent over proving bound

    y = m.addVars(n_sites, 2, vtype=GRB.INTEGER, lb=0, name="y")
    # flow vars only where coverage exists, to keep the LP sparse
    f = {}
    for j in range(n_sites):
        for t in range(2):
            idx = np.where(A[t, j])[0]
            for i in idx:
                f[j, t, i] = m.addVar(lb=0.0, ub=demand_w[i], name=f"f_{j},{t},{i}")

    m.setObjective(gp.quicksum(f.values()), GRB.MAXIMIZE)

    m.addConstr(
        gp.quicksum(cost[t] * y[j, t] for j in range(n_sites) for t in range(2))
        <= budget,
        name="budget",
    )
    for j in range(n_sites):
        m.addConstr(y[j, 0] + y[j, 1] <= site_max, name=f"sitemax_{j}")
    for i in range(n_demand):
        m.addConstr(
            gp.quicksum(f[j, t, i] for j in range(n_sites) for t in range(2)
                        if (j, t, i) in f) <= demand_w[i],
            name=f"dem_{i}",
        )
    for j in range(n_sites):
        for t in range(2):
            m.addConstr(
                gp.quicksum(f[j, t, i] for i in range(n_demand)
                            if (j, t, i) in f) <= cap[t] * y[j, t],
                name=f"cap_{j}_{t}",
            )

    if initial_solution:
        # warm start: seed y from a feasible plan (e.g. the greedy baseline)
        for j, t_name, count in initial_solution:
            t = 1 if t_name == "dcfc" else 0
            y[j, t].Start = int(count)

    m.optimize()
    if m.status in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT) and m.SolCount > 0:
        y_vars = {(j, t): y[j, t].X for (j, t), _ in y.items()}
        f_vars = {(j, t, i): v.X for (j, t, i), v in f.items()}
        return _package_budget(y_vars, f_vars, A, demand_w, cost,
                               status=m.status, objective=m.ObjVal)
    return {"status": str(m.status), "objective": None, "chargers": [],
            "site_chargers": [], "metrics": None, "solution_metrics": None}


def _solve_budget_cbc(demand_w, A, cost, cap, budget, site_max, time_limit_s):
    import pulp

    n_sites = A.shape[1]
    n_demand = A.shape[2]

    prob = pulp.LpProblem("ev_budget", pulp.LpMaximize)
    y = pulp.LpVariable.dicts("y", ((j, t) for j in range(n_sites) for t in range(2)),
                              lowBound=0, cat="Integer")
    f = pulp.LpVariable.dicts("f", (
        (j, t, i) for j in range(n_sites) for t in range(2)
        for i in np.where(A[t, j])[0]),
        lowBound=0, cat="Continuous")

    prob += pulp.lpSum(f.values())
    prob += pulp.lpSum(cost[t] * y[j, t] for j in range(n_sites) for t in range(2)) <= budget
    for j in range(n_sites):
        prob += y[j, 0] + y[j, 1] <= site_max
    for i in range(n_demand):
        prob += pulp.lpSum(f[j, t, i] for j in range(n_sites) for t in range(2)
                           if (j, t, i) in f) <= demand_w[i]
    for j in range(n_sites):
        for t in range(2):
            prob += pulp.lpSum(f[j, t, i] for i in range(n_demand)
                               if (j, t, i) in f) <= cap[t] * y[j, t]

    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit_s)
    prob.solve(solver)
    if pulp.LpStatus[prob.status] in ("Optimal", "Not Solved"):
        y_vars = {(j, t): y[j, t].value() for (j, t), _ in y.items()}
        f_vars = {(j, t, i): f[j, t, i].value() for (j, t, i), _ in f.items()}
        return _package_budget(y_vars, f_vars, A, demand_w, cost,
                               status=prob.status,
                               objective=pulp.value(prob.objective))
    return {"status": pulp.LpStatus[prob.status], "objective": None,
            "chargers": [], "site_chargers": [], "metrics": None,
            "solution_metrics": None}