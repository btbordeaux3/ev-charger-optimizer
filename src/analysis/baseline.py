"""Greedy baselines for before/after comparison.

The point of the before/after is to show what a naive "just build where the
traffic is" planner would pick vs. the optimized plan. Greedy is the natural
baseline: iteratively pick the site that covers the most remaining uncovered
demand, ignoring the duplication penalty and equity weighting.
"""
from __future__ import annotations

import numpy as np

from src.model.coverage import CoverageMatrix

_TYPE_NAMES = ("l2", "dcfc")


def greedy_mclp(
    demand_w: np.ndarray,
    A_l2: np.ndarray,
    A_dcfc: np.ndarray,
    k: int,
    rng=None,
) -> dict:
    """Greedy maximum-coverage heuristic (like Farthest-First / MaxCov).

    At each step pick the (site, type) covering the most remaining uncovered
    demand weight. Ties broken by site index (or random if rng given).
    Returns the same result shape as solve_mclp so downstream code is shared.
    """
    n_sites = A_l2.shape[0]
    n_demand = len(demand_w)
    demand_w = np.asarray(demand_w, dtype=float)
    A_l2 = np.asarray(A_l2, dtype=float)
    A_dcfc = np.asarray(A_dcfc, dtype=float)

    remaining_w = demand_w.copy()
    covered = np.zeros(n_demand, dtype=bool)
    site_types: list[tuple[int, str]] = []
    used_sites = set()
    cover_matrix = np.zeros(n_demand, dtype=float)

    for _ in range(k):
        best = None
        best_gain = -1.0
        for j in range(n_sites):
            if j in used_sites:
                continue
            for t, A in (("l2", A_l2), ("dcfc", A_dcfc)):
                gain = float((A[j] * remaining_w).sum())
                if gain > best_gain + 1e-9:
                    best_gain = gain
                    best = (j, t)
                elif rng is not None and abs(gain - best_gain) < 1e-9 and rng.random() < 0.1:
                    best = (j, t)
        if best is None:
            break
        j, t = best
        A = A_l2 if t == "l2" else A_dcfc
        newly = A[j] > 0
        cover_matrix += A[j]
        remaining_w[newly & ~covered] = 0.0
        covered[newly] = True
        used_sites.add(j)
        site_types.append((j, t))

    return {
        "status": "Greedy",
        "objective": None,
        "site_types": site_types,
        "covered": covered,
        "coverage_multiplicity": cover_matrix,
        "solution_metrics": _metrics(site_types, demand_w, A_l2, A_dcfc),
    }


def _metrics(site_types, demand_w, A_l2, A_dcfc):
    n_demand = len(demand_w)
    cover = np.zeros(n_demand)
    for j, t in site_types:
        A = A_dcfc if t == "dcfc" else A_l2
        cover += A[j].astype(float)
    covered = cover > 0
    weight_total = demand_w.sum() if demand_w.sum() else 1.0
    weight_covered = demand_w[covered].sum()
    return {
        "n_sites": len(site_types),
        "n_demand": n_demand,
        "n_covered_cells": int(covered.sum()),
        "n_uncovered_cells": int((~covered).sum()),
        "frac_cells_covered": float(covered.mean()),
        "weight_covered": float(weight_covered),
        "frac_weight_covered": float(weight_covered / weight_total),
        "duplicated_coverage": float((cover[covered] - 1).sum()),
        "mean_coverage_per_covered": float(cover[covered].mean()) if covered.any() else 0.0,
    }


def greedy_budget(
    demand_w: np.ndarray,
    A_l2: np.ndarray,
    A_dcfc: np.ndarray,
    budget: float,
    cost: dict,
    capacity: dict,
    site_max: int = 12,
    rng=None,
) -> dict:
    """Capacitated greedy baseline for the budget model.

    Repeatedly add the single charger (site, type) with the largest marginal
    increase in demand units served, reusing the same unit-allocation rules as
    the LP: a cell's remaining demand is served by the chargers that cover it,
    each charger's throughput is capped by its capacity. Returns the same shape
    as solve_budget so downstream code is shared.
    """
    n_sites = A_l2.shape[0]
    n_demand = len(demand_w)
    demand_w = np.asarray(demand_w, dtype=float)
    A = np.stack([np.asarray(A_l2, dtype=bool),
                  np.asarray(A_dcfc, dtype=bool)])  # (2, n_sites, n_demand)
    cost_arr = np.array([cost["l2"], cost["dcfc"]], dtype=float)
    cap_arr = np.array([capacity["l2"], capacity["dcfc"]], dtype=float)
    site_max_arr = np.broadcast_to(
        np.asarray(site_max, dtype=float), (n_sites,)
    )

    y = np.zeros((n_sites, 2), dtype=int)
    spent = 0.0
    served = np.zeros(n_demand)
    remaining = demand_w.copy()

    def best_next():
        best = None
        best_gain = -1.0
        for j in range(n_sites):
            if y[j].sum() >= site_max_arr[j]:
                continue
            for t in range(2):
                if spent + cost_arr[t] > budget + 1e-9:
                    continue
                idx = np.where(A[t, j])[0]
                if not idx.size:
                    continue
                # marginal: allocate this charger's capacity to the cells it
                # covers, serving the highest-remaining cells first; each cell
                # can absorb at most its remaining demand
                order = np.argsort(remaining[idx])[::-1]
                used = 0.0
                for k in order:
                    take = min(remaining[idx[k]], cap_arr[t] - used)
                    used += take
                    if used >= cap_arr[t] - 1e-9:
                        break
                total = used
                if total > best_gain + 1e-9:
                    best_gain = total
                    best = (j, t)
                elif rng is not None and abs(total - best_gain) < 1e-9 \
                        and rng.random() < 0.1:
                    best = (j, t)
        return best

    while True:
        pick = best_next()
        if pick is None:
            break
        j, t = pick
        y[j, t] += 1
        spent += cost_arr[t]
        # commit the served units
        idx = np.where(A[t, j])[0]
        order = np.argsort(remaining[idx])[::-1]
        used = 0.0
        for k in order:
            take = min(remaining[idx[k]], cap_arr[t] - used)
            served[idx[k]] += take
            remaining[idx[k]] -= take
            used += take
            if used >= cap_arr[t] - 1e-9:
                break

    chargers = [(j, _TYPE_NAMES[t], int(y[j, t]))
                for j in range(n_sites) for t in range(2) if y[j, t] > 0]
    total_w = demand_w.sum() if demand_w.sum() else 1.0
    served_w = served.sum()
    metrics = {
        "n_demand": n_demand,
        "n_chargers_total": int(y.sum()),
        "budget_spent": spent,
        "n_sites_used": len({j for j, _t, _n in chargers}),
        "demand_units_served": float(served_w),
        "frac_demand_served": float(served_w / total_w),
        "chargers_by_type": {
            "l2": int(y[:, 0].sum()),
            "dcfc": int(y[:, 1].sum()),
        },
    }
    return {
        "status": "Greedy",
        "objective": None,
        "objective_value": float(served_w),
        "site_chargers": chargers,
        "budget_spent": spent,
        "chargers": chargers,
        "served": served,
        "metrics": metrics,
        "solution_metrics": metrics,
    }

