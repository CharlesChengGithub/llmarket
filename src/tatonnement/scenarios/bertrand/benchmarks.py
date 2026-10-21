"""Static benchmarks under price floors: Nash and monopoly, continuous and on the grid.

Ported from the replication code of Cheng (2026); numerics unchanged.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .demand import Economy, floors_at

_DYNAMIC = ("b1", "b2", "lambda")


def discrete_myopic_attractors(e: Economy, cfg: dict, grid):
    """Attractors of simultaneous grid best responses with lagged floors, from every profile."""
    n = len(e.c)
    m = len(grid)
    N = m**n
    F = cfg.get("F", 0.0)
    actions = np.array(np.unravel_index(np.arange(N), (m,) * n)).T
    profiles = grid[actions]
    profits = e.profit(profiles, F)
    nashq = e.quantity(e.nash())
    typ = cfg.get("type", "none")
    if typ in _DYNAMIC:
        scale = float(cfg.get("lambda", 0.1)) if typ in ("b1", "lambda") else 1.0
        q = np.maximum(e.quantity(profiles), scale * nashq)
        floor = e.c + np.broadcast_to(F, (n,)) / q
    else:
        floor = np.tile(floors_at(e, e.nash(), cfg), (N, 1))
    stride = m ** np.arange(n - 1, -1, -1)
    nxt = np.zeros(N, dtype=np.int64)
    for i in range(n):
        joint = np.arange(N)[:, None] + (np.arange(m)[None, :] - actions[:, i, None]) * stride[i]
        legal = grid[None, :] >= floor[:, i, None]
        legal[~legal.any(axis=1), -1] = True
        nxt += np.argmax(np.where(legal, profits[joint, i], -np.inf), axis=1) * stride[i]
    labels = np.full(N, -1, dtype=int)
    cycles = []
    for initial in range(N):
        if labels[initial] >= 0:
            continue
        seen = {}
        path = []
        s = initial
        while labels[s] < 0 and s not in seen:
            seen[s] = len(path)
            path.append(s)
            s = int(nxt[s])
        if labels[s] >= 0:
            label = int(labels[s])
        else:
            nodes = path[seen[s]:]
            shift = int(np.argmin(nodes))
            nodes = nodes[shift:] + nodes[:shift]
            label = len(cycles)
            cycles.append(nodes)
        labels[path] = label
    out = []
    for label, nodes in enumerate(cycles):
        out.append(
            dict(
                length=len(nodes),
                joint_indices=nodes,
                prices=profiles[nodes].tolist(),
                basin_size=int(np.sum(labels == label)),
                profits=profits[nodes].mean(axis=0).tolist(),
            )
        )
    start = grid[np.argmin(abs(grid[:, None] - e.nash()), axis=0)]
    startid = int(np.searchsorted(grid, start) @ stride)
    return out, int(labels[startid]), profiles, profits, floor


def constrained_benchmarks(e: Economy, cfg: dict, grid) -> dict:
    """Nash and monopoly benchmarks under the floor rule cfg (see :func:`floors_at`)."""
    n = len(e.c)
    typ = cfg.get("type", "none")
    F = np.broadcast_to(cfg.get("F", 0.0), e.c.shape)
    dynamic = typ in _DYNAMIC
    attractions, chosen, profiles, profits, floor = discrete_myopic_attractors(e, cfg, grid)
    # At a steady state, feasibility refers to the prices themselves, not an arbitrary lag.
    legal = np.all(profiles >= floor - 1e-12, axis=1)
    if legal.any():
        dp = profiles[np.argmax(np.where(legal, profits.sum(axis=1), -np.inf))]
    else:
        dp = np.full(n, np.nan)
    pn0 = e.nash()
    pm0 = e.monopoly()
    candidates = []
    traces = []
    if not dynamic:
        f = floors_at(e, pn0, cfg)
        pn = e.nash(f)
        pm = e.monopoly(f)
        status = "fixed_point"
        iterations = 0
        uniqueness = "unique_logit_constant_floor"
        candidates = [pn]
        solver = "scalar_FOC"
        primary_start = pn0
    else:
        # Best-response dynamics from deterministic multi-starts, including the maximum floor.
        scale = float(cfg.get("lambda", 0.1)) if typ in ("b1", "lambda") else 1.0
        cap = e.c + F / (scale * e.quantity(pn0))
        starts = [pn0, pm0, np.full(n, grid[0]), np.full(n, grid[-1]), cap]
        if n == 2:
            axes = [np.unique(np.r_[pn0[i], pm0[i], grid[[0, -1]], cap[i]]) for i in range(n)]
            starts.extend(np.array(np.meshgrid(*axes, indexing="ij")).reshape(n, -1).T)
        for initial in starts:
            p = np.asarray(initial).copy()
            history = []
            tag = "not_converged"
            for it in range(1, 3001):  # noqa: B007  (it is reported after the loop)
                f = floors_at(e, p, cfg)
                new = np.array([e.br(p, i, f[i]) for i in range(n)])
                if max(abs(new - p)) < 1e-10:
                    tag = "fixed_point"
                    p = new
                    break
                if any(max(abs(new - z)) < 1e-10 for z in history[-32:]):
                    tag = "cycle"
                    p = new
                    break
                history.append(p)
                p = new
            if tag == "fixed_point" and not any(max(abs(p - z)) < 1e-7 for z in candidates):
                candidates.append(p.copy())
            traces.append(dict(start=np.asarray(initial).tolist(), status=tag, end=p.tolist(), iterations=it))
        primary = traces[0]
        pn = np.array(primary["end"])
        status = primary["status"]
        iterations = primary["iterations"]
        primary_start = pn0
        if len(candidates) > 1:
            uniqueness = "nonunique_proven_by_distinct_fixed_points"
        else:
            uniqueness = "single_found_multistart_not_global_proof"
        if np.all(pm0 >= floors_at(e, pm0, cfg) - 1e-12):
            pm = pm0
            solver = "unconstrained_global_optimum_feasible"
        else:
            # A local-search bound, not a global certificate.
            options = []
            if np.isfinite(dp).all():
                options.append((float(e.profit(dp, F).sum()), dp.copy(), "discrete_exhaustive"))
            upper = np.maximum.reduce([np.full(n, max(10.0, float(grid[-1]) + 3.0)), cap + 2.0, pm0 + 2.0])
            for initial in [pn0, pm0, np.full(n, grid[-1]), cap, *candidates]:
                sol = minimize(
                    lambda p: -e.profit(p, F).sum(),
                    np.minimum(initial, upper),
                    method="SLSQP",
                    bounds=list(zip(e.c, upper)),
                    constraints=[{"type": "ineq", "fun": lambda p: p - floors_at(e, p, cfg)}],
                    options={"ftol": 1e-12, "maxiter": 2000},
                )
                if sol.success and min(sol.x - floors_at(e, sol.x, cfg)) >= -1e-8:
                    options.append((-float(sol.fun), sol.x, "continuous_multistart_local"))
            if options:
                _, pm, selected = max(options, key=lambda x: x[0])
                solver = "max_discrete_and_local:" + selected
            else:
                pm = np.full(n, np.nan)
                solver = "no_feasible_candidate_found"
    cycle = attractions[chosen]
    nash_profits = e.profit(pn, F).tolist() if status == "fixed_point" else [float("nan")] * n
    inside = [bool(np.all((x >= grid[0] - 1e-9) & (x <= grid[-1] + 1e-9))) for x in candidates]
    return dict(
        nash_prices=pn.tolist(),
        monopoly_prices=pm.tolist(),
        nash_profits=nash_profits,
        monopoly_profits=e.profit(pm, F).tolist(),
        nash_status=status,
        nash_iterations=iterations,
        nash_primary_start=primary_start.tolist(),
        nash_uniqueness=uniqueness,
        nash_fixed_points=[x.tolist() for x in candidates],
        nash_fixed_points_inside_grid=inside,
        nash_multistart_traces=traces,
        discrete_nash_status="fixed_point" if cycle["length"] == 1 else "cycle",
        discrete_myopic_cycle=cycle["prices"],
        discrete_attractors=attractions,
        discrete_unique_fixed_point=bool(len(attractions) == 1 and cycle["length"] == 1),
        discrete_monopoly_prices=dp.tolist(),
        discrete_monopoly_profits=e.profit(dp, F).tolist(),
        monopoly_solver=solver,
    )
