"""Outcome analysis for converged fast-path sessions.

Ported from the replication code of Cheng (2026). Delta (profit gain) is
(mean cycle profit - Nash profit) / (monopoly profit - Nash profit), with unconstrained
static benchmarks, as in Calvano et al. (2020).
"""
from __future__ import annotations

import numpy as np

from .fast import cycle_from, mdp_qstar, policy_values


def analyze(prep: dict, policy, last_state: int, full: bool = True, irf_horizon: int = 15) -> dict:
    """Summarize the greedy cycle reached from last_state; with full=True also equilibrium tests."""
    p = prep
    e = p["e"]
    n = len(e.c)
    states, joints = cycle_from(policy, p["strides"], p["k"], last_state)
    cp = p["profiles"][joints]
    profits = p["rewards"][joints].mean(axis=0)
    F = e.profit(p["profiles"][0]) - p["rewards"][0]
    pn = e.nash()
    pm = e.monopoly()
    pin = e.profit(pn, F).mean()
    pim = e.profit(pm, F).mean()
    out = dict(
        cycle_length=len(states),
        cycle_states=states.tolist(),
        cycle_joint_indices=joints.tolist(),
        cycle_prices=cp.tolist(),
        average_profit=profits.tolist(),
        mean_price=float(cp.mean()),
        delta_gain=float((profits.mean() - pin) / (pim - pin)),
        cs=float(e.cs(cp).mean()),
        total_surplus=float(e.cs(cp).mean() + profits.sum()),
        symmetric_constant=bool(len(states) == 1 and np.all(cp == cp[0, 0])),
        cycle_empty_fallback=bool(p["empty"][:, states].any()),
    )
    fv = p["floor_values"][states]
    out["locked_highest_price"] = bool(
        len(states) == 1 and np.all(policy[:, states[0]] == p["counts"] - 1) and p["empty"][:, states].any()
    )
    out["at_floor_exact"] = bool(np.any(np.isclose(cp, fv, rtol=0, atol=1e-10)))
    out["at_floor_all"] = bool(np.all(np.isclose(cp, fv, rtol=0, atol=1e-10)))
    out["at_lowest_feasible"] = bool(all(policy[i, s] == p["starts"][i, s] for s in states for i in range(n)))
    if not full:
        return out
    _, loss, residual = mdp_qstar(policy, p["rewards"], p["counts"], p["strides"], p["starts"], p["k"], p["delta"])
    out.update(
        path_nash=bool(np.all(loss[:, states] <= 1e-9)),
        qloss_path=float(loss[:, states].mean()),
        qloss_all=float(loss.mean()),
        bellman_residual=float(residual),
    )
    # One-shot deviations evaluated with exact values of the joint greedy policy.
    values = policy_values(policy, p["rewards"], p["strides"], p["k"], p["delta"])
    oneshot_loss = np.zeros_like(values)
    for i in range(n):
        base = np.sum(policy.T.astype(np.int64) * p["strides"], axis=1) - policy[i].astype(np.int64) * p["strides"][i]
        actions = np.arange(p["counts"][i])
        joint = base[:, None] + actions[None, :] * p["strides"][i]
        sn = joint if p["k"] else np.zeros_like(joint)
        q = p["rewards"][joint, i] + p["delta"] * values[i, sn]
        legal = actions[None, :] >= p["starts"][i, :, None]
        best = np.max(np.where(legal, q, -np.inf), axis=1)
        chosen = q[np.arange(len(base)), policy[i]]
        oneshot_loss[i] = np.maximum(best - chosen, 0.0) / np.maximum(np.abs(best), 1e-15)
    out.update(
        nash_onpath_oneshot=bool(np.all(oneshot_loss[:, states] <= 1e-9)),
        qloss_oneshot=float(oneshot_loss[:, states].mean()),
    )
    # Impulse response to the most profitable static deviation, and its discounted payoff.
    irfs = []
    devchanges = []
    no_dev = 0
    for s in states:
        original = policy[:, s].astype(int)
        basejoint = int(original @ p["strides"])
        baseprice = p["profiles"][basejoint]
        for i in range(n):
            base_without = basejoint - original[i] * p["strides"][i]
            feasible = np.arange(p["starts"][i, s], p["counts"][i])
            trials = base_without + feasible * p["strides"][i]
            best = int(feasible[np.argmax(p["rewards"][trials, i])])
            if not np.any(p["prices"][i, feasible] < baseprice[i] - 1e-12):
                no_dev += 1
            joint = int(base_without + best * p["strides"][i])
            sn = joint if p["k"] else 0
            devchanges.append(p["rewards"][joint, i] + p["delta"] * values[i, sn] - values[i, s])
            path = [baseprice.copy(), p["profiles"][joint].copy()]
            cur = sn
            for _ in range(2, irf_horizon + 1):
                j = int(policy[:, cur].astype(int) @ p["strides"])
                path.append(p["profiles"][j].copy())
                cur = j if p["k"] else 0
            path = np.array(path)
            irfs.append(np.column_stack([path[:, i], np.mean(np.delete(path, i, axis=1), axis=1)]))
    dev = np.array(devchanges).reshape(len(states), n)
    out.update(
        irf=np.mean(irfs, axis=0).tolist(),
        deviation_nonprofitable=float(np.mean(dev.mean(axis=0) <= 1e-9)),
        phase_deviation_nonprofitable=float(np.mean(np.array(devchanges) <= 1e-9)),
        no_feasible_downward_deviation=float(no_dev / (len(states) * n)),
        deviation_profit_change=float(np.mean(devchanges)),
    )
    return out
